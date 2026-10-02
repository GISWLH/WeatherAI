"""ACE2-ERA5 stepper: normalise -> SFNO -> denormalise -> conservation correctors -> prescribed ocean SST (native PyTorch)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .sfno import SFNO, SFNOConfig

GRAVITY = 9.80665
LATENT_HEAT_OF_VAPORIZATION = 2.5e6


def _levels(prefix: str, n: int):
    return [f"{prefix}_{i}" for i in range(n)]


@dataclass
class ACE2Config:
    sfno: SFNOConfig = field(default_factory=SFNOConfig)
    in_names: tuple = ()
    out_names: tuple = ()
    n_levels: int = 8
    force_positive: tuple = ()
    conserve_dry_air: bool = True
    moisture_budget: str | None = "advection_and_precipitation"   # modifies advection and precipitation
    timestep_seconds: float = 6 * 3600.0
    next_step_forcing: tuple = ("DSWRFtoa",)
    sst_name: str = "surface_temperature"
    ocean_fraction_name: str = "ocean_fraction"
    water: str = "specific_total_water"
    pressure: str = "PRESsfc"
    latent_flux: str = "LHTFLsfc"
    precip: str = "PRATEsfc"
    advection: str = "tendency_of_total_water_path_due_to_advection"


class ACE2(nn.Module):
    """Holds the SFNO plus per-variable statistics, grid areas and hybrid-sigma coefficients.

    ``step`` takes physical-unit dicts of [B, H, W] tensors and returns the next state (all ``out_names``),
    following the sequence ``fme`` runs in ``step_with_adjustments`` for this checkpoint.
    """

    def __init__(self, cfg: ACE2Config, means: dict, stds: dict, area: torch.Tensor, ak: torch.Tensor, bk: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.net = SFNO(cfg.sfno)

        def vec(d, names):
            return torch.tensor([d[n] for n in names], dtype=torch.float32).view(1, -1, 1, 1)

        self.register_buffer("in_mean", vec(means, cfg.in_names))
        self.register_buffer("in_std", vec(stds, cfg.in_names))
        self.register_buffer("out_mean", vec(means, cfg.out_names))
        self.register_buffer("out_std", vec(stds, cfg.out_names))
        self.register_buffer("area", area.float())
        self.register_buffer("ak", ak.float())
        self.register_buffer("bk", bk.float())
        self.prognostic = tuple(n for n in cfg.in_names if n in cfg.out_names)
        self.forcing_names = tuple(n for n in cfg.in_names if n not in cfg.out_names)

    def area_mean(self, x):
        a = self.area.to(x.dtype)
        return (x * a).sum(dim=(-2, -1), keepdim=True) / a.sum()

    def _water(self, d):
        return torch.stack([d[n] for n in _levels(self.cfg.water, self.cfg.n_levels)], dim=-1)

    def total_water_path(self, d):
        ps = d[self.cfg.pressure]
        p_if = torch.stack([a + b * ps for a, b in zip(self.ak, self.bk)], dim=-1)
        return (self._water(d) * p_if.diff(dim=-1)).sum(-1) / GRAVITY

    def dry_air_pressure(self, d):
        return d[self.cfg.pressure] - GRAVITY * self.total_water_path(d)

    def global_dry_air(self, d):
        return self.area_mean(self.dry_air_pressure(d).double())

    # ---- correctors (same order as the checkpoint's corrector config)
    def _force_positive(self, gen):
        gen = dict(gen)
        for n in self.cfg.force_positive:
            gen[n] = gen[n].clamp(min=0.0)
        return gen

    def _conserve_dry_air(self, gen, target):
        dry = self.dry_air_pressure(gen).double()
        new_dry = dry - (self.area_mean(dry) - target.double())
        wat = self._water(gen).double()
        ak_d, bk_d = self.ak.double().diff(), self.bk.double().diff()
        new_ps = (new_dry + (ak_d * wat).sum(-1)) / (1 - (bk_d * wat).sum(-1))
        gen = dict(gen)
        gen[self.cfg.pressure] = new_ps.to(gen[self.cfg.pressure].dtype)
        return gen

    def _moisture_budget(self, inp, gen):
        c = self.cfg
        tend = (self.total_water_path(gen) - self.total_water_path(inp)) / c.timestep_seconds
        tend_gm = self.area_mean(tend)
        evap = gen[c.latent_flux] / LATENT_HEAT_OF_VAPORIZATION
        evap_gm = self.area_mean(evap)
        precip = gen[c.precip]
        precip_gm = self.area_mean(precip)
        gen = dict(gen)
        if c.moisture_budget.endswith("precipitation"):
            precip = precip * ((evap_gm - tend_gm) / precip_gm)
            gen[c.precip] = precip
        elif c.moisture_budget.endswith("evaporation"):
            evap = evap * ((tend_gm + precip_gm) / evap_gm)
            gen[c.latent_flux] = evap * LATENT_HEAT_OF_VAPORIZATION
        if c.moisture_budget.startswith("advection"):
            gen[c.advection] = tend - (evap - precip)
        return gen

    # ---- one step / rollout
    def network(self, inp: dict) -> dict:
        x = torch.stack([inp[n] for n in self.cfg.in_names], dim=1)
        y = self.net((x - self.in_mean) / self.in_std)
        y = y * self.out_std + self.out_mean
        return {n: y[:, i] for i, n in enumerate(self.cfg.out_names)}

    def step(self, state: dict, forcing: dict, next_forcing: dict, dry_air_target=None):
        """``forcing``: input-only variables for this step (``next_step_forcing`` names already taken at t+1 by the caller);
        ``next_forcing``: ocean_fraction and surface_temperature at t+1. Returns (next_state, dry_air_target)."""
        c = self.cfg
        inp = {**state, **forcing}
        gen = self._force_positive(self.network(inp))
        if c.conserve_dry_air:
            if dry_air_target is None:
                dry_air_target = self.global_dry_air(inp)
            gen = self._conserve_dry_air(gen, dry_air_target)
        if c.moisture_budget is not None:
            gen = self._moisture_budget(inp, gen)
        ocean = torch.round(next_forcing[c.ocean_fraction_name]) == 1
        gen[c.sst_name] = torch.where(ocean, next_forcing[c.sst_name], gen[c.sst_name])
        return gen, dry_air_target

    def rollout(self, ic: dict, forcing: dict, n_steps: int, return_every: int = 1):
        """``ic``: prognostic fields [B,H,W] at t0. ``forcing``: input-only fields + ocean_fraction + surface_temperature,
        each [B, n_steps+1, H, W] for t0..t0+n_steps. Returns a list of per-step output dicts."""
        c = self.cfg
        state = {k: ic[k] for k in self.prognostic}
        target, outs = None, []
        with torch.no_grad():
            for s in range(n_steps):
                f = {k: forcing[k][:, s + 1 if k in c.next_step_forcing else s] for k in self.forcing_names}
                nf = {k: forcing[k][:, s + 1] for k in (c.ocean_fraction_name, c.sst_name)}
                gen, target = self.step(state, f, nf, target)
                state = {k: gen[k] for k in self.prognostic}
                if (s + 1) % return_every == 0:
                    outs.append(gen)
        return outs
