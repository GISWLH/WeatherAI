"""Generate the numerical reference for the GenCast sampler parity test from the OFFICIAL JAX code.

Needs python>=3.12 with: jax haiku xarray gdm-xarray-jax chex dinosaur-dycore fiddle ... and a clone of
google-deepmind/weathernext (path in $WEATHERNEXT_SRC, default /workspace/sources/weathernext).
It runs the official ``dpm_solver_plus_plus_2s.Sampler`` with a closed-form toy denoiser and a
*fixed* noise tensor (the official sampler's RNG is replaced), and writes
tests/models/gencast/data/sampler_ref.npz.
"""
import os
import sys

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import xarray_jax

sys.path.insert(0, os.environ.get("WEATHERNEXT_SRC", "/workspace/sources/weathernext"))
from weathernext.weathernext1_gen import dpm_solver_plus_plus_2s as dpm  # noqa: E402
from weathernext.weathernext1_gen import samplers_utils as su  # noqa: E402

B, H, W = 2, 4, 8
noise = np.random.default_rng(0).standard_normal((B, H, W)).astype(np.float32)
su.spherical_white_noise_like = lambda tmpl: tmpl.map(
    lambda da: xarray_jax.DataArray(jnp.asarray(noise), dims=da.dims)
)


def toy(inputs, noisy_targets, noise_levels, forcings=None):  # D(x,s) = x/(1+s^2) + 0.1 x
    return noisy_targets * (1 / (1 + noise_levels**2)) + 0.1 * noisy_targets


out = {}
for tag, churn in (("churn", 2.5), ("det", 0.0)):
    cfg = dict(max_noise_level=80.0, min_noise_level=0.03, num_noise_levels=6, rho=7.0,
               stochastic_churn_rate=churn, churn_min_noise_level=0.75,
               churn_max_noise_level=float("inf"), noise_level_inflation_factor=1.05)

    def run():
        smp = dpm.Sampler(toy, **cfg)
        tmpl = xarray.Dataset({"v": xarray_jax.Variable(("batch", "lat", "lon"), jnp.zeros((B, H, W), jnp.float32))})
        return smp(xarray.Dataset(), tmpl, None)

    res = hk.transform(run).apply({}, jax.random.key(0))
    out[f"out_{tag}"] = np.asarray(xarray_jax.unwrap_data(res["v"]))
out["noise"] = noise
out["schedule_80_0.03_20_7"] = su.noise_schedule(80.0, 0.03, 20, 7.0)
out["churn_rates"] = su.stochastic_churn_rate_schedule(su.noise_schedule(80.0, 0.03, 20, 7.0), 2.5, 0.75, float("inf"))
dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests/models/gencast/data/sampler_ref.npz")
np.savez(dst, **out)
print("wrote", dst, {k: v.shape for k, v in out.items()})
