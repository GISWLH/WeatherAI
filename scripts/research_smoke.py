"""Three-paper smoke checks: synthetic inputs, no pretrained/skill claims."""
import json
import time
import torch


def run(device="cpu"):
    from weatherai import PBC_lite, GenFocal_lite, Translocation_lite
    from weatherai.models.genfocal import reflow_loss
    torch.manual_seed(19)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats()
    report = {"device": device, "torch": torch.__version__, "pretrained": False,
              "input_data": "synthetic", "forecast_skill_evaluated": False, "models": {}}
    if device.startswith("cuda"):
        report["gpu"] = torch.cuda.get_device_name()
    start = time.perf_counter()
    args_cpu = [torch.rand(64, 8, 16, 4) for _ in range(4)]
    obs_cpu = (torch.rand_like(args_cpu[0]) > .5).float()
    cpu = PBC_lite().fit(args_cpu[0], obs_cpu, *args_cpu[1:])
    reference = cpu(*args_cpu)
    args = [x.to(device) for x in args_cpu]
    pbc = PBC_lite().to(device).fit(args[0], obs_cpu.to(device), *args[1:])
    probs = pbc.predict_proba(*args)
    error = float((pbc(*args).cpu() - reference).abs().max())
    report["models"]["pbc"] = {"finite": bool(torch.isfinite(probs).all()), "valid_probabilities": bool(
        (probs >= -1e-6).all() and torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]), atol=1e-6)),
        "max_abs_vs_cpu": error, "cpu_agreement": error < 1e-4}
    gen = GenFocal_lite(2).to(device)
    x = torch.randn(1, 2, 8, 8, 2, device=device)
    cond = {k: torch.randn_like(x) for k in gen.debiasing.cfg.cond_keys}
    loss = reflow_loss(gen.debiasing, x, torch.randn_like(x), cond)
    loss = loss + gen.super_resolution.loss(torch.randn_like(x), x, torch.ones(1, device=device))
    loss.backward()
    grads = [p.grad for p in gen.parameters() if p.grad is not None]
    gen.eval()
    out = gen.downscale(x, cond, output_size=(4, 16, 16), flow_steps=3, diffusion_steps=4)
    report["models"]["genfocal"] = {"finite": bool(torch.isfinite(out["output"]).all()),
        "two_stage_backward": bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),
        "output_shape": list(out["output"].shape), "parameters": sum(p.numel() for p in gen.parameters()),
        "scope": "approximate two-stage U-ViT, ReFlow + EDM Heun"}
    trans = Translocation_lite().to(device)
    field = torch.randn(1, 4, 8, 16, device=device)
    erf = trans.effective_receptive_field(field, location=(3, 4))
    report["models"]["translocation"] = {"finite": bool(torch.isfinite(erf["gradient"]).all()),
        "nonzero_erf": bool(erf["gradient"].abs().sum() > 0), "gradient_shape": list(erf["gradient"].shape),
        "scope": "GraphCast lite signed ERF; not a new forecasting architecture"}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        report["peak_gpu_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
    report["seconds"] = time.perf_counter() - start
    checks = [v for model in report["models"].values() for v in model.values() if isinstance(v, bool)]
    report["status"] = "PASS" if all(checks) else "FAIL"
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.device), indent=2))
