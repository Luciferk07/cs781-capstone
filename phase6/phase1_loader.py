"""phase1_loader.py — load a Phase-1 model with its trained weights."""
import importlib, json, os

import torch

REGISTRY = os.environ.get("PHASE1_REGISTRY", "phase1_registry.json")


def load_phase1(name, map_location="cpu"):
    reg = json.load(open(REGISTRY))
    assert name in reg, f"{name!r} not in {REGISTRY}: {sorted(reg)}"
    r = reg[name]
    mod = importlib.import_module(r["module"])
    cfg = dict(r["model_cfg"])
    model = mod.build_model(cfg)
    ck = torch.load(r["ckpt"], map_location=map_location, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"{name}: missing {missing[:5]} unexpected {unexpected[:5]}"
    val = ck.get("val_loss") if isinstance(ck, dict) else None
    return model, dict(model_cfg=cfg, ckpt=r["ckpt"], module=r["module"], val_loss=r.get("val_loss", val))