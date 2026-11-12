import torch
import torch.nn as nn
import torch.nn.functional as F

from sum_model import SumModel

BIDIRECTIONAL = ("prefixlm", "encdec")


class PCRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.sm = SumModel(self.cfg)
        self.name = self.sm.name
        self._drop_unused()
        default_pool = "mean" if self.name in BIDIRECTIONAL else "mean_last"
        self.pool = self.cfg.get("pool", default_pool)
        if self.pool not in ("mean", "last", "first", "max", "mean_last"):
            raise ValueError(f"pool must be mean | last | first | max | mean_last, got {self.pool!r}")
        self.skip_prompt = bool(self.cfg.get("skip_prompt", True))
        d = self.sm.d * (2 if self.pool == "mean_last" else 1)
        p = int(self.cfg.get("proj_dim", 0) or 0)
        self.proj = None
        if p:
            self.proj = nn.Linear(d, p, bias=False)
            nn.init.orthogonal_(self.proj.weight)
        self.out_dim = p or d

    def _drop_unused(self):
        bb = self.sm.backbone
        if self.name == "encdec":
            bb.decoder = None
            bb.dec_ln = None
            self.sm.use_decoder = False
        hn = self.sm.head_name
        if hn is not None:
            head = getattr(bb, hn)
            if head.weight is not self.sm.emb.weight:
                for q in head.parameters():
                    q.requires_grad = False

    def states(self, ids):
        sm = self.sm
        x = sm._embed(ids)
        T = x.shape[1]
        sm._check_len(T, "prompt + chunk")
        if self.name == "prefixlm":
            h = sm._states_prefixlm(x, T)
        elif self.name == "encdec":
            h = sm._encode(x)
        elif self.name == "arch4":
            h = sm._states_arch4(x)
        else:
            h = sm._states_biscale(x)
        if self.skip_prompt and sm.n_prompt:
            h = h[:, sm.n_prompt:]
        return h

    def embed(self, ids):
        h = self.states(ids)
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            if self.pool == "mean":
                v = h.mean(1)
            elif self.pool == "last":
                v = h[:, -1]
            elif self.pool == "first":
                v = h[:, 0]
            elif self.pool == "max":
                v = h.amax(1)
            else:
                v = torch.cat([h.mean(1), h[:, -1]], dim=-1)
            if self.proj is not None:
                v = self.proj(v)
        return v

    def forward(self, ids):
        return self.embed(ids)

    def param_groups(self, base_lr):
        groups = self.sm.param_groups(base_lr)
        if self.proj is not None:
            groups.append({"params": list(self.proj.parameters()),
                           "lr": base_lr * float(self.cfg.get("proj_lr_mult", 1.0))})
        return groups


def build_pcr_model(model_cfg):
    return PCRModel(model_cfg)


@torch.no_grad()
def check_pcr_model(model_cfg, L=256, B=2, seed=0):
    m = PCRModel(model_cfg).eval()
    dev = m.sm.emb.weight.device
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, m.sm.vocab, (B, L), generator=g).to(dev)
    v = m.embed(ids)
    n_all = sum(q.numel() for q in m.parameters())
    n_tr = sum(q.numel() for q in m.parameters() if q.requires_grad)
    print(f"{model_cfg.get('phase1')}/{model_cfg.get('method')} pool={m.pool}: embed {tuple(v.shape)} | "
          f"trainable {n_tr/1e6:.3f}M of {n_all/1e6:.2f}M ({100*n_tr/n_all:.2f}%)")
    if m.name in BIDIRECTIONAL or m.pool in ("last", "mean_last"):
        ids2 = ids.clone()
        ids2[:, -1] = (ids2[:, -1] + 1) % m.sm.vocab
        moved = (m.embed(ids2) - v).abs().max().item()
        print(f"  changing the last token moves the vector by {moved:.2e} (expect > 0)")
    return v
