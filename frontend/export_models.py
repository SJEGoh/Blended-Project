"""Export the headline networks and the LSM rule to frontend/models.js for the path demo.

Reads the torch checkpoints without torch (they are zip archives of a pickle plus raw float32
storages), so this runs with the standard library only. For each headline setting it exports
  - the adversarial (GDA) hedger and the hedger trained against the LSM rule,
  - the fresh adversary trained against each of them (exploit checkpoints),
  - the market / cost / liquidity settings they were trained under,
plus the fitted LSM exercise rule for lam = 1.

Run from anywhere:  python frontend/export_models.py
"""
import json
import pickle
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
OUT = Path(__file__).resolve().parent / "models.js"

STEM = "lam1_a0p9_s1234_heuristic{suffix}_reduced"
# (id, label, checkpoint suffix) -- same settings as the headline experiments
SCENARIOS = [
    ("liqbase_uwcash", "Liquidity risk + unwind cost", "_liqbase_uwcash"),
    ("c0.005_uwcash", "0.5% cost + unwind cost", "_c0.005_uwcash"),
    ("c0.005", "0.5% transaction cost", "_c0.005"),
    ("frictionless", "Frictionless", ""),
]


class _Storage:
    def __init__(self, key, dtype):
        self.key, self.dtype = key, dtype


class _Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor(storage, offset, size, stride, *args):
    return _Tensor(storage, offset, size, stride)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch" and name == "device":
            return str
        raise pickle.UnpicklingError(f"unexpected global {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, stype, key, _, _ = pid
        dtype = getattr(stype, "dtype", None) or stype
        return _Storage(key, dtype)


def load_checkpoint(path: Path) -> dict:
    """Load a torch.save'd dict; tensors become nested lists of floats."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        root = names[0].split("/")[0]
        blob = _Unpickler(_BytesIO(z.read(f"{root}/data.pkl"))).load()
        cache = {}

        def materialise(t: _Tensor):
            st = t.storage
            if st.dtype not in ("FloatStorage", "float32"):
                raise ValueError(f"unsupported storage {st.dtype}")
            if st.key not in cache:
                raw = z.read(f"{root}/data/{st.key}")
                cache[st.key] = struct.unpack(f"<{len(raw) // 4}f", raw)
            data = cache[st.key]
            if len(t.size) == 0:
                return data[t.offset]
            if len(t.size) == 1:
                return [data[t.offset + i * t.stride[0]] for i in range(t.size[0])]
            if len(t.size) == 2:
                return [[data[t.offset + i * t.stride[0] + j * t.stride[1]] for j in range(t.size[1])]
                        for i in range(t.size[0])]
            raise ValueError(f"unsupported shape {t.size}")

        def walk(x):
            if isinstance(x, _Tensor):
                return materialise(x)
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [walk(v) for v in x]
            return x

        return walk(blob)


def _BytesIO(b):
    import io
    return io.BytesIO(b)


def mlp_layers(state: dict) -> list:
    """nn.Sequential(Linear, SiLU, Linear, SiLU, Linear) state dict -> [{W, b}, ...] in order."""
    idx = sorted({int(k.split(".")[1]) for k in state if k.startswith("net.")})
    rnd = lambda v: [round(x, 7) for x in v]
    return [dict(W=[rnd(row) for row in state[f"net.{i}.weight"]], b=rnd(state[f"net.{i}.bias"])) for i in idx]


def scenario(sid, label, suffix):
    stem = STEM.format(suffix=suffix)
    gda = load_checkpoint(CKPT / f"gda_{stem}.pt")
    lsm = load_checkpoint(CKPT / f"fixed-lsm_{stem}.pt")
    adv_gda = load_checkpoint(CKPT / f"exploit_{stem}.pt")
    adv_lsm = load_checkpoint(CKPT / f"exploit-fixed-lsm_{stem}.pt")
    cfg = gda["cfg"]
    assert not cfg["train"].get("hedger_pnl_input"), "demo assumes the hedger does not see running P&L"
    for blob in (lsm, adv_gda, adv_lsm):           # same market/cost model for every network in a scenario
        for sec in ("market", "costs", "liquidity"):
            assert blob["cfg"].get(sec) == cfg.get(sec), (sid, sec)
    liq = cfg.get("liquidity") or {}
    return dict(
        id=sid, label=label,
        market=cfg["market"], costs=cfg.get("costs") or dict(c=0.0, unwind="none"),
        liquidity=liq if liq.get("enabled") else None,
        nets=dict(
            hedger_gda=mlp_layers(gda["hedger"]),
            hedger_lsm=mlp_layers(lsm["hedger"]),
            adv_vs_gda=mlp_layers(adv_gda["stopper"]),
            adv_vs_lsm=mlp_layers(adv_lsm["stopper"]),
        ),
        sources=dict(hedger_gda=f"gda_{stem}.pt", hedger_lsm=f"fixed-lsm_{stem}.pt",
                     adv_vs_gda=f"exploit_{stem}.pt", adv_vs_lsm=f"exploit-fixed-lsm_{stem}.pt"),
    )


def main():
    bm = json.loads((ROOT / "results" / "benchmarks_lam1.json").read_text())
    data = dict(
        scenarios=[scenario(*s) for s in SCENARIOS],
        lsm_rule=bm["lsm_rule"],
        lsm_price=bm["lsm_price"], merton_european=bm["merton_european"],
    )
    js = "// Generated by frontend/export_models.py from checkpoints/ - do not edit by hand.\n"
    js += "window.DEMO_MODELS = " + json.dumps(data, separators=(",", ":")) + ";\n"
    OUT.write_text(js)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(js) / 1024:.0f} KB, {len(data['scenarios'])} scenarios)")


if __name__ == "__main__":
    main()
