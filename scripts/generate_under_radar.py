from pathlib import Path
import copy
import json
import shutil

SRC_ROOT = Path("nibi_configs/final_good_small_ranges")
DST_ROOT = Path("nibi_configs/final_good_under_radar")

SOURCE_MIN = 0.5
SOURCE_MAX = 1.5

NEW_MIN = 0.1
NEW_MAX = 0.5


def replace_strings(obj, old, new):
    if isinstance(obj, dict):
        return {
            k: replace_strings(v, old, new)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            replace_strings(v, old, new)
            for v in obj
        ]

    if isinstance(obj, str):
        return obj.replace(old, new)

    return obj


if not SRC_ROOT.exists():
    raise RuntimeError(
        f"Missing source directory: {SRC_ROOT}"
    )

# Clean only the new experiment directory so rerunning this
# generator cannot leave duplicate/stale under-radar configs.
if DST_ROOT.exists():
    shutil.rmtree(DST_ROOT)

created = []

for src in sorted(SRC_ROOT.rglob("*.json")):

    with src.open() as f:
        cfg = json.load(f)

    attack_cfg = cfg.get("attack", {})

    try:
        mn = float(attack_cfg["offset_distance_min"])
        mx = float(attack_cfg["offset_distance_max"])
    except (KeyError, TypeError, ValueError):
        continue

    # Clone only the already-validated 0.5-1.5 m experiments.
    if abs(mn - SOURCE_MIN) > 1e-9:
        continue

    if abs(mx - SOURCE_MAX) > 1e-9:
        continue

    parts = src.parts

    if "rpo" in parts:
        attack = "rpo"
    elif "cpo" in parts:
        attack = "cpo"
    else:
        raise RuntimeError(
            f"Cannot determine attack type from path: {src}"
        )

    if "core" in parts:
        variant = "core"
    elif "id" in parts:
        variant = "id"
    else:
        raise RuntimeError(
            f"Cannot determine ID variant from path: {src}"
        )

    cfg = copy.deepcopy(cfg)

    # Change ONLY attack magnitude.
    cfg["attack"]["offset_distance_min"] = NEW_MIN
    cfg["attack"]["offset_distance_max"] = NEW_MAX

    # Give the new experiment a unique identity.
    old_name = cfg.get("pipeline_name", src.stem)

    new_name = (
        f"under_radar_{attack}_0p1_0p5_"
        f"{variant}_{src.stem}"
    )

    cfg = replace_strings(
        cfg,
        old_name,
        new_name,
    )

    cfg["pipeline_name"] = new_name

    # Force independent cache identity.
    cfg.setdefault("cache", {})
    cfg["cache"]["version"] = (
        "under-radar-noise-floor-0p1-0p5-v1"
    )

    out_dir = DST_ROOT / attack / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{new_name}.json"

    with out_path.open("w") as f:
        json.dump(cfg, f, indent=2)

    created.append(out_path)


rpo_core = [
    p for p in created
    if "/rpo/core/" in str(p)
]

rpo_id = [
    p for p in created
    if "/rpo/id/" in str(p)
]

cpo_core = [
    p for p in created
    if "/cpo/core/" in str(p)
]

cpo_id = [
    p for p in created
    if "/cpo/id/" in str(p)
]

print("===== CREATED CONFIGS =====")
print("RPO no ID:  ", len(rpo_core))
print("RPO with ID:", len(rpo_id))
print("CPO no ID:  ", len(cpo_core))
print("CPO with ID:", len(cpo_id))
print("TOTAL:      ", len(created))

assert len(rpo_core) == 12, len(rpo_core)
assert len(rpo_id) == 12, len(rpo_id)
assert len(cpo_core) == 12, len(cpo_core)
assert len(cpo_id) == 12, len(cpo_id)
assert len(created) == 48, len(created)

rpo = sorted(rpo_core + rpo_id)
cpo = sorted(cpo_core + cpo_id)
all_configs = sorted(created)

Path("under_radar_rpo_configs.txt").write_text(
    "\n".join(str(p) for p in rpo) + "\n"
)

Path("under_radar_cpo_configs.txt").write_text(
    "\n".join(str(p) for p in cpo) + "\n"
)

Path("under_radar_all_configs.txt").write_text(
    "\n".join(str(p) for p in all_configs) + "\n"
)

print()
print("SUCCESS")
print("Created under_radar_all_configs.txt")
print("48 configs × 3 classifiers = 144 evaluations")
