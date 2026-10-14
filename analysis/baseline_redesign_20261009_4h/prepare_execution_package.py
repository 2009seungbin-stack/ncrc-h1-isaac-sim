"""Build authorized B1 package locally. No server access or checkpoint loading."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
OLD = WS / "analysis/high_score_search_20261009/execution"
DEST = ROOT / "execution/package"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = {
        "design.json": ROOT / "design.json",
        "decision_rules.py": ROOT / "decision_rules.py",
        "candidates/B1_selective_cost_rewards.py": ROOT / "candidates/B1_selective_cost_rewards.py",
        "fresh_rewards_original.py": WS / "analysis/short_sweep_20261002/recovered_20261003/short_sweep_20261002/fresh_rewards_original.py",
        "previous_source_hashes_20261002.json": OLD / "previous_source_hashes_20261002.json",
        "reference_models.json": OLD / "reference_models.json",
        "source_snapshot.py": OLD / "source_snapshot.py",
        "t_eval.py": OLD / "t_eval.py",
        "t_eval_go2_core.py": OLD / "t_eval_go2_core.py",
        "references/C0_5500.pt": OLD / "references/C0_5500.pt",
        "references/F10_3000.pt": OLD / "references/F10_3000.pt",
        "runner.py": ROOT / "execution/runner.py",
        "evaluation.py": ROOT / "execution/evaluation.py",
        "live_preflight.py": ROOT / "execution/live_preflight.py",
    }
    for name, src in sources.items():
        if not src.is_file():
            raise FileNotFoundError(f"Package input incomplete: {name}")
    DEST.mkdir(parents=True, exist_ok=True)
    for name, src in sources.items():
        dst = DEST / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert sha(src) == sha(dst)
    expected = json.loads((DEST / "reference_models.json").read_text(encoding="utf-8-sig"))
    for ref in expected:
        assert sha(DEST / ref["checkpoint"]) == ref["sha256"]
    assert sha(DEST / "fresh_rewards_original.py") == "09630e6f548b16a2fa0a81eef73aa393cf5f12eadccd0bad8a144b924e8c1be7"
    # Binary archive recovery smoke test; text clipboard alone is not a pass.
    probe = DEST / "recovery_probe.zip"
    with zipfile.ZipFile(probe, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("probe.bin", bytes(range(256))*16)
        z.writestr("purpose.txt", "B1 binary recovery smoke test; no training result.\n")
    entries = {name: {"bytes": (DEST/name).stat().st_size, "sha256": sha(DEST/name)}
               for name in [*sources, "recovery_probe.zip"]}
    manifest = {"scope": "B1_only_scratch1500_6144_seed42_total4h", "files": entries,
                "server_started_by_this_script": False, "checkpoint_deserialization": False}
    (DEST / "package_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    archive = ROOT / "execution/b1_baseline_20261009_4h_package.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in [*entries, "package_manifest.json"]:
            z.write(DEST/name, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, spec in entries.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == spec["sha256"]
    report = {"package_path": str(archive), "bytes": archive.stat().st_size, "sha256": sha(archive),
              "probe": entries["recovery_probe.zip"], "member_hashes_verified": True,
              "source_files": entries}
    (ROOT / "execution/package_verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("package_path", "bytes", "sha256", "probe")}, indent=2))


if __name__ == "__main__":
    main()
