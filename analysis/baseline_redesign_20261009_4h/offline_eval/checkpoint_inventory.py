"""CPU-only integrity inventory of all recovered B1 checkpoints, without inference."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RECOVERED = ROOT / "analysis/baseline_redesign_20261009_4h/recovered_full_20261009_1427/b1_baseline_20261009_4h"
EXPECTED_ITERATIONS = [0] + list(range(100, 1500, 100)) + [1499]
EXPECTED_LAYERS = [(512, 235), (256, 512), (128, 256), (12, 128)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensors(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from tensors(v, prefix + "/" + str(k))
    elif isinstance(value, (list, tuple)):
        for k, v in enumerate(value):
            yield from tensors(v, prefix + "/" + str(k))


def state_sha(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise ValueError("Non-tensor actor state: " + key)
        header = json.dumps([key, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        h.update(len(header).to_bytes(8, "big")); h.update(header)
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(len(raw).to_bytes(8, "big")); h.update(raw)
    return h.hexdigest()


def inspect(path, manifest):
    actual_sha = sha(path)
    relative = path.relative_to(RECOVERED).as_posix()
    expected = manifest["files"].get(relative)
    if not expected:
        raise ValueError("Checkpoint absent from remote archive manifest: " + relative)
    # Verify bytes before restricted deserialization; retain the original unchanged.
    if actual_sha != expected["sha256"] or path.stat().st_size != expected["bytes"]:
        raise ValueError("Manifest hash/size mismatch: " + relative)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict):
        raise ValueError("Checkpoint not a dictionary")
    state_key = "actor_state_dict" if "actor_state_dict" in ck else "model_state_dict"
    actor = ck[state_key]
    weights = [key for key, value in actor.items() if key.endswith("weight") and value.ndim == 2 and "critic" not in key]
    shapes = [tuple(actor[key].shape) for key in weights]
    floating = [(key, value) for key, value in tensors(ck) if value.is_floating_point() or value.is_complex()]
    nonfinite = [key for key, value in floating if not bool(torch.isfinite(value).all())]
    non_cpu = [key for key, value in tensors(ck) if value.device.type != "cpu"]
    filename_iteration = int(re.fullmatch(r"model_(\d+)\.pt", path.name).group(1))
    actual_iteration = ck.get("iter")
    if torch.is_tensor(actual_iteration):
        actual_iteration = actual_iteration.item()
    std = {}
    for key, value in actor.items():
        if "std" in key.lower():
            values = value.detach().cpu().double().reshape(-1)
            std[key] = dict(shape=list(value.shape), values=values.tolist(),
                            min=float(values.min()), mean=float(values.mean()), max=float(values.max()),
                            finite=bool(torch.isfinite(values).all()), positive=bool((values > 0).all()),
                            meaning="stored Gaussian exploration parameter; not a deterministic performance metric")
    biases = {key: list(value.shape) for key, value in actor.items() if key.endswith("bias")}
    actor_hash = state_sha(actor)
    mlp_state = {key: value for key, value in actor.items() if "std" not in key.lower()}
    mlp_hash = state_sha(mlp_state)
    current_after = sha(path)
    return dict(path=str(path.resolve()), relative_path=relative, file_sha256=actual_sha,
        bytes=path.stat().st_size, manifest_sha256=expected["sha256"], manifest_hash_matches=True,
        manifest_size_matches=True, unchanged_after_cpu_load=current_after == actual_sha,
        filename_iteration=filename_iteration, stored_iteration=actual_iteration,
        iteration_matches_filename=actual_iteration == filename_iteration,
        checkpoint_keys=list(ck), actor_state_key=state_key, actor_state_sha256=actor_hash,
        deterministic_actor_MLP_sha256=mlp_hash,
        actor_layers=[list(s) for s in shapes], actor_input_output_dimensions=[235,512,256,128,12],
        actor_structure_matches=shapes == EXPECTED_LAYERS, actor_weight_keys=weights,
        actor_bias_shapes=biases, actor_tensor_count=len(actor),
        floating_tensor_count_all_checkpoint_sections=len(floating),
        all_floating_tensors_finite=not nonfinite, nonfinite_tensor_paths=nonfinite,
        all_loaded_tensors_on_CPU=not non_cpu, non_CPU_tensor_paths=non_cpu,
        gaussian_std_parameters=std, policy_inference_performed=False,
        physical_evaluation_performed=False)


def main():
    torch.set_num_threads(2)
    manifest_path = RECOVERED / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = sorted((RECOVERED / "preserved/B1_selective_cost").rglob("model_*.pt"),
                   key=lambda p: int(re.fullmatch(r"model_(\d+)\.pt", p.name).group(1)))
    observed = [int(re.fullmatch(r"model_(\d+)\.pt", p.name).group(1)) for p in paths]
    if observed != EXPECTED_ITERATIONS:
        raise ValueError("Checkpoint coverage differs: " + str(observed))
    rows = [inspect(path, manifest) for path in paths]
    checks = dict(exact_16_checkpoint_coverage=len(rows)==16,
        all_remote_manifest_hashes_match=all(r["manifest_hash_matches"] for r in rows),
        all_remote_manifest_sizes_match=all(r["manifest_size_matches"] for r in rows),
        all_unchanged_after_cpu_deserialization=all(r["unchanged_after_cpu_load"] for r in rows),
        all_iteration_numbers_match=all(r["iteration_matches_filename"] for r in rows),
        all_actor_layers_match=all(r["actor_structure_matches"] for r in rows),
        all_floating_tensors_finite=all(r["all_floating_tensors_finite"] for r in rows),
        all_loaded_tensors_on_CPU=all(r["all_loaded_tensors_on_CPU"] for r in rows),
        all_std_parameters_finite_positive=all(s["finite"] and s["positive"] for r in rows for s in r["gaussian_std_parameters"].values()),
        distinct_actor_states_across_16_saves=len({r["actor_state_sha256"] for r in rows})==16,
        distinct_deterministic_actor_MLPs_across_16_saves=len({r["deterministic_actor_MLP_sha256"] for r in rows})==16)
    report = dict(status="CHECKPOINT_INTEGRITY_PASS" if all(checks.values()) else "CHECKPOINT_INTEGRITY_FLAG",
        scope="CPU-only weights_only=True; no policy inference/simulation/training/server actions",
        python=sys.executable, python_version=sys.version.split()[0], local_torch=torch.__version__,
        checkpoint_count=len(rows), expected_iterations=EXPECTED_ITERATIONS, checks=checks,
        archive_manifest_path=str(manifest_path.resolve()), archive_manifest_sha256=sha(manifest_path),
        checkpoints=rows, closed_loop_quality_measured=False,
        candidate_Q3=None, candidate_official_score_prediction=None,
        limitation="Integrity and changing actor parameters establish recoverable model files and training updates only; they do not establish survival, tracking performance, optimal checkpoint or score improvement.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"checkpoint_inventory.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False), encoding="utf-8")
    lines=["B1 전체 저장본 로컬 CPU 무결성 검사", "", f"결과: {report['status']}; 저장본 {len(rows)}개", ""]
    lines.extend(f"{key}: {value}" for key,value in checks.items())
    lines.extend(["", "반복 | 바이트 SHA 앞12자리 | actor 상태 SHA 앞12자리 | 학습 탐색 std 평균", ""])
    for row in rows:
        std_means = [s["mean"] for s in row["gaussian_std_parameters"].values()]
        lines.append(f"{row['stored_iteration']:4d} | {row['file_sha256'][:12]} | {row['actor_state_sha256'][:12]} | {','.join(f'{v:.9f}' for v in std_means)}")
    lines.extend(["", "torch.load(weights_only=True,map_location='cpu')로만 읽었습니다.",
                  "원본 파일을 변경하지 않았으며, 네트워크 추론/시뮬레이션/학습을 수행하지 않았습니다.",
                  "모든 값이 정상이고 actor가 변했다고 해도 낙상·추종 오차·점수 개선은 입증되지 않습니다.",
                  "1499와1000 중 성능이 더 좋은 저장본은 아직 선택할 근거가 없습니다."])
    (OUT/"checkpoint_inventory.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"checkpoint_count":len(rows),"checks":checks,"final_std":rows[-1]["gaussian_std_parameters"]},indent=2))


if __name__=="__main__":
    main()
