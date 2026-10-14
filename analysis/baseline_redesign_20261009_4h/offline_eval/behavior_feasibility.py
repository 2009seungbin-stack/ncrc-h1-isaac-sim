"""Local-only audit of whether recorded trajectories support B1 actor replay.

No simulation, installation, network access, checkpoint deserialization, or
synthetic observation inference. Historical kinematics are not substituted for
the actual noisy policy observation including height scans.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
HISTORY = ROOT / "analysis/short_sweep_20261002/recovered_20261003/short_sweep_20261002/results"
B1 = ROOT / "analysis/baseline_redesign_20261009_4h/recovered_full_20261009_1427/b1_baseline_20261009_4h"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_trace(path):
    arrays = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            row = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if value.dtype.kind in "bifu":
                row["finite"] = bool(np.isfinite(value).all())
            elif value.ndim == 0 and value.dtype.kind in "US":
                row["text"] = str(value.item())
            arrays[key] = row
    observation_keys = [k for k, row in arrays.items() if any(n in k.lower() for n in ("observation", "policy_obs", "actor_obs"))]
    candidate_235d = [k for k, row in arrays.items() if row["shape"] and row["shape"][-1] == 235]
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path),
            "arrays": arrays, "explicit_observation_keys": observation_keys,
            "arrays_with_last_dimension_235": candidate_235d,
            "sufficient_actor_input": bool(observation_keys and candidate_235d)}


def main():
    paths = sorted(p for p in HISTORY.glob("heldout_*/traces/*.npz")
                   if p.name.startswith(("C0_5500_", "F10_3000_")))
    traces = [inspect_trace(p) for p in paths]
    latest_npz = sorted(B1.rglob("*.npz"))
    latest = [inspect_trace(p) for p in latest_npz]
    checkpoints = sorted((B1 / "preserved/B1_selective_cost").rglob("model_1499.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError("Expected exactly one full-recovery B1 final checkpoint")
    ck = checkpoints[0]
    modules = {}
    for name in ("numpy", "torch", "isaaclab", "isaacsim"):
        spec = importlib.util.find_spec(name)
        modules[name] = str(spec.origin) if spec else None
    known_locations = [Path("D:/IsaacSim"), Path("C:/isaacsim"), ROOT / "IsaacLab"]
    recorded_observations = [r for r in traces + latest if r["sufficient_actor_input"]]
    report = dict(status="OFFLINE_BEHAVIOR_REPLAY_UNAVAILABLE", scope="local_only_no_server_no_simulation",
        python=sys.executable, python_version=sys.version.split()[0], torch_installed_version=importlib.metadata.version("torch"),
        modules=modules, known_install_paths=[{"path": str(p), "exists": p.exists()} for p in known_locations],
        local_simulation_note="No isaaclab/isaacsim module in this Python and no checked standard directory; no exhaustive machine-wide search and no runtime installed",
        B1_final_checkpoint={"path": str(ck.resolve()), "sha256": sha(ck), "bytes": ck.stat().st_size,
                             "deserialized_by_this_audit": False},
        historical_heldout_trace_count=len(traces), B1_recorded_trace_count=len(latest),
        usable_recorded_actor_observation_trace_count=len(recorded_observations), traces=traces, B1_traces=latest,
        missing_required_actor_state=["projected gravity", "joint positions relative to default", "height scan (187 values)",
                                     "the actual observation noise draw", "full policy observation order/scales/clipping verification"],
        available_historical_state="command3, body_lin3, body_ang3, action12, joint_vel12, joint_acc12, torque12, position3 (60 columns)",
        strongest_valid_conclusion="Historical traces can remeasure C0/F10 outcomes and diagnostics only. They cannot identify B1 closed-loop trajectories, falls, tracking errors, Q3 or official score.",
        rejected_methods=["Fill missing height/gravity/joint/noise with arbitrary zeros and call it an evaluation",
                          "Apply B1 actions to recorded reference velocities without stepping dynamics",
                          "Convert parameter distance/action differences into survival or points"],
        performed_actor_inference=False, B1_closed_loop_evaluation=False, B1_internal_Q3=None,
        B1_predicted_official_score=None, B1_predicted_gain=None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"behavior_feasibility.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    summary = f"""B1 로컬 행동 평가 가능성 감사

로컬 C0/F10 별도 검증 기록 {len(traces)}개를 실제로 읽었습니다.
이 파일에는 오차·유효 마스크·60열 진단이 있지만 actor 입력 235차원 관측은 없습니다.
회수된 이번 B1 폴더의 NPZ 기록은 {len(latest)}개입니다.

따라서 B1 정책을 실제 녹화 관측에 재생하는 유효한 행동 비교를 지금 수행할 수 없습니다.
정책 입력에는 저장하지 않은 자세 중력, 관절 상대위치, 187차원 높이 스캔과 실제 관측
노이즈가 필요합니다. 이를 임의로 채워 넣어 만든 네트워크 출력은 점수 평가가 아닙니다.

현재 Python의 torch는 설치되어 있지만 isaaclab/isaacsim 모듈은 찾지 못했습니다.
기본 후보 경로도 없으며, 전 기기 검색이나 설치는 수행하지 않았습니다.

이번 감사는 체크포인트 바이트 해시와 기존 기록을 읽었으며, B1 추론·시뮬레이션·학습·
서버 조작을 하지 않았습니다. B1 생존율/RMSE/Q3/공식 예상점수는 미측정입니다.
다른 팀 점수와 비교할 수 있는 B1 성능 증거가 이 감사에서 새로 생기지는 않았습니다.

정확한 파일별 키·배열 차원·SHA는 behavior_feasibility.json에 보존했습니다.
"""
    (OUT/"behavior_feasibility.txt").write_text(summary, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "historical_heldout_trace_count", "B1_recorded_trace_count", "usable_recorded_actor_observation_trace_count", "performed_actor_inference", "B1_internal_Q3")}, indent=2))


if __name__ == "__main__":
    main()
