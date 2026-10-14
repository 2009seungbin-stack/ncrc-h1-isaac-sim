"""Combine existing local audits without policy inference or server access."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "offline_eval"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


historical = read("historical_comparison.json")
training = read("training_metrics.json")
inventory = read("checkpoint_inventory.json")
feasibility = read("behavior_feasibility.json")
assert training["training_iterations"] == 1500
assert inventory["checkpoint_count"] == 16
assert all(inventory["checks"].values())
assert feasibility["performed_actor_inference"] is False
assert feasibility["B1_recorded_trace_count"] == 0
assert historical["B1_policy_internal_Q3"] is None

last100 = historical["last100_comparison"]["F10_historical_training"]
rows = [r for r in training["rows"] if 1400 <= r["iteration"] <= 1499]
assert len(rows) == 100
metrics = [
    ("선속도 추종 보상 / 가중치", "weight_normalized_episode_reward_xy", "linear_weight_removed_proxy", True),
    ("회전 추종 보상 / 가중치", "weight_normalized_episode_reward_yaw", "yaw_weight_removed_proxy", True),
    ("로그 XY 오차", "Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_xy", False),
    ("로그 yaw 오차", "Metrics/base_velocity/error_vel_yaw", "Metrics/base_velocity/error_vel_yaw", False),
    ("몸통 접촉 종료 로그", "Episode_Termination/base_contact", "Episode_Termination/base_contact", False),
    ("평균 지형 레벨", "Curriculum/terrain_levels", "Curriculum/terrain_levels", True),
]
comparison = []
for label, key, row_key, higher_is_better in metrics:
    item = last100[key]
    independent_b1_mean = mean(r[row_key] for r in rows)
    assert abs(item["B1"] - independent_b1_mean) < 1e-10
    relative = (item["B1"] / item["reference"] - 1) * 100
    assert abs(relative - item["relative_change_percent"]) < 1e-9
    comparison.append({"metric": label, "key": key, "B1": item["B1"], "F10": item["reference"],
                       "relative_change_percent": relative, "higher_is_better_for_training_diagnostic": higher_is_better})

checkpoints = {c["checkpoint"]: c["trailing_window"]["mean"] for c in training["checkpoints"]}
checkpoint_delta = {}
for key in ["Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw", "Episode_Termination/base_contact", "Curriculum/terrain_levels"]:
    a, b = checkpoints[1000][key], checkpoints[1499][key]
    checkpoint_delta[key] = {"1000_trailing50": a, "1499_trailing50": b, "relative_change_percent": (b / a - 1) * 100}

now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
source_names = ["training_metrics.json", "historical_comparison.json", "checkpoint_inventory.json", "behavior_feasibility.json"]
assessment = {
    "status": "LOCAL_TRAINING_DIAGNOSTIC_COMPLETE_B1_POLICY_PERFORMANCE_UNMEASURED",
    "created_utc": now,
    "scope": "Local training-log/config comparison and CPU checkpoint integrity only",
    "server_used": False, "simulation_performed": False, "policy_inference_performed": False,
    "verdict": "B1 currently has low priority for further long training; evidence does not support improvement over historical F10 at 1500 iterations",
    "verdict_is_research_priority_judgment_not_measured_policy_rank": True,
    "comparison_window_inclusive": [1400, 1499], "comparison": comparison,
    "checkpoint_training_context_comparison": checkpoint_delta,
    "checkpoint_integrity": {"count": 16, "all_checks_pass": True, "actor_dimensions": [235, 512, 256, 128, 12]},
    "B1_fixed_scenario_survival": None, "B1_fixed_scenario_RMSE": None, "B1_Q3": None,
    "B1_predicted_official_score": None, "B1_predicted_score_gain": None, "probability_of_93_points": None,
    "unmeasured_reason": "No B1 behavior traces; historical records omit real 235D actor observations; local simulator unavailable in checked paths",
    "comparability_limits": historical["comparability_limits"],
    "source_artifacts": [{"path": str(OUT / n), "sha256": sha(OUT / n)} for n in source_names],
    "chart": str(OUT / "B1_training_assessment.png"),
}
dump(OUT / "internal_assessment.json", assessment)

lines = [
    "Go2 B1 로컬 내부 평가", f"작성: {now}", "",
    "결론: 현재 B1은 장기 학습을 추가할 우선순위가 낮다.",
    "1500회 학습 로그에서 역사 F10보다 개선됐다는 근거가 없으며, 후반 추종·접촉 종료·지형 진행 지표가 함께 불리하다.",
    "이는 연구 우선순위 판단이다. 실제 B1 정책의 고정 시나리오 점수가 더 낮다고 확정한 결과는 아니다.", "",
    "서버를 사용하지 않고 수행한 평가", "- 0~1499의 1500회 로그를 파싱하고 여러 학습 구간과 저장 시점별 곡선을 분석했다.",
    "- 역사 F10/F00의 같은 학습 회차와 cfg를 대조했다. B1/F10은 seed42,6144환경,24steps 및 PPO 설정이 같고 저장 cfg 차이는 보상/실험 메타데이터다.",
    "- 16개 체크포인트를 CPU에서 읽어 원격 크기/SHA,저장 iter,235→512→256→128→12 구조,모든 부동소수점 텐서의 유한값을 확인했다.",
    "- 저장본마다 실제 actor MLP 가중치가 변했다. 모델 파일 손상이나 학습 갱신 부재의 증거는 없다.",
    "- B1 정책 실행,시뮬레이션,학습,서버 접속은 하지 않았다.", "",
    "같은 학습 회차 1400~1499의 평균 비교", "지표 | B1 | 역사 F10 | B1 상대 변화",
]
for item in comparison:
    if item["key"] == "Episode_Termination/base_contact":
        lines.append(f'{item["metric"]} | {item["B1"]*100:.4f}% | {item["F10"]*100:.4f}% | {(item["B1"]-item["F10"])*100:+.4f}%p')
    else:
        lines.append(f'{item["metric"]} | {item["B1"]:.6f} | {item["F10"]:.6f} | {item["relative_change_percent"]:+.2f}%')
lines += ["", "곡선에서 읽을 수 있는 것",
    "초기 학습은 진행됐지만 약600회 이후 XY 로그 오차가 약0.70에서 정체했다. 지형 레벨이 올라가면서 몸통 접촉 종료 로그는 증가했다.",
    "500~599에는 일부 F10 대비 이점이 있었지만 900~999 이후 여러 창에서 두 추종 보상·접촉 종료·지형 레벨이 함께 불리했다.",
    "1000과1499의 직전50회 비교는 XY 오차 약4.90%,yaw 오차 약2.01% 개선과 접촉 종료 약1.263%p 증가가 함께 나타났다.",
    "이 기간에는 지형 난이도도 변하므로 최적 체크포인트를 결정할 수 없다. 자동 총보상 선택 model600도 고정 성능 최적이라는 증거는 아니다.",
    "F00는 C0 보상을 사용하는 별도 scratch run이며 실제 C0_5500의 초기 로그가 아니다. B1은 F00 대비 회전 추종/접촉에서 일부 이점,선속도 추종에서 불리한 혼합 결과다.", "",
    "점수와 상승폭", "B1의 G1~G7 생존율·전체 구간 RMSE/MAE·Q3는 미측정이며,공식 예상점수와 상승폭은 산출하지 않았다.",
    "B1 행동 기록은0개이고 기존28개 C0/F10 기록에도 실제235차원 관측이 없다. 누락된 자세·관절 상대위치·높이 스캔·관측 노이즈를 임의로 채우는 방식은 유효한 평가가 아니다.",
    "확인한 Python/기본 설치 경로에는 로컬 IsaacLab/IsaacSim이 없었다. 서버나 설치를 추가하지 않았다.",
    "기존 공식 Go2 점수 앵커1개만으로 학습 로그→공식 점수 변환을 식별할 수 없다. 93점 가능성이나 +몇점을 숫자로 제시하면 새 가정을 측정치처럼 보이게 한다.", "",
    "해석 한계", "보상/가중치는 계수 차이만 제거하며 episode 길이·reset·집계·탐색·학습 명령 분포까지 같게 만들지 않는다. 로그 error는 검증 RMSE가 아니다.",
    "접촉 종료는 iteration 로그의 산술평균이며 원시 episode를 합친 낙상률 또는 G1~G7 생존율이 아니다.",
    "과거 runtime 동일성은 입증하지 못했고 각 run은 독립 training seed 반복이 없다. 연속100회 로그를 독립 실험100개로 취급하지 않는다.",
    "이번 결과로 추가 학습을 해도 영원히 개선되지 않는다고 주장할 수는 없다. 다만 제한된 연구 시간의 다음 예산을 B1 연장에 배정할 근거는 약하다.", "",
    "자료", "internal_assessment.json: 본 판단과 수치/원본 분석 artifact SHA", "historical_comparison.json/.py: 원본 로그·cfg 해시,설정 diff,같은 회차 비교",
    "training_metrics.json / curve_summary.txt: 전체 시계열·구간·저장시점 요약", "checkpoint_inventory.json/.txt: 16개 저장본 무결성", "behavior_feasibility.json/.txt: 입력 기록 및 로컬 실행 가능성 감사",
    "B1_training_assessment.png/.svg: 학습 곡선", "",
    "운영 상태: 서버02 OFF 및 go2 PAUSED 상태를 유지했다. 이번 로컬 분석은 기존 selection0/validation0 상태를 변경하지 않는다.",
]
(OUT / "internal_assessment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Keep the completed shutdown and incomplete physical evaluation ledger intact.
ledger_path = BASE / "execution.json"
ledger = json.loads(ledger_path.read_text(encoding="utf-8-sig"))
protected_keys = ["status", "selection_completed", "validation_completed", "evaluation_coverage", "server_shutdown_confirmed", "server_off_confirmed_utc", "automation_status", "official_score_prediction"]
protected = {k: ledger[k] for k in protected_keys}
ledger["offline_internal_assessment"] = {
    "completed_utc": now, "status": assessment["status"], "server_used": False,
    "policy_performance_measured": False, "research_priority": "B1_FURTHER_LONG_TRAINING_LOW",
    "model_integrity_verified_count": 16, "training_iterations_analyzed": 1500,
    "official_score_prediction": None, "predicted_gain": None,
    "report": str(OUT / "internal_assessment.txt"), "json": str(OUT / "internal_assessment.json"),
    "chart": str(OUT / "B1_training_assessment.png"),
    "report_sha256": sha(OUT / "internal_assessment.txt"), "json_sha256": sha(OUT / "internal_assessment.json"),
}
assert {k: ledger[k] for k in protected_keys} == protected
dump(ledger_path, ledger)

marker = "[LOCAL_OFFLINE_INTERNAL_ASSESSMENT]"
entry = (f"\n{marker} {now}\n"
         "사용자 승인 범위: 내부 평가,서버 사용 금지. 로컬 로그/설정/CPU tensor만 분석. 전체1500회 및16개 체크포인트 검사 완료.\n"
         "1400~1499 F10 대비 정규화 XY/yaw 보상-4.9638%/-6.9157%,로그 XY/yaw 오차+11.8630%/+13.4591%,접촉 종료12.5887%vs9.8374%,지형5.204448vs5.749749.\n"
         "판단:B1 추가 장기 학습 우선순위 낮음. 고정 시나리오 B1 성능/Q3/공식점수/상승폭 미측정. 서버OFF/자동화PAUSED 및 selection0/validation0 유지.\n"
         "보고서 offline_eval/internal_assessment.txt/.json,학습곡선 B1_training_assessment.png/.svg 보존.\n")
for name in ["operations.txt", "final_analysis.txt"]:
    path = BASE / name
    contents = path.read_text(encoding="utf-8-sig")
    if marker not in contents:
        path.write_text(contents.rstrip() + "\n" + entry, encoding="utf-8")
print(json.dumps({"status": assessment["status"], "metrics_independently_verified": len(comparison), "ledger_protected_fields_preserved": True, "report": str(OUT / "internal_assessment.txt")}, ensure_ascii=False))
