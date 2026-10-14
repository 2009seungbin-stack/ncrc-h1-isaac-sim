"""Freeze the authorized single-candidate four-hour design locally."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
OLD = WS / 'analysis/baseline_redesign_20261009_4h'
ORIGINAL = WS / 'analysis/short_sweep_20261002/recovered_20261003/short_sweep_20261002/fresh_rewards_original.py'
NAME = 'S5_torque_half'
WEIGHTS = dict(track_lin_vel_xy_exp=2.0, track_ang_vel_z_exp=1.5,
               feet_air_time=.01, lin_vel_z_l2=-3.0, ang_vel_xy_l2=-.08,
               action_rate_l2=-.01, dof_acc_l2=-2.5e-7,
               dof_torques_l2=-.0001, flat_orientation_l2=0.0, dof_pos_limits=0.0)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def main():
    design = json.loads((OLD/'design.json').read_text(encoding='utf-8-sig'))
    design.update(status='AUTHORIZED_READY_PENDING_LOGIN', execution_authorized=True,
                  primary_candidate=NAME, candidate_name=NAME,
                  candidate_reward_file='quadruped_rewards_S5_torque_half.py',
                  candidate_selection_basis='Single torque-cost relaxation preserving the other nine F10 weights; hypothesis not a demonstrated score improvement',
                  candidates={NAME: dict(role='only_scheduled_new_policy', trained=False, weights=WEIGHTS)},
                  objective='Measure whether torque-cost relaxation improves full-window tracking without losing yaw or survival, within four actual allocation hours')
    design['budget'].update(planning_multiplier=1.05, selection_seconds=900,
                            validation_seconds=1650, computation_margin_seconds=300,
                            training_start_latest_seconds=1800,
                            allocation_minutes=240, stop_when_remaining_seconds=1500,
                            conservative_evaluation_planning_seconds=2450,
                            basis='Previous same-runtime training and older evaluation timings; diagnostic buffer and boot allowance are planning assumptions, not a guarantee')
    design['budget']['deadlines_relative_to_actual_allocation_start']['dashboard_shutdown_request_latest'] = 12900
    design['budget']['deadlines_relative_to_actual_allocation_start']['training_begin_latest'] = 1800
    design['budget']['deadlines_relative_to_actual_allocation_start']['setup_end'] = 1200
    design['budget']['setup_target_seconds'] = 900
    design['budget']['setup_hard_backstop_seconds'] = 1200
    design['evaluation']['screen'] = dict(enabled=False, reason='Concentrate the four-hour allocation on one fresh training and complete reference-matched validation; old B1 checkpoints remain preserved locally')
    design['evaluation']['required_new_diagnostics'] += ['full_env_raw_velocity_error_and_validity_masks', 'full_env_p99_absolute_error', 'signed_transition_error', 'ordinary_window_yaw_error']
    design['evaluation']['diagnostic_implementation_status'] = 'IMPLEMENTED_CPU_TESTED_REQUIRES_REMOTE_RUNTIME_VERIFICATION'
    design['official_target']['reason'] = 'Local error/Q3 has no validated official-score mapping; one independent training seed; archived C0/F10 have different training lengths and historical runtimes'
    design['execution_readiness'].update(future_runner_ready=False, requires_future_separate_training_instruction=False,
                                          user_training_authorization_received=True)
    design['forbidden_in_current_task'] = ['second_new_policy', 'training_retry', 'resume_training', 'extra_evaluation', 'budget_extension', 'official_submission', 'local_PC_shutdown']
    design['scope'] = dict(allowed='Existing ten static REWARD_WEIGHTS values only',
                          protected='Provided train/play/agent_cfg/env_cfg, PPO, observations, actions, reward kernels and task sources unchanged',
                          candidate_changes={'dof_torques_l2': {'F10': -.0002, NAME: -.0001}},
                          no_score_forecast=True)
    raw = ORIGINAL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '09630e6f548b16a2fa0a81eef73aa393cf5f12eadccd0bad8a144b924e8c1be7'
    nodes = [n.value for n in ast.parse(raw.decode('utf-8')).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'REWARD_WEIGHTS' for t in n.targets)]
    assert len(nodes) == 1
    node = nodes[0]
    lines = raw.splitlines(keepends=True)
    a = sum(map(len, lines[:node.lineno-1]))+node.col_offset
    b = sum(map(len, lines[:node.end_lineno-1]))+node.end_col_offset
    literal = ('{\n'+''.join('    '+repr(k)+': '+repr(v)+',\n' for k,v in WEIGHTS.items())+'}').encode('utf-8')
    candidate = raw[:a]+literal+raw[b:]
    out = ROOT/'execution'/design['candidate_reward_file']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(candidate)
    assert ast.literal_eval(next(n.value for n in ast.parse(candidate.decode('utf-8')).body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REWARD_WEIGHTS' for t in n.targets))) == WEIGHTS
    design['candidate_reward_sha256'] = hashlib.sha256(candidate).hexdigest()
    write_json(ROOT/'design.json', design)
    ledger = dict(status='AUTHORIZED_NOT_STARTED_LOGIN_PENDING', server_started=False,
                  allocation_start_epoch=None, effective_shutdown_deadline_epoch=None,
                  remote_folder='/workspace/student_workspace/research_execution_20261010_4h',
                  tmux_session='go2_research_4h_20261010', actual_server=None,
                  authorized_total_seconds=14400, candidate=NAME, policy_training_started=False,
                  retrieval_complete=False, server_off_verified=False,
                  existing_go2_automation='PAUSED', latest_user_authorization='연구 더 빡세게 해서 실제 서버 사용 4시간하자',
                  blocker='Portal session expired; user direct sign-in requested once, no password requested in chat',
                  design_sha256=hashlib.sha256((ROOT/'design.json').read_bytes()).hexdigest(),
                  official_submission_authorized=False, local_PC_shutdown_authorized=False)
    if not (ROOT/'execution.json').exists():
        write_json(ROOT/'execution.json', ledger)
    operations = ROOT/'operations.txt'
    if not operations.exists():
        operations.write_text('2026-10-10: User authorized one actual four-hour allocation inclusive of setup, training, validation, retrieval and verified OFF.\n'
            'No server has been started. Portal authentication expired; user sign-in requested once.\n'
            'Selected S5_torque_half: F10 ten-weight reward with torque penalty -0.0002 to -0.0001 only.\n'
            'Initial old B1 screen is disabled to preserve time for independent validation and retrieval.\n'
            'Scheduled: scratch1500/env6144/seed42; checkpoint1000 vs1499 selection with C0/F10; frozen selected checkpoint with C0/F10 across G1-G7 and two separate reset seeds.\n'
            'Local diagnostic metrics are not an official score forecast or a 93-point guarantee.\n', encoding='utf-8')
    print(json.dumps(dict(candidate=NAME, candidate_sha256=design['candidate_reward_sha256'],
                          design_sha256=ledger['design_sha256'], server_started=False), indent=2))

if __name__ == '__main__':
    main()
