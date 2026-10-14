"""Build untrained Go2 reward proposals and audit existing local evidence.

No simulator, policy training, portal mutation, or remote execution is performed.
Weights are hypotheses. Reweighting an old trajectory does not predict a new policy.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT.parent
RECOVERED = ANALYSIS / 'short_sweep_20261002/recovered_20261003/short_sweep_20261002'
ORIGINAL = RECOVERED / 'fresh_rewards_original.py'
CONTROL_ENV = next((RECOVERED / 'preserved/short_20261002_control').glob('*/params/env.yaml'))
INPUT_PATHS = {
    'source': ORIGINAL,
    'control_env': CONTROL_ENV,
    'old_research': ANALYSIS / 'score_breakthrough_20261002/third/analysis.json',
    'anchor': ANALYSIS / 'score_reverse_20261001/analysis.json',
    'budget': ANALYSIS / 'research_directions_20261003/objective_and_budget_audit.json',
    'heldout': RECOVERED / 'heldout_ranking.json',
    'short': RECOVERED / 'short_ranking.json',
    'saved': RECOVERED / 'saved_ranking.json',
    'C0_checkpoint': ANALYSIS / 'short_sweep_20261002/inputs/models/C0_5500.pt',
    'official_C0_export': ANALYSIS / 'score_reverse_20261001/official_r4/policy.pt',
    'local_C0_export': ANALYSIS.parent / 'submit_C0/policy.pt',
    'portal_evidence': ROOT / 'portal_evidence.json',
}
BASE = dict(track_lin_vel_xy_exp=2.0, track_ang_vel_z_exp=1.5,
            feet_air_time=0.01, lin_vel_z_l2=-3.0, ang_vel_xy_l2=-0.08,
            action_rate_l2=-0.01, dof_acc_l2=-2.5e-7,
            dof_torques_l2=-0.0002, flat_orientation_l2=0.0, dof_pos_limits=0.0)
DRIVE = dict(track_lin_vel_xy_exp=3.0, track_ang_vel_z_exp=2.25)
FAST = dict(feet_air_time=0.0, lin_vel_z_l2=-1.5)
SCREEN_IDS = ('S1_tracking_pair', 'R1_tracking_double')
SCREEN_ITERATIONS = 1500
OFFICIAL_TOTAL_TARGET = 93
SPECS = [
    ('S0_control', 'primary', {}, 'F10 reward reference; freshly trained comparison.'),
    ('S1_tracking_pair', 'primary', DRIVE,
     'Strengthen both tracking terms by 1.5 while retaining their ratio; retain smoothing.'),
    ('S2_short_step', 'primary', FAST,
     'Remove the long-airtime preference and halve vertical-motion cost; gait is not prescribed.'),
    ('S3_long_step', 'primary', dict(feet_air_time=0.06, lin_vel_z_l2=-1.5),
     'Contrast a stronger long-airtime preference against S2 at the same vertical-motion cost.'),
    ('S4_tracking_short_step', 'primary', dict(DRIVE, **FAST),
     'Combine tracking gain, air-time prior removal and body-normal velocity cost relief; short steps are not prescribed. Reserve after deeper review.'),
    ('S5_torque_half', 'primary', dict(dof_torques_l2=-0.0001),
     'Reduce motor-effort cost by half without relaxing acceleration/action-change smoothing.'),
    ('R1_tracking_double', 'conditional', dict(track_lin_vel_xy_exp=4.0, track_ang_vel_z_exp=3.0),
     'Proposed paired screen against S1: tracking gain 2x with fixed ratio and unchanged auxiliary weights; no superiority claim.'),
    ('R2_tracking_torque_half', 'conditional', dict(DRIVE, dof_torques_l2=-0.0001),
     'Only if tracking gain and torque relief are individually promising.'),
    ('R3_tracking_acc_relief', 'conditional', dict(DRIVE, dof_acc_l2=-1.875e-7),
     'Only if retained trajectories show response suppression; prior acceleration-only short test failed.'),
    ('R4_tracking_roll_flex', 'conditional', dict(DRIVE, ang_vel_xy_l2=-0.04),
     'Allow transient roll/pitch angular motion while retaining vertical-motion cost; check body metrics.'),
    ('R5_tracking_orientation', 'conditional', dict(DRIVE, flat_orientation_l2=-0.3),
     'Only with orientation measurements; slopes may conflict with world-horizontal preference.'),
    ('R6_tracking_limits', 'conditional', dict(DRIVE, dof_pos_limits=-0.5),
     'Only if joint positions show frequent soft-limit use; do not add a penalty blindly.'),
    ('R7_linear_priority', 'conditional', dict(track_lin_vel_xy_exp=2.5,
                                              track_ang_vel_z_exp=1.25,
                                              dof_torques_l2=-0.0001),
     'If linear tracking remains weaker: an alternative task ratio with effort relief.'),
    ('R8_tracking_body_quiet', 'conditional', dict(DRIVE, lin_vel_z_l2=-4.5,
                                                 ang_vel_xy_l2=-0.12, feet_air_time=0.0),
     'Opposite terrain branch: stronger body-motion control plus stronger tracking; inspect slope/stairs.'),
    ('R9_tracking_long_step', 'conditional', dict(DRIVE, feet_air_time=0.12, lin_vel_z_l2=-1.5),
     'Only if S3 beats S2: stronger long-step exploration; not a foot-clearance reward.'),
]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reward_node(tree):
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'REWARD_WEIGHTS' for t in node.targets)]
    if len(nodes) != 1 or not isinstance(nodes[0].value, ast.Dict):
        raise ValueError('Expected one literal reward dictionary')
    return nodes[0]


def non_reward_ast(tree):
    tree = copy.deepcopy(tree)
    reward_node(tree).value = ast.Constant(value=None)
    return ast.dump(tree, include_attributes=False)


def yaml_weights(text):
    active, term, weights = False, None, {}
    for line in text.splitlines():
        if line == 'rewards:':
            active = True
            continue
        if active and line and not line[0].isspace() and not line.startswith('#'):
            break
        if not active:
            continue
        match = re.fullmatch(r'  ([A-Za-z_][A-Za-z_0-9]*):(?: null)?', line)
        if match:
            term = match.group(1)
        match = re.fullmatch(r'    weight: (.+)', line)
        if match and term:
            weights[term] = float(match.group(1))
    return weights


def render_candidate(source, weights):
    original_tree = ast.parse(source.decode('utf-8'))
    node = reward_node(original_tree).value
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
    literal = ('{\n' + ''.join(f'    {key!r}: {value!r},\n' for key, value in weights.items()) + '}').encode('utf-8')
    result = source[:start] + literal + source[end:]
    tree = ast.parse(result.decode('utf-8'))
    compile(tree, '<untrained_reward_proposal>', 'exec')
    if non_reward_ast(tree) != non_reward_ast(original_tree):
        raise ValueError('Changed code outside reward dictionary')
    if ast.literal_eval(reward_node(tree).value) != weights:
        raise ValueError('Rendered weights mismatch')
    if not (result[:start] == source[:start] and result[start+len(literal):] == source[end:]):
        raise ValueError('Changed bytes outside reward dictionary')
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    source = ORIGINAL.read_bytes()
    inputs = {name: {'path': str(path), 'sha256': sha(path.read_bytes())}
              for name, path in INPUT_PATHS.items()}
    env_text = CONTROL_ENV.read_text(encoding='utf-8')
    active = yaml_weights(env_text)
    dt_matches = re.findall(r'^  dt: ([^\r\n]+)$', env_text, re.MULTILINE)
    decimation_matches = re.findall(r'^decimation: ([^\r\n]+)$', env_text, re.MULTILINE)
    if len(dt_matches) != 1 or len(decimation_matches) != 1:
        raise ValueError('Historical environment time step is ambiguous')
    step_dt = float(dt_matches[0]) * int(decimation_matches[0])
    if active != BASE:
        raise ValueError(f'Historical control weights differ: {active}')
    candidates = []
    for name, group, changes, intent in SPECS:
        weights = dict(BASE, **changes)
        if set(weights) != set(active) or not all(math.isfinite(v) for v in weights.values()):
            raise ValueError('Unsupported or nonfinite weights')
        if any(weights[k] > 0 for k in weights if k.endswith('_l2') or k == 'dof_pos_limits'):
            raise ValueError('Penalty sign inversion is excluded')
        candidate = render_candidate(source, weights)
        path = ROOT / 'candidates' / f'{name}_PROPOSAL.py'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(candidate)
        candidates.append(dict(id=name, group=group, intent=intent, changes=changes,
                               effective_weights=weights, path=str(path), sha256=sha(candidate),
                               current_budget_role=('proposed_screen' if name in SCREEN_IDS else 'offline_reserve'),
                               proposed_screen_iterations=(SCREEN_ITERATIONS if name in SCREEN_IDS else None),
                               trained=False, validated_on_robot=False, predicted_official_score=None,
                               syntax_valid=True, outside_dictionary_bytes_unchanged=True,
                               active_terms_verified_from_historical_env=True))
    rows = [r['effective_weights'] for r in candidates]
    if len({json.dumps(r, sort_keys=True) for r in rows}) != len(rows):
        raise ValueError('Duplicate proposals')

    historical = read_json(INPUT_PATHS['old_research'])
    medians = historical['training_terms']['F10'][0]['medians']
    reweighted = []
    for row in candidates:
        terms = {k: v * row['effective_weights'][k.split('/')[-1]] / BASE[k.split('/')[-1]]
                 for k, v in medians.items() if k.startswith('Episode_Reward/') and BASE[k.split('/')[-1]] != 0}
        reweighted.append({'candidate': row['id'], 'fixed_old_trajectory_term_medians': terms,
                           'unidentified_new_terms': [k for k in row['changes'] if BASE[k] == 0],
                           'warning': 'Arithmetic reweighting only; not new-policy reward, gradient, or score prediction.'})

    anchor = read_json(INPUT_PATHS['anchor'])['anchors']['Go2_R4']
    if not math.isclose(sum(r['official_point'] for r in anchor['rows']), 58.14, abs_tol=.001):
        raise ValueError('Official anchor mismatch')
    saved_reference = read_json(INPUT_PATHS['saved'])['models']['C0_5500']['checkpoint_sha256']
    if inputs['C0_checkpoint']['sha256'] != saved_reference:
        raise ValueError('C0 checkpoint differs from previously evaluated reference')
    if inputs['official_C0_export']['sha256'] != inputs['local_C0_export']['sha256']:
        raise ValueError('Local C0 export differs from downloaded official submission')
    target = {}
    for total in (90, 92, OFFICIAL_TOTAL_TARGET, 95):
        gain = total - (30 + anchor['official_sim'])
        target[str(total)] = {'needed_simulation_gain_if_report_stays_30': gain,
                              'needed_weighted_scenario_quality_gain': gain / 70,
                              'needed_quality_gain_if_only_G3_G4_G5_improve': gain / (70 * .5),
                              'warning': 'Required official arithmetic, not predicted candidate performance.'}
    budget = read_json(INPUT_PATHS['budget'])
    seconds = budget['budget_measured_seconds_per_iteration']
    portal = read_json(INPUT_PATHS['portal_evidence'])
    board = portal['board']
    non_simulation_points = board['Go2_intent_points'] + board['Go2_report_points']
    required_simulation_points = OFFICIAL_TOTAL_TARGET - non_simulation_points
    scenario_weights = portal['rules']['scenario_weights']
    terrain_points = sum(board['scenario_points'][2:5])
    terrain_maximum = portal['rules']['simulation_points'] * sum(scenario_weights[2:5])
    simulation_gain = required_simulation_points - anchor['official_sim']
    goal = {
        'selected_by_user': True, 'official_total_target': OFFICIAL_TOTAL_TARGET,
        'reference_official_total': anchor['official_sim'] + non_simulation_points,
        'non_simulation_points_assumption': non_simulation_points,
        'required_simulation_points': required_simulation_points,
        'simulation_maximum': portal['rules']['simulation_points'],
        'required_simulation_gain': simulation_gain,
        'required_weighted_scenario_quality': required_simulation_points / portal['rules']['simulation_points'],
        'required_weighted_quality_gain': simulation_gain / portal['rules']['simulation_points'],
        'terrain_only_if_other_scores_unchanged': {
            'reference_points': terrain_points, 'maximum_points': terrain_maximum,
            'required_points': terrain_points + simulation_gain,
            'required_average_quality': (terrain_points + simulation_gain) / terrain_maximum},
        'uniform_quality_point_allocation_example': [
            {'scenario': f'G{i+1}', 'reference_official_points': reference,
             'illustrative_target_points': weight * required_simulation_points,
             'illustrative_gain': weight * required_simulation_points-reference}
            for i, (weight, reference) in enumerate(zip(scenario_weights, board['scenario_points']))],
        'selection_priority': 'Prioritize gains across the full weighted scenario set while preserving survival and recording tracking regressions. Partial short screens cannot certify target achievement.',
        'completion_evidence': 'Actual official Go2 total at least 93; offline metrics and arithmetic are not evidence of achievement.',
        'official_score_prediction': None, 'target_achieved': False}
    remaining = portal['board']['remaining_minutes'] / 60
    screen = len(SCREEN_IDS) * SCREEN_ITERATIONS * seconds / 3600
    exploration_cap, main_cap, validation_cap = 6.0, 10.0, 3.0
    total_cap = exploration_cap + main_cap + validation_cap
    if remaining < total_cap or screen >= exploration_cap:
        raise ValueError('Revised plan does not preserve its bounded budget')
    plan = {'quota_remaining_hours_last_verified_20261009': remaining,
            'quota_source': str(INPUT_PATHS['portal_evidence']),
            'historical_seconds_per_iteration': seconds,
            'revision_reason': 'User requires protecting time for final training and validation instead of spending it all on experiments.',
            'exploration': {
                'candidate_ids': list(SCREEN_IDS), 'iterations_each': SCREEN_ITERATIONS,
                'scratch': True, 'training_seed': 42, 'training_environments': 6144,
                'training_only_projection_hours': screen,
                'total_including_startup_evaluation_retrieval_shutdown_cap_hours': exploration_cap,
                'within_cap_allowance_at_historical_speed_hours': exploration_cap-screen,
                'fixed_existing_comparator_ids': ['C0_5500', 'F10_3000'],
                'new_control_training': False,
                'previous_proposed_candidate_ids': ['S1_tracking_pair', 'S4_tracking_short_step'],
                'revision_reason': 'Deeper source and trajectory review favors tracking-strength comparison; S4 body-normal and air-time explanation was weaker than initially stated.',
                'selection_gate': 'Matched 1500-iteration candidate performance, late learning trend and terrain survival; no automatic early-rank promotion.',
                'ambiguous_outcome': 'Leave convergence unresolved and preserve quota; do not automatically launch long training.',
                'comparison_limit': 'Unequal-length comparisons against existing C0/F10 support policy selection, not reward-change causality.'},
            'main_training': {
                'max_new_candidates': 1, 'scratch': True, 'training_seed': 42,
                'target_iterations': 5000, 'maximum_iterations_if_measured_time_fits': 6000,
                'training_only_projection_5000_hours': 5000*seconds/3600,
                'training_only_projection_6000_hours': 6000*seconds/3600,
                'allocation_total_cap_hours': main_cap,
                'checkpoint_shortlist_iterations': [2500, 3000, 4000, 'final'],
                'checkpoint_screen_replaces_last_checkpoint_default': True,
                'checkpoint_selection_and_final_verification_share_existing_validation_budget': True,
                'resume_assumed_free': False, 'second_training_seed_automatic': False,
                'admission': 'Measure fresh speed and startup/evaluation cost; planned iterations must fit allocation with recovery/shutdown reserve.'},
            'final_validation_retrieval': {
                'protected_total_cap_hours': validation_cap,
                'required_scenarios': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'],
                'fixed_official_C0_required': True,
                'admission': 'Do not consume this reserve on candidate exploration. Recheck measured evaluation and recovery cost before training.'},
            'planned_total_cap_hours': total_cap,
            'protected_uncommitted_hours_after_all_caps': remaining-total_cap,
            'quota_preserved_after_exploration_cap_hours': remaining-exploration_cap,
            'additional_candidates_or_seed_repeats_automatic': False,
            'superseded_plan': {'status': 'WITHDRAWN_NOT_RUN',
                                'screen_candidate_count': 6, 'screen_iterations_each': 1200,
                                'total_iterations': 19200,
                                'training_plus_assumed_overhead_hours': 19200*seconds/3600+4},
            'allocation_hours_limit': 10,
            'minimum_shutdown_retrieval_reserve_hours_each_allocation': .75,
            'not_authorized_or_launched': True,
            'warning': 'Caps are limits, not measured completion times. Historical speed is an estimate. No extra candidate, control, seed repeat or long run is automatic.'}
    feet = [{'airtime_s': air, 'landings_per_s_assumption': 4,
             'step_dt_s_from_historical_env': step_dt,
             'dt_omitted_weighted_landing_sum_per_second': {str(w): 4*(air-.5)*w for w in (0, .01, .06, .12)},
             'integrated_reward_per_second_with_step_dt': {str(w): 4*(air-.5)*w*step_dt for w in (0, .01, .06, .12)},
             'assumptions': 'One rewarded control step per first contact, command gate active, fixed landing frequency; not measured gait or official score.',
             'reward_manager_dt_evidence': 'IsaacLab v2.3.2 upstream RewardManager.compute; installed manager was not locally preserved.'}
            for air in (.15, .25, .5, .65)]
    audit = {'status': 'OFFLINE_RESEARCH_UNTRAINED_PROPOSALS',
             'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'inputs': inputs, 'candidate_count': len(candidates),
             'primary_count_including_control': sum(r['group']=='primary' for r in candidates),
             'proposed_screen_count': len(SCREEN_IDS),
             'official_target_arithmetic': target,
             'selected_official_goal': goal,
             'terrain_only_example_quality_increase': .054,
             'terrain_only_example_total_if_other_scores_stay_fixed': 88.14+70*.5*.054,
             'fixed_old_trajectory_reweighting': reweighted,
             'feet_airtime_arithmetic_not_foot_clearance': feet,
             'budget': plan,
             'all_input_hashes_unchanged': all(sha(path.read_bytes()) == inputs[name]['sha256']
                                             for name, path in INPUT_PATHS.items()),
             'fixed_official_reference': {
                 'C0_checkpoint_matches_prior_evaluation': True,
                 'C0_export_matches_downloaded_official_submission': True,
                 'required_in_final_evaluation': True,
                 'new_S0_is_F10_reward_control_not_official_C0': True},
             'no_server_started': True, 'no_policy_training': True, 'no_new_robot_evaluation': True,
             'no_official_submission': True}
    (ROOT / 'candidate_manifest.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': audit['status'], 'candidate_count': len(candidates),
                      'primary_count': audit['primary_count_including_control'],
                      'outside_reward_bytes_preserved': True,
                      'official_targets': target, 'selected_official_goal': goal,
                      'budget': plan}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
