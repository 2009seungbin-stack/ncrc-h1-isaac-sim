"""Record completed local checks; no remote execution or inference."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    design_path = ROOT / 'design.json'
    design = json.loads(design_path.read_text(encoding='utf-8'))
    design['execution_readiness']['future_runner_ready'] = True
    design['execution_readiness']['local_CPU_contract_checks_passed'] = True
    design['execution_readiness']['actual_remote_preflight_verified'] = False
    design['evaluation']['observation_command_boundary'] = (
        'Retain the inherited cached observation after command injection for all '
        'current policies equally. The first actor input at a boundary may hold '
        'the previous command for one control step. Historical October 2 Q3 used '
        'a refreshed command observation and is not directly comparable.')
    design['execution_readiness']['separate_final_receipts_to_retrieve'] = [
        'launch_finished.json', 'streaming_control_audit.jsonl', 'streaming_control_console.log']
    design_path.write_text(json.dumps(design, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    suites = {'runner_tests.py':44, 'start_once_tests.py':22,
              'evaluation_tests.py':44, 'streaming_control_libc_tests.py':50,
              'live_preflight_tests.py':28}
    code = ['runner.py','start_once.py','evaluation.py',
            'streaming_control_libc.py','live_preflight.py',
            'checkpoint_capture_probe.py','preflight_receipt.py']
    receipt = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Windows local CPU contracts and mocks only',
        tests={name:dict(passed=count,sha256=digest(ROOT/'execution'/name))
               for name,count in suites.items()},
        total_tests_passed=sum(suites.values()),
        source_hashes={name:digest(ROOT/'execution'/name) for name in code},
        remote_Linux_ABI_verified=False, server_started=False,
        simulator_or_training_executed=False,
        independent_review_file='integration_review.txt',
        independent_review_sha256=digest(ROOT/'integration_review.txt'))
    (ROOT/'execution/local_checks_receipt.json').write_text(
        json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    ledger_path = ROOT/'execution.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    assert ledger['server_started'] is False
    ledger['local_CPU_checks'] = receipt
    ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report_path = ROOT/'research_decision_ko.txt'
    report = report_path.read_text(encoding='utf-8')
    report = report.replace('로컬 준비·검증 중.', '로컬 패키지 준비·CPU 검증 완료.')
    report += ('\n현재 비교의 관측 경계\n현재 evaluator는 명령 주입 직후 기존 cached observation을 유지한다.\n'
               '명령전환 첫 actor 입력에 이전 명령이 한 control step 남을 수 있으나\n'
               '이번 후보와 C0/F10 모두 동일 조건이다. 10월2일 평가는 명령관측을\n'
               '즉시 갱신했으므로 과거 Q3와 이번 Q3의 절대값을 직접 비교하지 않는다.\n'
               '\n검증 완료 범위\n로컬 CPU 테스트188개와 독립 통합 검토를 완료했다.\n'
               '실제 Linux capture/pidfd, GPU 양보/복구, 학습, 평가, 회수, OFF는 아직 미실행이다.\n')
    report_path.write_text(report,encoding='utf-8')
    print(json.dumps(dict(tests_passed=receipt['total_tests_passed'],
                         design_sha256=digest(design_path),server_started=False)))

if __name__ == '__main__':
    main()
