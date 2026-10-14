회수 압축파일 검증·오프라인 재계산

필요 환경: Python 3.10 이상, NumPy. 현재 로컬 Python 3.10.11 / NumPy 2.2.6에서 확인.
사용 예시(PowerShell, 경로는 실제 다운로드 파일로 바꿀 것):

python -B "D:\UserData\Desktop\issac_sim\analysis\high_score_search_20261009\postprocess\validate_retrieval.py" --archive "D:\다운로드\tracking_screen_20261009_results.tar.gz" --sidecar "D:\다운로드\tracking_screen_20261009_results.tar.gz.sha256.json" --destination "D:\UserData\Desktop\issac_sim\analysis\high_score_search_20261009\postprocess\retrieved_20261009_001"

destination은 존재하지 않아야 하고 그 상위 폴더는 이미 존재해야 한다.
안전하지 않은 tar 경로·링크·특수 파일·중복·대소문자/Windows 경로 별칭은 압축 해제 전에 거부한다.
실패한 추출 폴더는 자동으로 지우거나 재사용하지 않는다. 다음 실행에는 새 destination을 지정한다.
출력: destination/validation.json, analysis.json, report_ko.txt.
추출 원본: destination/tracking_screen_20261009/.
반환 코드: 0=모든 검증 통과, 2=검증 실패. 실패의 JSON은 stderr에도 출력한다.

검증: archive/sidecar 바이트·SHA, gzip CRC, 내부 manifest의 전체 파일 집합·바이트·SHA,
두 model_1499 체크포인트 해시, 실제 YAML 설정, 최종 학습 로그·이벤트 파일,
20개 결과 JSON / 28개 정책·시나리오 / 56개 시드 결과 / 56개 NPZ의 형식과 수치.
NPZ는 allow_pickle=False로 읽고 .pt는 해시만 계산한다. 학습·시뮬레이터·정책 실행은 하지 않는다.
원래 summary의 full-window 품질을 재계산하고 MAE·steady(매 5초의 첫 0.5초 제외)를 별도로 출력한다.
생존이 끝난 뒤의 trace 샘플은 원래 평가기와 같이 채점에서 제외한다.
TensorBoard 이벤트는 존재/크기/SHA만 확인하며 내부 이벤트 레코드는 해석하지 않는다.
파일 안의 COMPLETE/복원 증거는 확인하지만 실제 runner 종료와 서버 종료는 이 스크립트가 증명하지 않는다.
성공한 resume가 과거 state.error 문자열을 보존할 수 있으므로 retained_historical_error에 기록한다.

자체 확인(합성 파일만 사용):
python -B "D:\UserData\Desktop\issac_sim\analysis\high_score_search_20261009\postprocess\tests_validator.py"
