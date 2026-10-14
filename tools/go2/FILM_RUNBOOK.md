# FILM RUNBOOK — 후보를 "같은 명령"으로 찍어 눈으로 고르는 절차

2026-09-16 작성. 오늘 실측한 렌더 경로(서버 11분 사용)를 그대로 도구로 굳힌 것이다.

## 0. 왜 이 방식인가

- **play.py 영상은 후보 비교에 못 쓴다.** env 마다 명령을 무작위로 뽑아서, 후보가 아니라 주사위를
  비교하게 된다. `render_go2.py` 는 t_eval_go2 의 G1~G7 명령 스케줄을 고정해 넣는다.
- **눈이 잡는 것 / 숫자가 잡는 것은 다르다.** 09-13 스크리닝에서 S1 은 낙상비 0.06 으로 GB(0.19)의
  1/3 이었지만, 09-16 영상에서 12초 내내 **누워만 있었다.** 누운 로봇은 넘어지지 않는다.
  집계 지표는 *안정적 보행*과 *퇴화한 정지*를 구분 못 한다 — 영상은 1초면 구분한다.
  → 영상 = 예선(퇴화 제거), `t_eval_go2.py` = 결선(순위).
- **resume 분기로 비용을 1/4 로.** scratch 8개 = 24시간, 트렁크 1회 + 분기 8개 ≈ 9시간.
  단 **제출본은 우승 값으로 scratch 재학습**한다(설계의도 20점 방어).

## 1. 설치 (서버 켠 직후, 붙여넣기 한 번)

`tools/go2/film_paste.txt` 전체를 code-server 터미널에 paste → Return.
(12,958자 / 64줄. 붙여넣기 방법은 메모리 `ncrc-server-file-transfer-via-paste-event`)

설치되는 것: `/workspace/training/quadruped/render_go2.py`, `sweep_go2.sh`
검증: 끝에 md5 가 찍힌다. 로컬과 같아야 한다.

    render_go2.py  89beb0c24ca923d417c8acc4965f278f
    sweep_go2.sh   a8de1ecb188a4568ed7cc16e127e91d3

## 2. 영상 한 편만 급히 필요할 때

```bash
cd /workspace/training/quadruped
/workspace/IsaacLab/isaaclab.sh -p render_go2.py --task Quadruped-v0 \
    --checkpoint /workspace/training/quadruped/ckpt_keep/<run>/model_2000.pt \
    --scenario G1 --label GB --num_envs 32 --headless --enable_cameras --video_dir /workspace/films
```

약 100초(부팅 75초 + 렌더 20초). 결과: `films/GB_G1.mp4` + `GB_G1.json`(구간별 명령 vs 실제 속도).
전진 명령에 실제 vx 가 0 근처면 로그에 **"전진 거부" 경고**가 찍힌다.

체크포인트가 없으면 보드 백업에서 되살린다 → `s1_export.sh` 의 ZIP_URL 방식.
(2026-09-11 18:51 백업 = GB iter1800, md5 `e363b10e` — 네 발로 걷는 유일한 Go2 체크포인트)

## 3. 스윕 (트렁크 → 분기 → 영상 → 회수)

```bash
# 0단계: GPU 안 쓰고 train.py 인자 정찰 + 보상 파일 검증만
bash sweep_go2.sh

# 확인 후 본실행 (예: 트렁크 1500 iter 새로 학습)
TRUNK_ITERS=1500 BRANCH_ITERS=800 ENVS=6144 ARMS="A0 A1 A2 A3" CONFIRM=1 bash sweep_go2.sh

# 이미 트렁크가 있으면
TRUNK_RUN=2026-09-17_10-00-00 CONFIRM=1 bash sweep_go2.sh
```

- ⚠️ **train.py 를 모르는 인자로 부르지 않는다.** 2026-09-05 에 `--help` 한 번으로 진행 중이던 run 의
  중간 체크포인트 71개가 후처리에 삭제됐다. 그래서 0단계는 소스를 **읽기만** 한다.
  `--resume` 이 안 보이면 `RESUME_ARGS='...'` 로 직접 넘겨라.
- `ENVS` 는 **6144 이하.** 8192 는 iter 14 부터 PhysX 접촉 패치 한도(327,680)를 넘겨서
  후보 비교에 교란 변수를 섞는다.
- 45초마다 `ckpt_keep/` 로 복사하는 가드가 자동으로 돈다(model_best 선택 결함 대비).
- 끝나면 `/workspace/films.tgz` 하나로 묶인다 → 브라우저 remote-resource fetch 로 한 번에 회수.

기본 팔(전진 거부가 병목이므로 "가는 이득 vs 버티는 이득" 축):

| 팔 | 값 | 의도 |
|---|---|---|
| A0 | 무수정 | 대조군 |
| A1 | track_lin 1.5 | NVIDIA 원본 추종 보상 복원 |
| A2 | track_lin 2.0 | 더 강한 추종 |
| A3 | track_lin 1.5 + lin_vel_z −2.0 | 추종 + 상하동 허용(계단에서 발 들기) |
| A4 | A3 + feet_air_time 0.02 | 발 들기 자체를 보상 |
| A5 | track_lin 1.5 + ang_vel_xy −0.05 | 추종 + 흔들림 허용 |

한 번에 다 돌리지 말고 **A0/A1/A2 로 track_lin 1차원부터** 훑어라(약 3시간). 전진이 살아나는
구간을 찾은 뒤 거기서 나머지를 건다.

## 4. 로컬에서 격자로 보기

```bash
cd tools/go2
bash make_grid.sh <films 폴더>          # <label>_<scenario>.mp4 들을 격자 한 영상으로
bash make_grid.sh <films 폴더> G1       # G1 만
```

`grid_G1.mp4` 가 나오고, 이어서 각 후보의 1구간 **명령 vx vs 실제 vx / 추종 / 생존** 표가 찍힌다.
전진 거부 패턴이면 표에 `← 전진거부` 가 붙는다. 눈으로 2~3개 추린 뒤 그 후보만 `t_eval_go2.py` 로
정식 채점한다.

## 5. 예산

| 항목 | 시간 |
|---|---|
| 잔여 서버 예산(2026-09-16 기준) | 69시간 39분 |
| 트렁크 1500 iter @6144 | 약 2.5시간(추정) |
| 분기 1개 800 iter + 영상 2편 | 약 55분(추정) |
| 우승 값 scratch 재학습 | 약 5시간 |

추정치는 S1 실측(8192 envs 6.6 s/iter)과 GB 실측(12288 envs 9.2 s/iter)에서 외삽한 것이다.
6144 envs 는 아직 실측이 없다 — 첫 트렁크에서 s/iter 를 확인하고 이 표를 갱신할 것.
