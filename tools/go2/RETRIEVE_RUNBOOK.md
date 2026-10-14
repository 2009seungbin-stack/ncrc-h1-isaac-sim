# ▶ 현재 대상 (2026-09-21 08:15 시작): 설정 D1 = C0 + `ang_vel_xy_l2` −0.08→−0.16, 2500회, **서버 11**
- 탭: `https://11.robotics.connect.or.kr/code/...` / 학습 로그 `/workspace/films/train_D1.log` / 채점 라벨 `D1_1000 D1_2000 D1_<끝>`
- 예상: 학습 끝 11:36, 묶음(`/workspace/longrun_out.tgz`) 12:10경. 아래 본문의 "26"은 "11"로, "C0"는 "D1"으로 읽는다.
- 로컬에 풀 곳: `issac_sim/longrun_D1/`. 받기 전에 **policy.pt 가 묶음에 들어 있는지** `/workspace/out` 목록으로 확인하고,
  없으면 `ckpt_keep/<run>/exported/policy.pt` 를 `/workspace/out/D1_policy.pt` 로 복사한 뒤 tgz 를 다시 만든다(다운로드는 1회뿐).
- **판정 기준(사전 확정)** — 비교 대상은 C0 의 2000회 지점: 3축 62.45 / 6축 52.81 / 생존 0.992 / vx 0.083 / roll 0.502 / pitch 0.392.
  - 채택: D1_2000 이 6축 **+1.5 이상**, 3축 **−1.0 이내**, 전체 생존 ≥ 0.99
  - 기각: vx 가중 RMSE > 0.15 (다시 서 버림) — 점수와 무관
  - 결과는 `longrun_D1/RESULT.md` + 메모리 `ncrc-r5-d1-shake-penalty.md`(신규) + MEMORY.md 한 줄. 점수 계산은 `longrun_C0/RESULT.md` 의 표와 같은 방식(3축·6축·생존·vx·roll·pitch 가중 RMSE).
- 회수 확인 후 서버 종료, 예약 삭제. **R4 제출은 하지 않는다.** 채택이어도 5000회 재학습은 사용자에게 묻고 한다.

---

# 장기 학습 결과 회수 절차 (2026-09-20 시작 run, 서버 26)

예약 작업이 깨우면 이 순서대로 한다. 사용자는 자리에 없다 — 질문하지 말고 끝까지 한다.

## 0. 준비
- `tabs_context_mcp` → code-server 탭(`https://26.robotics.connect.or.kr/code/...`)의 tabId 를 새로 얻는다(옛 id 재사용 금지).
- 탭이 없거나 /unauthorized 면: 보드(`https://robotics.connect.or.kr/board`)에서 "VS Code 열기"(좌표 실클릭) 또는 `https://26.robotics.connect.or.kr/code` 로 이동.
- 헬퍼가 사라졌을 수 있으니 매번 다시 정의한다:
```js
window.__read=async(p)=>{const r=await fetch('/code/vscode-remote-resource?path='+encodeURIComponent(p)+'&t='+Date.now()); if(!r.ok) return 'HTTP '+r.status; const t=await r.text(); return t.replace(/[0-9a-f]{16,}/g,m=>m.slice(0,8)+'..').replace(/[=&?]/g,'~')};
window.__paste=(txt)=>{const ta=document.querySelector('.xterm-helper-textarea'); ta.focus(); const dt=new DataTransfer(); dt.setData('text/plain',txt); ta.dispatchEvent(new ClipboardEvent('paste',{clipboardData:dt,bubbles:true,cancelable:true})); return 'pasted'};
```
  paste 뒤엔 터미널을 클릭해 포커스하고 Return. browser_batch 안의 wait 합계는 30초 이하.

## 1. 상태 판정 — `await window.__read('/workspace/LONGRUN_STATUS.txt')` 의 마지막 줄들
| 보이는 것 | 뜻 | 할 일 |
|---|---|---|
| `LONGRUN DONE` + `BUNDLE ... bytes` | 끝 | §2 회수 |
| `채점:` 은 있는데 DONE 없음 | 채점 중(지점당 9분) | 아무것도 하지 말고 다음 예약을 기다린다 |
| `LONGRUN C0 6000 iter` 한 줄뿐 | 학습 중 | `/workspace/films/train_C0.log` 에서 `Learning iteration` 마지막 줄만 확인하고 끝 |
| `HTTP 404/302` | 워크스페이스 삭제 또는 서버 꺼짐 | §4 |
| `!!` 줄 | 부분 실패 | §3 |

## 2. 회수 (오리진 26 의 다운로드는 **첫 1회만** 확실하다 — 다른 것을 먼저 받지 말 것)
1. 크기 확인: BUNDLE 줄의 bytes. 12MB 를 넘으면 영상을 빼고 다시 묶는다:
   `cd /workspace/out && tar czf /workspace/longrun_out.tgz --exclude='*.mp4' .`
2. 다운로드(상태 코드·바이트 수만 반환할 것):
```js
const r=await fetch('/code/vscode-remote-resource?path='+encodeURIComponent('/workspace/longrun_out.tgz')); const b=await r.arrayBuffer(); const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([b])); a.download='longrun_out.tgz'; document.body.appendChild(a); a.click(); r.status+' '+b.byteLength
```
3. 로컬 확인: `ls -la /c/Users/2009s/Downloads/longrun_out*.tgz` — 크기가 BUNDLE bytes 와 같아야 한다.
   같으면 `C:\Users\2009s\Desktop\issac_sim\longrun_C0\` 에 풀고, `A2_policy.pt` 를 `submit_A2/policy.pt` 로 복사(md5 앞 8자리 fc69b698).
4. **다운로드가 안 떨어졌으면**(차단): json 은 텍스트로 읽어 온다 —
   `await window.__read('/workspace/out/C0_<N>_ALL.json')` 을 지점마다 읽어 `longrun_C0/` 에 Write. policy.pt 는 못 받는다 →
   체크포인트는 서버에 남겨 두고(§5 의 서버 종료를 **하지 않는다**) 사용자에게 자동 다운로드 허용을 부탁하는 메시지를 남긴다.

## 3. 부분 실패 시
- export 실패(`!! export 실패`): `/workspace/submit/model_<N>.pt` 가 묶음에 들어 있다. 그대로 회수.
- 채점 실패(`!! C0_xxx/... 실패`): `/workspace/te_*.log` 끝을 읽고, 해당 지점만
  `cd /workspace/training/quadruped && ARMS="C0_<N>" RUNS="C0_<N>=<run>:<N>" bash teval_arms.sh` 로 재실행(run 이름은 `/workspace/SWEEP_RUNS.txt`).
  재실행하면 `/workspace/out` 이 비워지므로 끝난 뒤 `cp /workspace/submit/* /workspace/out/ && cd /workspace/out && tar czf /workspace/longrun_out.tgz .`
- 학습이 중간에 죽음(마지막 ckpt 가 5800 미만): 있는 ckpt 로 채점까지는 스크립트가 알아서 했을 것. 그대로 회수하고 보고.

## 4. 워크스페이스가 사라졌다면
회수할 것이 없다. 보드 '학습 파일 내역' 최신 zip(model_best, 초기 ckpt 일 수 있음)만 남는다. 서버를 새로 켜지 말고 사실만 기록하고 끝낸다.

## 5. 회수 성공 뒤
1. 로컬에서 3축·6축 점수 계산(아래) → A2(56.34 / 46.32, 생존 0.940)와 비교표 작성.
2. **교체 기준(사전 확정)**: 6000 본이 3축·6축 둘 다 A2 보다 높고 전체 생존율 ≥ 0.940 이면 제출 후보 = C0_6000, 아니면 A2 유지.
3. 결과를 `longrun_C0/RESULT.md` 와 메모리(`ncrc-r4-longrun-c0-live.md` 갱신)에 적는다.
4. 보드에 R4 제출 폼이 열렸는지 확인. **제출은 하지 않는다**(사용자 확인 필요) — 열렸다는 사실과 후보만 보고.
5. 서버 종료: 보드의 "서버 종료하기" — **회수가 로컬 파일 크기로 확인된 경우에만.** 확인 모달은 좌표 실클릭.
6. 남은 예약 작업을 CronList → CronDelete 로 전부 지운다.

### 점수 계산 (로컬)
```python
import json,math,sys
S=json.load(open(sys.argv[1],encoding='utf-8'))['scenarios']
for mode in (3,6):
    t=0
    for s in S.values():
        r,e=s['rmse'],s['rmse_extra']; m=[r['vx']**2,r['vy']**2,r['yaw']**2]
        if mode==6: m+=[e['vz']**2,e['wx']**2,e['wy']**2]
        t+=s['w']*math.exp(-math.sqrt(sum(m)/len(m)))*s['survival']
    print(mode,'축',round(t*70,2))
print('생존',round(sum(s['w']*s['survival'] for s in S.values()),3))
```
