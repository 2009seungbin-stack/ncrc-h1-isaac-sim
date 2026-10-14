#!/bin/bash
# make_grid.sh — 회수한 후보 클립들을 격자 한 영상으로 합친다 (로컬 전용, ffmpeg 필요).
#
# 왜: 사람은 "이게 1.4 m/s 같나?"는 못 맞히지만, 같은 명령을 받는 후보들을 **나란히** 놓으면
#     누가 뒤처지는지 0.5초면 본다. 눈이 잘하는 건 절대 판정이 아니라 상대 비교다.
#     render_go2.py 로 찍은 클립은 명령·seed·카메라가 같으므로 격자에 붙일 자격이 있다.
#
# 사용:
#   bash make_grid.sh films/                       # films/*.mp4 전부, 시나리오별로 묶어 격자 생성
#   bash make_grid.sh films/ G1                    # G1 만
#   TILE_W=480 LABELS=0 bash make_grid.sh films/   # 라벨 없이, 작은 타일로
#
# 출력: <디렉터리>/grid_<시나리오>.mp4
set -u
SRC=${1:-films}
ONLY=${2:-}
TILE_W=${TILE_W:-640}
TILE_H=${TILE_H:-360}
LABELS=${LABELS:-1}
FONT=${FONT:-C\\:/Windows/Fonts/arial.ttf}
FPS=${FPS:-50}

command -v ffmpeg >/dev/null || { echo "!! ffmpeg 이 PATH 에 없다"; exit 1; }
[ -d "$SRC" ] || { echo "!! 디렉터리 없음: $SRC"; exit 1; }

# 시나리오 목록 = 파일명 <label>_<scenario>.mp4 의 뒷부분.
# grid_*.mp4 는 이 스크립트가 만든 결과물이다 — 다시 입력으로 빨려 들어가지 않게 뺀다.
clip_list(){ ls "$SRC"/*_"$1".mp4 2>/dev/null | grep -v '/grid_'; }
scenarios=$(ls "$SRC"/*.mp4 2>/dev/null | grep -v '/grid_' \
            | sed 's/.*_\([A-Z0-9]*\)\.mp4/\1/' | sort -u)
[ -n "$ONLY" ] && scenarios="$ONLY"
[ -n "$scenarios" ] || { echo "!! $SRC 에 <label>_<scenario>.mp4 가 없다"; exit 1; }

for sc in $scenarios; do
  clips=$(clip_list "$sc")
  n=$(echo "$clips" | grep -c .)
  [ "$n" -ge 2 ] || { echo "[$sc] 클립이 $n 개라 격자를 안 만든다"; continue; }

  cols=1; while [ $((cols*cols)) -lt "$n" ]; do cols=$((cols+1)); done
  rows=$(( (n + cols - 1) / cols ))

  inputs=""; filter=""; stack=""; layout=""; i=0
  for f in $clips; do
    label=$(basename "$f" .mp4); label=${label%_$sc}
    inputs="$inputs -i \"$f\""
    draw=""
    if [ "$LABELS" = "1" ]; then
      draw=",drawtext=fontfile='$FONT':text='$label':x=12:y=10:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8"
    fi
    filter="$filter[$i:v]scale=$TILE_W:$TILE_H,fps=$FPS,setsar=1$draw[v$i];"
    stack="$stack[v$i]"
    x=$(( (i % cols) * TILE_W )); y=$(( (i / cols) * TILE_H ))
    [ -n "$layout" ] && layout="$layout|"
    layout="$layout${x}_${y}"
    i=$((i+1))
  done

  out="$SRC/grid_$sc.mp4"
  echo "[$sc] 후보 $n 개 → ${cols}x${rows} 격자 → $out"
  eval ffmpeg -hide_banner -loglevel error $inputs \
    -filter_complex "\"${filter}${stack}xstack=inputs=$n:layout=$layout:fill=black[v]\"" \
    -map "\"[v]\"" -c:v libx264 -crf 20 -pix_fmt yuv420p -shortest -y "\"$out\"" \
    && ls -la "$out"
done

# 수치 요약 — 영상 옆에 붙여 보는 용도. 눈으로 고른 뒤 이 표로 확인한다.
PYTHONIOENCODING=utf-8 python - "$SRC" <<'PY' 2>/dev/null
import glob, json, os, sys
rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    seg0 = d["segments"][0]
    rows.append((d["scenario"], d["label"], seg0["cmd"][0], seg0["got"][0],
                 d["tracking_ref"], d["survival"]))
if rows:
    print("\n시나리오 라벨      1구간 명령vx  실제vx   추종(참고)  생존")
    for sc, lb, c, g, t, s in sorted(rows):
        flag = "  ← 전진거부" if abs(c) > 0.2 and abs(g) < 0.15 else ""
        print(f"{sc:>4} {lb:<10} {c:+9.2f} {g:+8.2f} {t:10.4f} {s:6.3f}{flag}")
PY
