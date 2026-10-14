"""Read-only timings; compare observed overlap against prior serial timing reference."""
import csv
import datetime as dt
import glob
import json
from pathlib import Path
import statistics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root=Path(__file__).resolve().parent
def gpu(name):
    rows=list(csv.DictReader((root/name).open()))
    def col(prefix):
        key=next(k for k in rows[0] if k.strip().startswith(prefix))
        return [float(r[key].strip().split()[0]) for r in rows]
    def timestamp(row):
        t=dt.datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f')
        return t.replace(tzinfo=dt.timezone(dt.timedelta(hours=9))).timestamp()
    return dict(samples=len(rows),gpu_mean=statistics.mean(col('utilization.gpu')),
                gpu_min=min(col('utilization.gpu')),gpu_max=max(col('utilization.gpu')),
                max_memory_mib=max(col('memory.used')),mean_power_w=statistics.mean(col('power.draw')),
                start=timestamp(rows[0]),end=timestamp(rows[-1]))

solo=gpu('profile_gpu_6144.csv')
overlap=gpu('profile_gpu_train_plus_eval.csv')
events=glob.glob(str(root/'preserved/F00/*/events.out.tfevents.*'))
ea=EventAccumulator(events[0],size_guidance={'scalars':0})
ea.Reload()
points=ea.Scalars('Perf/total_fps')
steps=[dict(iter=b.step,wall=b.wall_time,seconds=(b.wall_time-a.wall_time)/(b.step-a.step))
       for a,b in zip(points,points[1:]) if b.step>a.step]
def timing(begin,end):
    selected=[p['seconds'] for p in steps if begin<=p['wall']<=end]
    return dict(samples=len(selected),mean_seconds=statistics.mean(selected) if selected else None,
                median_seconds=statistics.median(selected) if selected else None)

data=dict(solo=solo,overlap=overlap,solo_training=timing(solo['start'],solo['end']),
          overlap_training=timing(overlap['start']+40,overlap['end']),
          recent_training=timing(points[-1].wall_time-60,points[-1].wall_time),
          latest_iteration=points[-1].step,
          caveat='One warmup-stage overlap test, not a steady-state hardware benchmark; resource use is not throughput.')
(root/'profile_summary.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
