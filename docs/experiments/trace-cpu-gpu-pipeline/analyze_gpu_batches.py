#!/usr/bin/env python3
"""Reduce saved host phase timestamps; render all batch times and one median timeline."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

r=Path(sys.argv[1]).resolve()if len(sys.argv)>1 else Path(__file__).resolve().parent
j=json.loads((r/'GPU-BATCHES.json').read_text())
conditions=['serial','overlap','overlap-uncompressed']
labels={'serial':'Sequential; raw xz','overlap':'CPU overlap; raw xz','overlap-uncompressed':'CPU overlap; raw text'}
rows=[];stats={}
for batch in j['batches']:
    assert batch['verified'] and batch['exit_code']==0
    cases=batch['cases'];caps=[x['receipts']['capture']for x in cases];cpus=[x['receipts']['postprocess']for x in cases]
    gaps=[(caps[i+1]['started_monotonic_ns']-caps[i]['ended_monotonic_ns'])/1e9 for i in range(len(caps)-1)]
    overlap=0
    for cpu in cpus:
        for cap in caps:
            overlap+=max(0,min(cpu['ended_monotonic_ns'],cap['ended_monotonic_ns'])-max(cpu['started_monotonic_ns'],cap['started_monotonic_ns']))/1e9
    row={'condition':batch['condition'],'repeat':batch['repeat'],'batch_seconds':batch['wall_seconds'],
       'capture_seconds':sum(x['wall_seconds']for x in caps),'postprocess_seconds':sum(x['wall_seconds']for x in cpus),
       'between_capture_gaps_seconds':sum(gaps),'cpu_time_overlapped_capture_seconds':overlap,
       'raw_bytes':sum(x['raw_bytes']for x in cases),'grouped_bytes':sum(x['grouped_bytes']for x in cases)}
    rows.append(row)
for c in conditions:
    v=[x for x in rows if x['condition']==c];assert len(v)==5
    stats[c]={}
    for metric in ['batch_seconds','capture_seconds','postprocess_seconds','between_capture_gaps_seconds','cpu_time_overlapped_capture_seconds','raw_bytes','grouped_bytes']:
        a=[x[metric]for x in v];stats[c][metric]={'median':statistics.median(a),'min':min(a),'max':max(a),'values':a}
base=stats['serial']['batch_seconds']['median']
for c in conditions:stats[c]['median_elapsed_reduction_percent_from_serial']=100*(base-stats[c]['batch_seconds']['median'])/base
out=r/'reduction';out.mkdir(exist_ok=True)
(out/'summary.json').write_text(json.dumps({'conditions':stats,'scope':'Host collection elapsed time and overlap of CPU postprocessing with a capture process; not GPU active cycles.','batch_count':len(rows),'gpu_capture_count':sum(len(x['cases'])for x in j['batches']),'verified_launch_outputs':4*sum(len(x['cases'])for x in j['batches'])},indent=2)+'\n')
with(out/'batch-times.csv').open('w')as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(8.0,3.0),gridspec_kw={'width_ratios':[1,1.7]},layout='constrained')
for i,c in enumerate(conditions):
    a=stats[c]['between_capture_gaps_seconds'];axes[0].plot([i,i],[a['min'],a['max']],color='0.55',lw=1)
    offsets=[-0.10,-0.05,0,0.05,0.10];axes[0].scatter([i+x for x in offsets],a['values'],s=13,color='0.25',zorder=3)
    axes[0].plot([i-0.18,i+0.18],[a['median'],a['median']],color='#185b89',lw=2)
axes[0].set_xticks(range(3),['Sequential\nraw xz','CPU overlap\nraw xz','CPU overlap\nraw text'])
axes[0].set_ylabel('Time between capture processes\nper 3-input batch (s)');axes[0].set_ylim(bottom=0);axes[0].set_title('Five independent batches per setting',fontsize=9)
ax=axes[1]
for i,c in enumerate(conditions):
    med=stats[c]['batch_seconds']['median'];batch=min((x for x in j['batches']if x['condition']==c),key=lambda x:abs(x['wall_seconds']-med));origin=batch['started_monotonic_ns'];y=2*(2-i)
    for case in batch['cases']:
        for phase,offset,color in [('capture',0.35,'0.35'),('postprocess',0,'#185b89')]:
            receipt=case['receipts'][phase];x=(receipt['started_monotonic_ns']-origin)/1e9;duration=receipt['wall_seconds']
            ax.broken_barh([(x,duration)],(y+offset,0.27),facecolors=color)
ax.set_yticks([4.3,2.3,0.3],[labels[c]for c in conditions]);ax.set_xlabel('Host time from batch launch (s)');ax.set_ylim(-0.2,5);ax.set_xlim(left=0)
ax.set_title('Capture and CPU intervals (median batch)',fontsize=9);ax.legend(handles=[Patch(color='0.35',label='Capture process'),Patch(color='#185b89',label='CPU postprocess')],loc='center right',bbox_to_anchor=(1,0.72),frameon=False,fontsize=7)
for suffix in ['png','svg','pdf']:fig.savefig(out/('trace-pipeline-throughput.'+suffix),dpi=180)
plt.close(fig)
print(json.dumps({k: {'elapsed': v['batch_seconds'], 'reduction_percent': v['median_elapsed_reduction_percent_from_serial'], 'gaps': v['between_capture_gaps_seconds'], 'overlap': v['cpu_time_overlapped_capture_seconds']} for k, v in stats.items()}))
