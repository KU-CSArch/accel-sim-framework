#!/usr/bin/env python3
"""Render explicit phase costs and the real paused-CPU independence experiment."""
import csv,json,statistics,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
r=Path(sys.argv[1]).resolve();out=r/'two-phase-reduction';out.mkdir(exist_ok=True)
rows=[];summary={}
for key,file in [('instruction-batch','TIMING-BATCHES.json'),('vector-add','VECTOR-TIMING-BATCHES.json')]:
 j=json.loads((r/file).read_text());summary[key]={}
 for b in j['batches']:
  rows.append({'workload':key,'mode':b['mode'],'repeat':b['repeat'],'capture_seconds':b['capture_process_seconds'],'cpu_seconds':b['cpu_phase']['wall_seconds'],'sequential_phase_sum_seconds':b['capture_process_seconds']+b['cpu_phase']['wall_seconds'],'record_bytes':b['raw_record_bytes']})
 for mode in ['inline','two-phase']:
  a=[x for x in rows if x['workload']==key and x['mode']==mode];assert len(a)==5
  summary[key][mode]={}
  for metric in ['capture_seconds','cpu_seconds','sequential_phase_sum_seconds','record_bytes']:
   vals=[x[metric]for x in a];summary[key][mode][metric]={'median':statistics.median(vals),'min':min(vals),'max':max(vals),'values':vals}
with(out/'all-batches.csv').open('w')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7.4,2.65),layout='constrained')
colors={'inline':'0.35','two-phase':'#25658c'};positions=[-0.25,-0.125,0,0.125,0.25]
for ax,key,title in zip(axes,summary,['FADD + IMAD + SHFL; one CTA each','Vector add; 4,096 CTAs']):
 for i,metric in enumerate(['capture_seconds','cpu_seconds']):
  for mode,dx in [('inline',-0.17),('two-phase',0.17)]:
   a=summary[key][mode][metric];x=i+dx
   ax.plot([x,x],[a['min'],a['max']],color=colors[mode],lw=1)
   ax.scatter([x+v*0.17 for v in positions],a['values'],s=13,color=colors[mode],zorder=3)
   ax.plot([x-.07,x+.07],[a['median'],a['median']],lw=2,color=colors[mode])
 ax.set_xticks([0,1],['Capture process','Later CPU phase']);ax.set_ylabel('Host time per batch (s)');ax.set_ylim(bottom=0,top=32);ax.set_title(title,fontsize=9)
axes[0].legend(handles=[Patch(color=colors['inline'],label='Inline text formatting'),Patch(color=colors['two-phase'],label='Separate phases')],loc='upper right',frameon=False,fontsize=7)
for suffix in ['png','svg','pdf']:fig.savefig(out/('two-phase-costs.'+suffix),dpi=190)
plt.close(fig)
events=json.loads((r/'independence-attempt02/RESULT.json').read_text())['events'];byname={x['name']:x for x in events};origin=events[0]['monotonic_ns']
def seconds(name):return(byname[name]['monotonic_ns']-origin)/1e9
held=seconds('CPU formatter A held');released=seconds('CPU formatter A released')
fig,ax=plt.subplots(figsize=(7.4,2.2),layout='constrained')
intervals=[(4,'GPU capture A started','GPU capture A finished','0.4'),(0,'GPU capture B started','GPU capture B finished','0.4')]
for y,a,b,color in intervals:ax.broken_barh([(seconds(a),seconds(b)-seconds(a))],(y-.22,.44),facecolors=color)
for rep in range(6):
 a=seconds(f'native-{rep:02d} started');b=seconds(f'native-{rep:02d} finished');ax.broken_barh([(a,b-a)],(2-.22,.44),facecolors='#25658c',edgecolors='white',linewidth=.3)
for rep in range(8):
 a=seconds(f'ncu-held-{rep:02d} started');b=seconds(f'ncu-held-{rep:02d} finished');ax.broken_barh([(a,b-a)],(1-.22,.44),facecolors='#25658c',edgecolors='white',linewidth=.3)
start=seconds('CPU phase A started');end=seconds('CPU phase A finished')
ax.broken_barh([(start,held-start),(released,end-released)],(3-.22,.44),facecolors='0.4')
ax.broken_barh([(held,released-held)],(3-.22,.44),facecolors='white',edgecolors='0.4',hatch='////',linewidth=.7)
ax.axvline(released,color='0.45',ls=':',lw=.8);ax.text((held+released)/2,3,'CPU formatter paused deliberately',ha='center',va='center',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','pad':1})
ax.set_yticks([4,3,2,1,0],['FADD trace','CPU phase for FADD','Native: 6 processes','NCU: 8 collections','Next IMAD trace']);ax.set_ylim(-.45,4.45);ax.set_xlim(left=0);ax.set_xlabel('Host process interval from first capture launch (s)')
for suffix in ['png','svg','pdf']:fig.savefig(out/('cpu-hold-independence.'+suffix),dpi=190)
plt.close(fig)
readings=[]
for d in sorted((r/'independence-attempt02/ncu-held').glob('process-*')):
 data=list(csv.reader((d/'metrics.csv').read_text().splitlines()));header=next(i for i,x in enumerate(data)if x and x[0]=='ID');names=data[header];units=data[header+1];values=data[header+2]
 row={'process':d.name}
 for metric in ['gpu__time_duration.sum','sm__cycles_elapsed.max','smsp__inst_issued.sum']:
  index=names.index(metric);row[metric]={'unit':units[index],'value':float(values[index])}
 readings.append(row)
summary['independence']={'held_seconds':released-held,'native_processes':6,'native_launch_outputs':60,'ncu_collections':8,'ncu_launch_outputs':32,'following_gpu_capture_processes':1,'scope':'Deliberate CPU pause proves process independence; it is excluded from throughput measurements. All following jobs finish before release.'}
summary['ncu_readings']=readings
summary['scope']='Host capture and trace processing on a shared host. Five observations per setting. No claim of native instruction timing or simulation accuracy. Original single-CTA record verification overlapped part of the vector experiment; other CPU simulations and GPU1 collection were active.'
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({key:{mode:{metric:val for metric,val in data.items()if metric in ['capture_seconds','cpu_seconds','sequential_phase_sum_seconds']}for mode,data in modes.items()}for key,modes in summary.items()if key in ['instruction-batch','vector-add']},indent=2))
