#!/usr/bin/env python3
"""Five rotated batches per condition: serial, overlapped, and overlapped with uncompressed capture."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

r=Path(__file__).resolve().parent
m=json.loads((r/'experiment.json').read_text());w=Path(m['worktree']);traces=Path(m['traces'])
summary={'scope':m['purpose'],'batches':[],'source_identities':{}}
for name in ['run_hw_trace.py','trace_pipeline.py']:
    p=w/'util/tracer_nvbit'/name;summary['source_identities'][name]=hashlib.sha256(p.read_bytes()).hexdigest()

def save():
    (r/'GPU-BATCHES.json').write_text(json.dumps(summary,indent=2)+'\n')

env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'inputs/native-runtime')
env.pop('LD_PRELOAD',None);env.pop('CUDA_FORCE_PTX_JIT',None)
env.update({'EXCLUDE_PRED_OFF':'0','TRACE_FILE_COMPRESS':'1','TOOL_COMPRESS':'0','ALLOW_REG_VAL_TRACING':'0',
            'ENABLE_WATCHDOG':'0','SPINLOCK_HANDLING_MODE':'0','TRACE_LINEINFO':'0','TOOL_TRACE_CORE':'0',
            'ACTIVE_FROM_START':'1','USER_DEFINED_FOLDERS':'1'})
for rep in range(5):
    labels=['serial','overlap','overlap-uncompressed']
    order=labels[rep%3:]+labels[:rep%3]
    for condition in order:
        env['TRACE_FILE_COMPRESS']='0'if condition=='overlap-uncompressed'else'1'
        name=f'repeat-{rep:02d}-{condition}';directory=r/'batches'/name;trace=traces/name
        directory.mkdir(parents=True);trace.mkdir(parents=True)
        for binary in m['cases']:
            case=directory/binary/'one-warp-2048-rounds';case.mkdir(parents=True)
            target=trace/binary/'one-warp-2048-rounds/traces';target.mkdir(parents=True)
            (case/'traces').symlink_to(target,target_is_directory=True)
        argv=[sys.executable,str(w/'util/tracer_nvbit/run_hw_trace.py'),'-B','trace-pipeline-validation',
              '-D',m['gpu'],'--spinlock_handling','none','--postprocess-workers',str(0 if condition=='serial'else 1),
              '--max-pending-postprocess','2','--postprocess-threads','2','--postprocess-format','tracez',
              '--keep-raw','--results-dir',str(directory)]
        began=time.monotonic_ns();utc=datetime.now(timezone.utc).isoformat()
        with(directory/'launcher.stdout.log').open('w')as stdout,(directory/'launcher.stderr.log').open('w')as stderr:
            p=subprocess.run(argv,cwd=w,env=env,stdout=stdout,stderr=stderr,timeout=600)
        ended=time.monotonic_ns()
        batch={'repeat':rep,'condition':condition,'argv':argv,'directory':str(directory),'trace_directory':str(trace),
               'started_utc':utc,'ended_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':(ended-began)/1e9,
               'started_monotonic_ns':began,'ended_monotonic_ns':ended,'exit_code':p.returncode,'trace_file_compress':int(env['TRACE_FILE_COMPRESS']),'cases':[]}
        summary['batches'].append(batch);save()
        assert p.returncode==0,(directory,p.returncode)
        for binary in m['cases']:
            case=directory/binary/'one-warp-2048-rounds'
            receipts={phase:json.loads((case/(phase+'.receipt.json')).read_text())for phase in ['capture','postprocess']}
            assert all(x['exit_code']==0 for x in receipts.values())
            records=[json.loads(x)for x in(case/'capture.stdout.log').read_text().splitlines()if x.startswith('{"experiment":')]
            assert len(records)==1,(case,records)
            record=records[0];assert len(record['trials'])==4
            golden=(case/'golden.bin').read_bytes()
            for trial in record['trials']:
                assert trial['mismatches']==0,(case,trial)
                assert (case/f"output-launch-{trial['launch']}.bin").read_bytes()==golden
            raw=list((case/'traces').glob('kernel-*.trace.xz'))+list((case/'traces').glob('kernel-*.trace'));grouped=list((case/'traces').glob('kernel-*.tracez'))
            assert len(raw)==len(grouped)==4,(case,len(raw),len(grouped))
            names=[x for x in(case/'traces/kernelslist.g').read_text().splitlines()if x.startswith('kernel-')]
            assert len(names)==4 and all((case/'traces'/x).stat().st_size for x in names)
            batch['cases'].append({'binary':binary,'receipts':receipts,'golden_sha256':hashlib.sha256(golden).hexdigest(),
                'capture_record':record,'raw_bytes':sum(p.stat().st_size for p in raw),
                'grouped_bytes':sum(p.stat().st_size for p in grouped),'kernel_names':names})
        batch['verified']=True;save()
        print(name,round(batch['wall_seconds'],3),'seconds, all 12 outputs correct',flush=True)
print('All fifteen batches completed',flush=True)
