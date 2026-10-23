#!/usr/bin/env python3
"""Five independent batches for inline formatting and explicit GPU/CPU phases."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

r=Path(__file__).resolve().parent;m=json.loads((r/'experiment.json').read_text());w=Path(m['worktree']);t=Path(m['traces'])
sys.path.insert(0,str(w/'util/tracer_nvbit'))
from trace_pipeline import run_phase

def environment():
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'inputs/native-runtime')
    for key in ['LD_PRELOAD','CUDA_INJECTION64_PATH','CUDA_FORCE_PTX_JIT']:env.pop(key,None)
    env.update({'EXCLUDE_PRED_OFF':'0','TRACE_FILE_COMPRESS':'1','TOOL_COMPRESS':'0','ALLOW_REG_VAL_TRACING':'0','ENABLE_WATCHDOG':'0','SPINLOCK_HANDLING_MODE':'0','TRACE_LINEINFO':'0','TOOL_TRACE_CORE':'0','ACTIVE_FROM_START':'1','USER_DEFINED_FOLDERS':'1'})
    return env

def invoke(argv,stem,env):
    before=time.monotonic_ns();started=datetime.now(timezone.utc).isoformat()
    with(stem.with_suffix('.stdout.log')).open('w')as stdout,(stem.with_suffix('.stderr.log')).open('w')as stderr:
        p=subprocess.run(argv,cwd=w,env=env,stdout=stdout,stderr=stderr,timeout=600)
    receipt={'argv':list(map(str,argv)),'exit_code':p.returncode,'started_utc':started,'started_monotonic_ns':before,'ended_monotonic_ns':time.monotonic_ns()}
    receipt['wall_seconds']=(receipt['ended_monotonic_ns']-before)/1e9
    stem.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert p.returncode==0,(stem,p.returncode)
    return receipt

def verify(directory):
    items=[]
    for binary in m['cases']:
        case=directory/binary/'one-warp-2048-rounds'
        records=[json.loads(line)for line in(case/'gpu.stdout.log').read_text().splitlines()if line.startswith('{"experiment":')]
        assert len(records)==1 and len(records[0]['trials'])==4
        gold=(case/'golden.bin').read_bytes()
        for trial in records[0]['trials']:
            assert trial['mismatches']==0 and(case/f"output-launch-{trial['launch']}.bin").read_bytes()==gold
        items.append({'binary':binary,'record':records[0],'golden_sha256':hashlib.sha256(gold).hexdigest(),'receipt':json.loads((case/'gpu.receipt.json').read_text())})
    return items

summary={'scope':'Host capture/CPU trace-processing cost. No native instruction timing claim.','batches':[],'source_hashes':{str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [w/'util/tracer_nvbit/run_hw_trace.py',w/'util/tracer_nvbit/trace_pipeline.py',w/'util/tracer_nvbit/tracer_tool/tracer_tool.so',w/'util/tracer_nvbit/tracer_tool/formatting/format-trace-capture']}}
env=environment()
for repeat in range(5):
    for mode in (['inline','two-phase']if repeat%2==0 else['two-phase','inline']):
        name=f'repeat-{repeat:02d}-{mode}';directory=r/'timings'/name;trace=t/'timings'/name
        directory.mkdir(parents=True);trace.mkdir(parents=True)
        if mode=='two-phase':
            argv=[sys.executable,str(w/'util/tracer_nvbit/run_hw_trace.py'),'--phase','gpu','--results-dir',str(directory),'--traces-dir',str(trace),'-B','trace-pipeline-validation','-D',m['gpu']]
            gpu=invoke(argv,directory/'gpu-phase',env)
            assert not list(trace.rglob('*.trace.xz')) and not(directory/'cpu-traces.json').exists()
        else:
            cases=[];begin=time.monotonic_ns()
            for binary in m['cases']:
                case=directory/binary/'one-warp-2048-rounds';case.mkdir(parents=True);tr=trace/binary/'one-warp-2048-rounds';(tr/'traces').mkdir(parents=True)
                exports={**{key:env[key]for key in ['LD_LIBRARY_PATH','EXCLUDE_PRED_OFF','TRACE_FILE_COMPRESS','TOOL_COMPRESS','ALLOW_REG_VAL_TRACING','ENABLE_WATCHDOG','SPINLOCK_HANDLING_MODE','TRACE_LINEINFO','TOOL_TRACE_CORE','ACTIVE_FROM_START','USER_DEFINED_FOLDERS']},'CUDA_VISIBLE_DEVICES':m['gpu'],'TRACES_FOLDER':str(tr),'TRACE_DEFER_FORMATTING':'0','CUDA_INJECTION64_PATH':str(w/'util/tracer_nvbit/tracer_tool/tracer_tool.so')}
                script='set -e\n'+''.join('export '+key+'='+shlex.quote(value)+'\n'for key,value in exports.items())+'exec '+shlex.join([str(r/'inputs'/binary),'1','2048','1','1','1','3'])+'\n'
                (case/'run_inline.sh').write_text(script);receipt=run_phase(case,'run_inline.sh','gpu');cases.append({'name':str(Path(binary)/'one-warp-2048-rounds'),'directory':str(case),'traces':str(tr/'traces'),'status':'collected','receipt':receipt})
            gpu={'wall_seconds':(time.monotonic_ns()-begin)/1e9}
            (directory/'gpu-traces.json').write_text(json.dumps({'phase':'inline-formatting-baseline','cases':cases,'completed':True},indent=2)+'\n')
        verified=verify(directory)
        cpu=invoke([sys.executable,str(w/'util/tracer_nvbit/run_hw_trace.py'),'--phase','cpu','--results-dir',str(directory),'--cpu-workers','1','--postprocess-format','tracez'],directory/'cpu-phase',env)
        item={'repeat':repeat,'mode':mode,'directory':str(directory),'trace_directory':str(trace),'gpu_phase':gpu,'cpu_phase':cpu,'cases':verified,'capture_process_seconds':sum(c['receipt']['wall_seconds']for c in verified),'raw_record_bytes':sum(p.stat().st_size for p in trace.rglob('*.records')),'numerical_outputs_correct':12}
        summary['batches'].append(item);(r/'TIMING-BATCHES.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(name,'capture',round(item['capture_process_seconds'],3),'CPU',round(cpu['wall_seconds'],3),'all 12 outputs correct',flush=True)
print('All ten matched batches finished',flush=True)
