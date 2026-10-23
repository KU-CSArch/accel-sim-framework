#!/usr/bin/env python3
"""Hold an actual CPU formatter while real native, NCU and GPU trace jobs finish."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

r=Path(__file__).resolve().parent;m=json.loads((r/'experiment.json').read_text());w=Path(m['worktree']);traces=Path(m['traces'])

env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'inputs/native-runtime')
for key in ['LD_PRELOAD','CUDA_INJECTION64_PATH','CUDA_FORCE_PTX_JIT']:env.pop(key,None)
env.update({'EXCLUDE_PRED_OFF':'0','TRACE_FILE_COMPRESS':'1','TOOL_COMPRESS':'0','ALLOW_REG_VAL_TRACING':'0','ENABLE_WATCHDOG':'0','SPINLOCK_HANDLING_MODE':'0','TRACE_LINEINFO':'0','TOOL_TRACE_CORE':'0','ACTIVE_FROM_START':'1','USER_DEFINED_FOLDERS':'1'})
out=r/'independence-attempt02';out.mkdir()
events=[]
def event(name,**values):
    events.append({'name':name,'monotonic_ns':time.monotonic_ns(),'utc':datetime.now(timezone.utc).isoformat(),**values})
    (out/'events.json').write_text(json.dumps(events,indent=2)+'\n')
def command(argv,label,cwd=w,environment=env):
    event(label+' started',argv=list(map(str,argv)))
    with(out/(label+'.stdout.log')).open('w')as stdout,(out/(label+'.stderr.log')).open('w')as stderr:
        z=subprocess.run(argv,cwd=cwd,env=environment,stdout=stdout,stderr=stderr,timeout=600)
    event(label+' finished',exit_code=z.returncode)
    assert z.returncode==0,(label,z.returncode)
    return z

def gpu_arguments(name,suite):
    return [sys.executable,str(w/'util/tracer_nvbit/run_hw_trace.py'),'--phase','gpu','--results-dir',str(out/name),'--traces-dir',str(traces/'independence-attempt02'/name),'-B',suite,'-D',m['gpu']]

def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    return {'pid':pid,'start_ticks':(p/'stat').read_text().split(') ',1)[1].split()[19],'command':(p/'cmdline').read_bytes().decode()}

def descendants(pid):
    result=[];pending=[pid];seen=set()
    while pending:
        parent=pending.pop()
        if parent in seen:continue
        seen.add(parent)
        task_dir=Path('/proc')/str(parent)/'task'
        if not task_dir.exists():continue
        for task in task_dir.iterdir():
            try:children=(task/'children').read_text().split()
            except FileNotFoundError:continue
            for child in children:
                value=int(child);result.append(value);pending.append(value)
    return result

command(gpu_arguments('capture-A','trace-pipeline-validation:fadd-c1-o0'),'GPU capture A')
assert not list((traces/'independence-attempt02/capture-A').rglob('*.trace.xz'))
cpu_argv=[sys.executable,str(w/'util/tracer_nvbit/run_hw_trace.py'),'--phase','cpu','--results-dir',str(out/'capture-A')]
held=None;cpu=None
try:
    cpu_out=(out/'cpu-A.stdout.log').open('w');cpu_err=(out/'cpu-A.stderr.log').open('w')
    cpu=subprocess.Popen(cpu_argv,cwd=w,env=env,stdout=cpu_out,stderr=cpu_err,start_new_session=True)
    event('CPU phase A started',pid=cpu.pid,argv=cpu_argv)
    deadline=time.monotonic()+15
    while held is None:
        for pid in descendants(cpu.pid):
            current=identity(pid)
            if current and current['command'].startswith(str(w/'util/tracer_nvbit/tracer_tool/formatting/format-trace-capture')+'\x00'):
                assert str(traces/'independence-attempt02/capture-A')in current['command']
                os.kill(pid,signal.SIGSTOP);held=current;break
        if held is None:
            assert cpu.poll()is None,'CPU phase ended before the hold experiment'
            if time.monotonic()>deadline:raise TimeoutError('Formatter process was not found')
            time.sleep(0.002)
    event('CPU formatter A held',identity=held)
    assert (Path('/proc')/str(held['pid'])/'stat').read_text().split(') ',1)[1].split()[0]=='T'
    # Six actual native processes, each with three warmups and seven measured launches.
    native_env=env.copy();native_env['CUDA_VISIBLE_DEVICES']=m['gpu']
    for rep in range(6):
        d=out/'native'/f'process-{rep:02d}';d.mkdir(parents=True)
        command([str(r/'inputs/fadd-c1-o0'),'1','2048','1','1','7','3'],f'native-{rep:02d}',cwd=d,environment=native_env)
        gold=(d/'golden.bin').read_bytes();files=list(d.glob('output-launch-*.bin'));assert len(files)==10 and all(p.read_bytes()==gold for p in files)
    # Eight actual NCU processes. No tracer is preloaded into these applications.
    for rep in range(8):
        d=out/'ncu-held'/f'process-{rep:02d}';d.mkdir(parents=True)
        argv=['/usr/local/cuda-13.0/bin/ncu','--clock-control','none','--cache-control','none','--metrics','gpu__time_duration.sum,sm__cycles_elapsed.max,smsp__inst_issued.sum','--launch-skip','3','--launch-count','1','--csv','--log-file',str(d/'ncu.csv'),'--export',str(d/'profile'),str(r/'inputs/fadd-c1-o0'),'1','2048','1','1','1','3']
        command(argv,f'ncu-held-{rep:02d}',cwd=d,environment=native_env)
        gold=(d/'golden.bin').read_bytes();files=list(d.glob('output-launch-*.bin'));assert len(files)==4 and all(p.read_bytes()==gold for p in files)
    command(gpu_arguments('capture-B','trace-pipeline-validation:imad-c1-o0'),'GPU capture B')
    current=identity(held['pid']);assert current==held and cpu.poll()is None
    assert (Path('/proc')/str(held['pid'])/'stat').read_text().split(') ',1)[1].split()[0]=='T'
    event('All following GPU jobs completed while CPU formatter A remained held',native_processes=6,native_outputs=60,ncu_processes=8,ncu_outputs=32,next_trace_captures=1)
finally:
    if held and identity(held['pid'])==held:
        os.kill(held['pid'],signal.SIGCONT);event('CPU formatter A released',pid=held['pid'])
    if cpu:
        try:code=cpu.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(cpu.pid,signal.SIGTERM);code=cpu.wait(timeout=20)
        event('CPU phase A finished',exit_code=code)
        cpu_out.close();cpu_err.close()
        assert code==0,code
command([sys.executable,str(w/'util/tracer_nvbit/run_hw_trace.py'),'--phase','cpu','--results-dir',str(out/'capture-B')],'CPU phase B')
(out/'RESULT.json').write_text(json.dumps({'passed':True,'scope':'Causal phase independence. The CPU pause is deliberate and is excluded from throughput measurements.','events':events},indent=2)+'\n')
print('Held real CPU formatter; six native processes, eight NCU processes and the next GPU capture finished before CPU release.',flush=True)
