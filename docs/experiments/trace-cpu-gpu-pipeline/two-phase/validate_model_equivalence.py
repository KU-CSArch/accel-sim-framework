#!/usr/bin/env python3
"""Replay the same saved IMAD records via original text and deferred tracez writers."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]).resolve();m=json.loads((r/'experiment.json').read_text());t=Path(m['traces'])
w=Path(m['worktree']);out=r/'model-equivalence';out.mkdir()
source=t/'model-capture/imad-c1-o0/one-warp-128-rounds/traces'
reference=t/'model-equivalence-original-text';reference.mkdir()
for p in source.glob('*.records'):
 dst=reference/p.name.removesuffix('.records')
 z=subprocess.run([str(r/'reference-writer'),str(p),str(dst)],capture_output=True,text=True)
 assert z.returncode==0,z.stderr
for p in source.glob('kernelslist_ctx*'):
 (reference/p.name).write_text(p.read_text().replace('.trace.xz','.trace'))
with(out/'postprocess.stdout.log').open('w')as stdout,(out/'postprocess.stderr.log').open('w')as stderr:
 z=subprocess.run([str(r/'inputs/post-traces-processing'),str(reference),'-j','2','--text'],stdout=stdout,stderr=stderr)
assert z.returncode==0
trace_paths={'original-text':str(Path('/trace')/reference.relative_to(t)/'kernelslist.g'),'deferred-tracez':str(Path('/trace')/source.relative_to(t)/'kernelslist.g')}
def run(item):
 mode,rep=item;d=out/mode/f'replay-{rep:02d}';d.mkdir(parents=True)
 shutil.copy2(r/'inputs/config/gpgpusim.config',d/'gpgpusim.config');shutil.copy2(r/'inputs/trace.config',d/'trace.config')
 argv=['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}','--volume',f'{r}:/results','--volume',f'{t}:/trace:ro','--workdir',str(Path('/results')/d.relative_to(r)),'--env','LD_LIBRARY_PATH=/results/inputs/runtime','--env','CUDA_INSTALL_PATH=/usr/local/cuda','--env','CUDA_VISIBLE_DEVICES=',m['image'],'/results/inputs/accel-sim.out','-config','gpgpusim.config','-config','trace.config','-trace',trace_paths[mode]]
 before=time.monotonic_ns()
 with(d/'stdout.log').open('w')as stdout,(d/'stderr.log').open('w')as stderr:z=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=300)
 text=(d/'stdout.log').read_text();cycles=list(map(int,re.findall(r'^gpu_tot_sim_cycle\s*=\s*(\d+)',text,re.M)))
 row={'mode':mode,'replicate':rep,'argv':argv,'exit_code':z.returncode,'wall_seconds':(time.monotonic_ns()-before)/1e9,'cumulative_kernel_cycles':cycles,'instructions':list(map(int,re.findall(r'^gpu_tot_sim_insn\s*=\s*(\d+)',text,re.M)))}
 (d/'receipt.json').write_text(json.dumps(row,indent=2)+'\n')
 assert z.returncode==0 and len(cycles)==4,(d,z.returncode,cycles)
 return row
with ThreadPoolExecutor(max_workers=2)as pool:rows=list(pool.map(run,[(a,b)for a in trace_paths for b in range(4)]))
assert len({tuple(x['cumulative_kernel_cycles'])for x in rows})==1
assert len({tuple(x['instructions'])for x in rows})==1
(out/'RESULT.json').write_text(json.dumps({'status':'passed','scope':'Unchanged A6000 trace cycle model; same saved records through original text and deferred tracez. This is encoding/regression equivalence, not a claim of hardware timing accuracy.','attempts':rows},indent=2)+'\n')
print('PASS eight replays',rows[0]['cumulative_kernel_cycles'],flush=True)
