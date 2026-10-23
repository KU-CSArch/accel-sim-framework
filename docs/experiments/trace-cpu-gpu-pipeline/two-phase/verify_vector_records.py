#!/usr/bin/env python3
"""Compare each saved two-phase record stream with the independent original writer."""
import hashlib,json,lzma,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();m=json.loads((r/'experiment.json').read_text());t=Path(m['traces'])
out=r/'vector-record-verification';out.mkdir(exist_ok=False)
rows=[]
paths=sorted(t.glob('vector-timings/**/*.records'))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
for path in paths:
 temp=out/'original-writer.txt';argv=[str(r/'reference-writer'),str(path),str(temp)]
 p=subprocess.run(argv,capture_output=True,text=True);assert p.returncode==0,(argv,p.stderr)
 raw=Path(str(path)[:-8]+'.xz');assert raw.is_file()
 h=hashlib.sha256();size=0
 with temp.open('rb')as a,lzma.open(raw,'rb')as b:
  while True:
   x=a.read(4*1024*1024);y=b.read(len(x)if x else 1);assert x==y,(path,size)
   if not x:break
   h.update(x);size+=len(x)
 row={'records':str(path),'record_sha256':sha(path),'formatted':str(raw),'text_sha256':h.hexdigest(),'text_bytes':size,'instruction_records':int(p.stdout),'original_writer_exit':p.returncode,'original_writer_argv':argv}
 grouped=Path(str(path)[:-8]+'z')
 if grouped.exists():
  z=subprocess.run([str(r/'verify_tracez_all_warps'),str(temp),str(grouped)],capture_output=True,text=True)
  assert z.returncode==0,(grouped,z.stderr)
  assert int(z.stdout)==row['instruction_records']
  row.update(grouped_sha256=sha(grouped),grouped_records=int(z.stdout))
 rows.append(row)
 (out/'verification.json').write_text(json.dumps({'status':'in-progress','kernels':rows},indent=2)+'\n')
 print('verified',len(rows),'/',len(paths),path.name,flush=True)
assert len(paths)==20,len(paths)
report={'status':'passed','scope':'Complete saved native record stream formatted by independent original writer; all output bytes compared. Grouped tracez checked for all CTAs and warps of every repeated vector case. No ISA/timing accuracy claim.','kernels':rows,'kernel_count':len(rows),'instruction_records':sum(x['instruction_records']for x in rows),'grouped_kernel_count':sum('grouped_records'in x for x in rows)}
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');temp.unlink()
print('PASS',len(rows),report['instruction_records'],flush=True)
