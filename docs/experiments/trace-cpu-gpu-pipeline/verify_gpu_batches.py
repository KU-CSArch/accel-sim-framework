#!/usr/bin/env python3
"""Verify every saved output and full raw/tracez instruction string in the batch capsule."""
import collections
import csv
import hashlib
import json
import lzma
from pathlib import Path
import shutil
import subprocess
import sys

r=Path(sys.argv[1]).resolve()if len(sys.argv)>1 else Path(__file__).resolve().parent
j=json.loads((r/'GPU-BATCHES.json').read_text())
out=r/'independent-verification';out.mkdir(exist_ok=True)
rows=[];outputs=0;counts=collections.defaultdict(set)
for batch in j['batches']:
    for case in batch['cases']:
        d=Path(batch['directory'])/case['binary']/'one-warp-2048-rounds';t=d/'traces'
        for phase,expected in case['receipts'].items():
            assert json.loads((d/(phase+'.receipt.json')).read_text())==expected
        gold=(d/'golden.bin').read_bytes()
        assert hashlib.sha256(gold).hexdigest()==case['golden_sha256']
        for p in d.glob('output-launch-*.bin'):
            assert p.read_bytes()==gold;outputs+=1
        stats={}
        for p in t.glob('stats*'):
            for row in list(csv.reader(p.read_text().splitlines()))[1:]:
                stats[row[0]]=int(row[-1])
        raw=sorted(list(t.glob('kernel-*.trace.xz'))+list(t.glob('kernel-*.trace')))
        assert len(raw)==4
        for p in raw:
            temporary=out/'current-raw.txt'
            if p.suffix=='.xz':
                with lzma.open(p,'rb')as source,temporary.open('wb')as sink:
                    shutil.copyfileobj(source,sink)
                grouped=p.with_suffix('').with_name(p.with_suffix('').name+'z')
            else:
                shutil.copy2(p,temporary);grouped=p.with_name(p.name+'z')
            argv=[str(r/'verify_tracez'),str(temporary),str(grouped)]
            z=subprocess.run(argv,text=True,capture_output=True)
            record={'batch':Path(batch['directory']).name,'binary':case['binary'],'raw':str(p.resolve()),'grouped':str(grouped.resolve()),'argv':argv,'exit_code':z.returncode,'stdout':z.stdout,'stderr':z.stderr}
            rows.append(record)
            if z.returncode:
                (out/'FAILURE.json').write_text(json.dumps(record,indent=2)+'\n')
                raise RuntimeError(record)
            record['matching_instruction_records']=int(z.stdout.strip())
            assert record['matching_instruction_records']==stats[p.name]
            record['raw_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            record['tracez_sha256']=hashlib.sha256(grouped.read_bytes()).hexdigest()
            counts[case['binary']].add(record['matching_instruction_records'])
assert outputs==180 and len(rows)==180 and all(len(v)==1 for v in counts.values())
report={'status':'passed','batches':len(j['batches']),'captures':45,'numerically_checked_launch_outputs':outputs,'verified_trace_kernels':len(rows),'verified_instruction_records':sum(x['matching_instruction_records']for x in rows),'instruction_counts_per_kernel':{k:sorted(v)for k,v in counts.items()},'scope':'All one-CTA/one-warp validation captures: every grouped instruction string equals its raw record, including PC, masks, registers, immediates and addresses. This is trace preservation, not timing accuracy validation.','source_identities':j['source_identities'],'kernels':rows}
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'current-raw.txt').unlink()
print(json.dumps({k:v for k,v in report.items()if k!='kernels'}))
