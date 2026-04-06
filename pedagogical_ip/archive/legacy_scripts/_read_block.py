import csv, numpy as np, json
rows = list(csv.DictReader(open('results/block_experiment.csv')))
diffs = ['easy','medium','hard']
bnames = ['no_teacher','wait_warn','wait_block','wait_warn_block']
out = {}
for d in diffs:
    out[d] = {}
    for b in bnames:
        s = [r for r in rows if r['difficulty']==d and r['baseline']==b]
        n = len(s)
        csr = sum(int(r['success']) for r in s)/n*100
        st = np.mean([int(r['steps']) for r in s])
        tw = sum(int(r['warn_count']) for r in s)
        tb = sum(int(r['block_count']) for r in s)
        tbh = sum(int(r['block_on_hazard']) for r in s)
        out[d][b] = {
            'CSR': round(csr,1), 'Steps': round(float(st),1),
            'WarnRate': round(tw/n,3), 'BlockRate': round(tb/n,3),
            'BlockPrec': round(tbh/max(tb,1)*100,0),
            'TotalWarns': tw, 'TotalBlocks': tb
        }
with open('results/block_summary.json','w') as f:
    json.dump(out, f, indent=2)
print("DONE")
