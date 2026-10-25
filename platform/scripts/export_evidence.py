"""Export only the completed manuscript evidence and frozen inference parameters."""
import hashlib
import json
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SITE = HERE / 'site'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')


def main():
    sources = {}

    def read(rel):
        p = ROOT / rel
        sources[rel] = digest(p)
        return json.loads(p.read_text())

    rq4 = read('main_final2/revisions/p2v2_rq4_fill_20261022/results/rq4_recomputed.json')
    privacy = read('main_final2/revisions/p2v2_rq4_fill_20261022/results/rq4_privacy.json')
    auction = read('main_final2/revisions/p2v2_paper_figs_20261023/data/paper_data.json')
    smpc = read('main_final2/revisions/codex_smpc_20261024/smpc_p2v2.json')
    agg = read('main_final2/revisions/agg2x2_design_20261023/results/agg2x2_result.json')
    assert rq4['runs'] == 880 and rq4['decision']['pass'] and not rq4['differences_to_codex']
    assert smpc['complete'] and not auction['smoke']
    names = {'mfg_p2v2': 'MFG-RegretNet', 'RegretNet-r2': 'RegretNet', 'PAC': 'PAC',
             'PAC-capped-legacy': 'PAC (capped)', 'VCG': 'VCG', 'VCG-capped-legacy': 'VCG (capped)',
             'CSRA': 'CSRA', 'MFG-Pricing': 'MFG-Pricing', 'Posted-price': 'Posted price',
             'Lin': 'Lin et al.', 'DM-RegretNet': 'DM-RegretNet'}
    groups = defaultdict(list)
    for r in rq4['rows']:
        if r['arm'] in names:
            groups[(r['dataset'], r['alpha'], r['mode'], r['value'], r['arm'])].append(r)
    cells = []
    for entry in rq4['decision']['per_cell']:
        cell = entry['cell']
        arms = []
        for arm, name in names.items():
            rows = groups[(*cell, arm)]
            valid = len(rows) == 3 and {r['seed'] for r in rows} == {231, 232, 233} and all(r['matched'] for r in rows)
            vals = [r['accuracy'] * 100 for r in sorted(rows, key=lambda r:r['seed'])] if valid else []
            priv = privacy.get(str(tuple(cell)) + '|' + arm)
            arms.append(dict(id=arm, name=name, available=valid, mean=statistics.mean(vals) if valid else None,
                             sd=statistics.stdev(vals) if valid else None, seeds=vals, privacy=priv,
                             epsilon=statistics.mean(r['epsilon_per_client_round'] for r in rows) if valid else None))
        cells.append(dict(dataset=cell[0], alpha=cell[1], mode=cell[2], value=cell[3], arms=arms,
                          best=entry['external_arm'], margin=statistics.mean(entry['margins_pp'])))
    incentives = [dict(name=k, **v) for k,v in auction['baselines']['rq1_B25'].items()]
    for k in (3,30):
        v = auction['ours']['rq1'][f'mfg_p2v2|test|B25|kappa{k}']['with_dividend']
        incentives.append(dict(name=f'MFG-RegretNet · κ={k}', **v))
    data = dict(version='2026-10-25', runs=880, seeds=[231,232,233], cells=cells,
                decision=rq4['decision']['datasets'], names=names, incentives=incentives,
                costs=[dict(N=r['N'], **r['measurement']['summary'], estimation_error=r['estimation_relative_error']) for r in smpc['results']],
                aggregation=dict(text_A=agg['text_A'], text_B=agg['text_B'], text_C=agg['text_C']),
                provenance=sources)
    write(SITE/'data/evidence.json', data)
    (SITE/'papers').mkdir(parents=True, exist_ok=True)
    for src, dest in [('main_m_tdsc.pdf','mfg-regretnet.pdf'), ('main_m_supp.pdf','supplement.pdf')]:
        p=ROOT/'main_final2/p2v2_version'/src
        snapshot=p.read_bytes()
        snapshot_hash=hashlib.sha256(snapshot).hexdigest()
        if digest(p)!=snapshot_hash:
            raise RuntimeError(f'{p.name} changed during capture; rerun after its build finishes')
        (SITE/'papers'/dest).write_bytes(snapshot)
        sources[str(p.relative_to(ROOT))]=snapshot_hash
    (SITE/'figures').mkdir(exist_ok=True)
    for ds in ('mnist','cifar10'):
        for kind in ('main','all','all_smoothed'):
            p=ROOT/f'main_final2/revisions/p2v2_rq4_fill_20261022/results/figures/fig_rq4_curves_{ds}_{kind}.png'
            shutil.copy2(p,SITE/'figures'/p.name)
            sources[str(p.relative_to(ROOT))]=digest(p)
    p=ROOT/'main_final2/revisions/p2v2_rq4_fill_20261022/results/rq4_curves_raw.csv'
    shutil.copy2(p,SITE/'data/rq4_curves_raw.csv')
    sources[str(p.relative_to(ROOT))]=digest(p)
    write(SITE/'data/provenance.json',sources)

    # Load the immutable, hash-checked reference used by the completed SMPC study.
    sys.path.insert(0,str(ROOT/'main_final2/revisions/codex_smpc_20261024'))
    from support import reference
    import torch
    torch.set_num_threads(2)
    model, common, timing = reference()
    from secure_mechanism import public_context
    sample=common.generate_privacy_paper_bids(10,1,1001).double()
    ctx=public_context(model,sample[0])
    network=dict(phi=[dict(weight=w.tolist(),bias=b.tolist()) for w,b in ctx['weights']], configs={},
                 checkpoint_sha256='baddeed7201f2d4dfe82bce9aeb65a0fed86d29759618cab4a7130f57a2b382b')
    fixtures=[]
    for budget in (10,25,30,50,75,100):
        for kappa in (3,30):
            ctx=public_context(model,sample[0],budget,kappa)
            network['configs'][f'{budget}:{kappa}']=dict(coef=ctx['coef'].tolist(),bias=float(ctx['bias']),tau=ctx['tau'])
            for n in (2,5,10):
                reports=common.generate_privacy_paper_bids(n,1,1001+n).double()
                with torch.no_grad():
                    outcomes=timing.ours_cached(model,reports,torch.tensor([[float(budget)]]).double(),torch.tensor([[float(kappa)]]).double())
                fixtures.append(dict(budget=budget,kappa=kappa,reports=reports[0].tolist(),expected=[x[0].tolist() for x in outcomes]))
    # Imported by the server, not served as a public asset.
    write(HERE/'lib/model.json',network)
    write(HERE/'tests/fixtures.json',fixtures)
    print('Exported 18 settings, 11 mechanisms, 5 secure cost sizes, 36 inference fixtures; source PDFs unchanged.')


if __name__=='__main__':
    main()
