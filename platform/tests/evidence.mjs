import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const root=new URL('../../',import.meta.url);
const manifest=JSON.parse(readFileSync(new URL('reproducibility/MANIFEST.json',root)));
let checked=0;
for(const [file,hash] of Object.entries(manifest)){
  if(!file.startsWith('platform/site/')&&!file.startsWith('reproducibility/evidence/'))continue;
  assert.equal(createHash('sha256').update(readFileSync(new URL(file,root))).digest('hex'),hash,file);
  checked++;
}
const d=JSON.parse(readFileSync('site/data/evidence.json'));
for(const c of d.cells){
  const arms=c.arms.filter(x=>x.available),ours=arms.find(x=>x.id==='mfg_p2v2');
  const best=arms.filter(x=>x.id!=='mfg_p2v2').sort((a,b)=>b.mean-a.mean)[0];
  assert.equal(best.id,c.best);
  assert.ok(Math.abs(ours.mean-best.mean-c.margin)<1e-9);
}
console.log(JSON.stringify({passed:true,releaseHashes:checked,cells:d.cells.length}));
