// Read-only verification of a published website against this release's assets.
import assert from 'node:assert/strict';
import {readFileSync,readdirSync,writeFileSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
const base=process.env.TEST_URL||'https://privacy-as-commodity.pages.dev';
const checked=[];
const files=[...readdirSync('site').filter(x=>x.endsWith('.html')),
  'app.js','auction.js','utils.js','style.css','data/evidence.json',
  'data/provenance.json','data/rq4_curves_raw.csv',
  'papers/mfg-regretnet.pdf','papers/supplement.pdf',
  ...readdirSync('site/figures').map(x=>'figures/'+x)];
for(const file of files){
  const url=base+'/'+(file==='index.html'?'':file);
  const response=await fetch(url,{signal:AbortSignal.timeout(30000)});
  assert.equal(response.status,200,url);
  const data=Buffer.from(await response.arrayBuffer());
  const sha=data=>createHash('sha256').update(data).digest('hex');
  assert.equal(sha(data),sha(readFileSync('site/'+file)),url);
  if(file==='index.html')assert.ok(response.headers.get('content-security-policy')?.includes("script-src 'self'"));
  checked.push({file,sha256:sha(data),status:response.status});
}
const health=await fetch(base+'/api/health').then(r=>r.json());
assert.equal(health.ok,true);assert.equal(health.currency,'demo credits');
mkdirSync('qa',{recursive:true});
writeFileSync('qa/public-assets.json',JSON.stringify({base,checkedAt:new Date().toISOString(),checked,health},null,2));
console.log(JSON.stringify({passed:true,base,assetHashes:checked.length,health}));
