import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {clearMarket} from '../lib/mechanism.js';
const fixtures=JSON.parse(fs.readFileSync(new URL('./fixtures.json',import.meta.url)));
test('Frozen Python checkpoint parity across 36 profiles, all budgets and both κ values',()=>{
  let max=0;
  for(const f of fixtures){const r=clearMarket(f.reports,f.budget,f.kappa);
    [r.e,r.p,r.h].forEach((out,j)=>out.forEach((x,i)=>{max=Math.max(max,Math.abs(x-f.expected[j][i]));assert.ok(Math.abs(x-f.expected[j][i])<1e-8);}));
    assert.ok(r.spent<=f.budget+1e-8);
  }
  console.log('Maximum JS/Python absolute difference:',max);
});
test('Rejects nonfinite and out-of-domain reports',()=>{
  assert.throws(()=>clearMarket([[NaN,1,10],[.2,1,20]]));
  assert.throws(()=>clearMarket([[.2,6,10],[.2,1,20]]));
  assert.throws(()=>clearMarket([[.2,1,10],[.2,1,20]],999));
});
