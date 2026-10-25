import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {readdirSync,writeFileSync,mkdirSync} from 'node:fs';
const base=process.env.TEST_URL||'http://127.0.0.1:8091';
const session='pac-routes-20261025';
function browser(...args){
  const p=spawnSync('agent-browser',['--session',session,'--args','--no-sandbox','--json',...args],{encoding:'utf8',timeout:40000});
  assert.equal(p.status,0,p.stderr||p.stdout);
  const result=JSON.parse(p.stdout);
  assert.ok(result.success,JSON.stringify(result.error));
  return result.data;
}
const pages=readdirSync('site').filter(x=>x.endsWith('.html')).map(x=>x.replace('.html',''));
const results=[];
for(const width of [1440,390]){
  browser('set','viewport',String(width),width===1440?'1000':'844');
  for(const page of pages){
    browser('open',base+'/'+(page==='index'?'':page));
    browser('wait','h1');
    const r=browser('eval','({width:innerWidth,documentWidth:document.documentElement.scrollWidth,heading:document.querySelector("h1").textContent,loading:!!document.querySelector("main .loading"),body:document.body.dataset.page})').result;
    assert.equal(r.body,page);
    assert.ok(!r.loading);
    assert.ok(r.documentWidth<=width+1,`${page} overflows at ${width}px: ${r.documentWidth}`);
    results.push({page,width,heading:r.heading,overflow:false});
  }
}
mkdirSync('qa',{recursive:true});
writeFileSync('qa/browser-routes.json',JSON.stringify({base,checkedAt:new Date().toISOString(),results},null,2));
console.log(JSON.stringify({passed:true,pages:pages.length,widths:[1440,390],checks:results.length}));
