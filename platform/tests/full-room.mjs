// Exercises maximum supported room size against the actual Pages API and D1.
import assert from 'node:assert/strict';
const base=process.env.TEST_URL||'http://127.0.0.1:8091';
async function call(path,body,token){
  const response=await fetch(base+'/api/'+path,{method:'POST',headers:{'Content-Type':'application/json',Origin:base,...token?{Authorization:'Bearer '+token}:{}},body:JSON.stringify(body),signal:AbortSignal.timeout(30000)});
  const data=await response.json();
  assert.ok(response.ok,JSON.stringify({status:response.status,error:data.error}));
  return data;
}
const created=await call('rooms',{title:'QA · full 10-institution market',budget:30,kappa:30});
const id=created.room.id,host=created.token,participants=[];
for(let i=0;i<10;i++){
  const own=await call(`rooms/${id}/join`,{name:`QA institution ${i+1}`,records:50+30*i});
  participants.push(own);
  await call(`rooms/${id}/verify`,{id:own.room.me.id},host);
}
await call(`rooms/${id}/open`,{},host);
for(let i=0;i<10;i++)await call(`rooms/${id}/bid`,{value:.1+.08*i,cap:1+.4*i},participants[i].token);
const start=performance.now();
const cleared=await call(`rooms/${id}/clear`,{},host);
const elapsed=performance.now()-start;
assert.equal(cleared.room.status,'awarded');
assert.equal(cleared.room.participants.length,10);
assert.ok(cleared.room.clearing.spent<=30+1e-8);
for(const p of participants)await call(`rooms/${id}/accept`,{confirm:true},p.token);
const settled=await call(`rooms/${id}/clear`,{},host);
assert.equal(settled.room.status,'settled');
assert.ok(Math.abs(settled.room.credited-cleared.room.clearing.spent)<1e-8);
console.log(JSON.stringify({passed:true,room:id,participants:10,clearWallMs:elapsed,inferenceMs:cleared.room.clearing.durationMs,credited:settled.room.credited}));
