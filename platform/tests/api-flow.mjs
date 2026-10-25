// Full HTTP/database test. Use only a test host; creates a clearly named QA room.
import assert from 'node:assert/strict';
const base=process.env.TEST_URL||'http://127.0.0.1:8091';
async function call(path,body,token){
  const r=await fetch(base+'/api/'+path,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json',Origin:base,...token?{Authorization:'Bearer '+token}:{}},...body===undefined?{}:{body:JSON.stringify(body)}});
  const d=await r.json();return {status:r.status,...d};
}
const created=await call('rooms',{title:'QA · concurrent settlement test',budget:50,kappa:3});
assert.equal(created.status,201);const id=created.room.id,host=created.token;
const post=(action,b,t)=>call(`rooms/${id}/${action}`,b,t);
// Concurrent updates must preserve both registrations.
const [a,b]=await Promise.all([post('join',{name:'QA Alpha',records:150}),post('join',{name:'QA Beta',records:300})]);
assert.equal(a.status,200);assert.equal(b.status,200);
let state=await call('rooms/'+id,undefined,host);assert.equal(state.room.participants.length,2);
assert.ok(!JSON.stringify(state).includes(a.token));assert.equal(state.room.me,null);
assert.equal((await post('open',{},a.token)).status,403);
assert.equal((await post('open',{},host)).status,409);
for(const p of state.room.participants)assert.equal((await post('verify',{id:p.id},host)).status,200);
assert.equal((await post('open',{},host)).status,200);
assert.equal((await post('join',{name:'Late joiner',records:100})).status,409);
assert.equal((await post('bid',{value:.2,cap:2},host)).status,403);
assert.equal((await post('bid',{value:-1,cap:2},a.token)).status,400);
assert.equal((await post('bid',null,a.token)).status,400);
await Promise.all([post('bid',{value:.2,cap:2},a.token),post('bid',{value:.6,cap:3},b.token)]);
state=await call('rooms/'+id,undefined,host);
assert.ok(state.room.participants.every(p=>p.submitted));
assert.ok(state.room.participants.every(p=>!('bid'in p)&&!('key'in p)));
const clears=await Promise.all([post('clear',{},host),post('clear',{},host)]);
assert.ok(clears.every(r=>r.status===200));
assert.equal(clears[0].room.clearing.spent,clears[1].room.clearing.spent);
assert.ok(!('mean'in clears[0].room.clearing));
assert.equal((await post('bid',{value:.3,cap:1},a.token)).status,409);
assert.equal((await post('accept',{confirm:false},a.token)).status,400);
const accepts=await Promise.all([post('accept',{confirm:true},a.token),post('accept',{confirm:true},a.token),post('reject',{},b.token)]);
assert.ok(accepts.every(r=>r.status===200));
state=await call('rooms/'+id,undefined,a.token);
assert.equal(state.room.status,'settled');
const own=state.room.participants.find(p=>p.id===state.room.me.id);
assert.ok(Math.abs(state.room.credited-own.outcome.payment-own.outcome.dividend)<1e-8);
assert.equal((await post('reject',{},a.token)).status,409);
const receiptState=await call('rooms/'+id,undefined,a.token);
assert.equal(receiptState.room.credited,state.room.credited);
assert.equal((await post('cancel',{},host)).status,409);
const cross=await fetch(base+'/api/rooms/'+id+'/clear',{method:'POST',headers:{'Content-Type':'application/json',Origin:'https://unrelated.invalid',Authorization:'Bearer '+host},body:'{}'});
assert.equal(cross.status,403);
console.log(JSON.stringify({passed:true,room:id,status:state.room.status,checks:['concurrent joins','host authorization','sealed bids','input validation','registration lock','idempotent clearing','parallel duplicate acceptance','zero rejected payment','persistent reload','cross-origin rejection'],credited:state.room.credited}));
