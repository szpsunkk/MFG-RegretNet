import {clearMarket} from './mechanism.js';

const TTL=7*24*3600*1000;
class RequestError extends Error {constructor(message,status=400){super(message);this.status=status;}}
const fail=(m,s=400)=>{throw new RequestError(m,s);};
const token=()=>crypto.randomUUID()+crypto.randomUUID();
const hash=async s=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(s)))).map(x=>x.toString(16).padStart(2,'0')).join('');
const now=()=>new Date().toISOString();
function event(s,text){s.events.push({time:now(),text});s.events=s.events.slice(-70);}
function number(v,min,max){if(typeof v!=='number'||!Number.isFinite(v)||v<min||v>max)fail(`Enter a number between ${min} and ${max}.`);return v;}
function label(v){if(typeof v!=='string'||v.trim().length<2||v.trim().length>50||/[\u0000-\u001f\u007f]/.test(v))fail('Use a name with 2–50 characters.');return v.trim();}
async function body(request){
  if(Number(request.headers.get('Content-Length')||0)>4096)fail('Request too large.',413);
  const text=await request.text();if(text.length>4096)fail('Request too large.',413);
  let data;try{data=JSON.parse(text);}catch{fail('Invalid JSON.');}
  if(!data||typeof data!=='object'||Array.isArray(data))fail('A JSON object is required.');
  return data;
}
function json(d,status=200){return Response.json(d,{status,headers:{'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'}});}

export function publicRoom(s,who) {
  const host=who===s.host, own=s.participants.find(p=>p.key===who);
  const settled=s.participants.filter(p=>p.decision==='accepted');
  return {id:s.id,title:s.title,budget:s.budget,kappa:s.kappa,status:s.status,created:s.created,expires:s.expires,
    version:s.version,role:host?'host':own?'participant':'visitor',me:own?{id:own.id,bid:own.bid??null,decision:own.decision??null,
      utility:own.outcome?own.outcome.payment+own.outcome.dividend-own.bid.value*own.outcome.epsilon:null}:null,
    participants:s.participants.map(p=>({id:p.id,name:p.name,records:p.records,verified:p.verified,submitted:!!p.bid,
      decision:p.decision??null,outcome:p.outcome??null})),events:s.events,
    clearing:s.clearing?Object.fromEntries(Object.entries(s.clearing).filter(([key])=>key!=='mean')):null,credited:settled.reduce((a,p)=>a+p.outcome.payment+p.outcome.dividend,0),
    released:settled.reduce((a,p)=>a+p.outcome.epsilon,0),
    notice:'Demonstration credits; no money, private records, or model updates are transferred.'};
}

async function rate(db,key,limit,windowMs){
  const t=Date.now(), slot=Math.floor(t/windowMs);
  const row=await db.prepare('INSERT INTO rate_limits(key,count,expires_at) VALUES(?,1,?) ON CONFLICT(key) DO UPDATE SET count=count+1 RETURNING count').bind(key+':'+slot,t+windowMs).first();
  if(row.count>limit)fail('Too many requests. Please try again shortly.',429);
}

export async function api(request,env,ctx={waitUntil:()=>{}}) {
  try {
    const url=new URL(request.url), parts=url.pathname.split('/').filter(Boolean);
    if(url.pathname==='/api/health')return json({ok:!!env.DB,version:'2026-10-25',currency:'demo credits'});
    if(!env.DB)fail('Auction storage is not configured.',503);
    if(!['GET','POST'].includes(request.method))fail('Method not allowed.',405);
    if(request.method==='POST'){
      const origin=request.headers.get('Origin');
      if(origin&&origin!==url.origin)fail('Use the auction website to submit this request.',403);
      if(!request.headers.get('Content-Type')?.startsWith('application/json'))fail('JSON required.',415);
      const ipHash=await hash(request.headers.get('CF-Connecting-IP')||'local');
      await rate(env.DB,'write:'+ipHash,160,60000);
      if(parts.length===2)await rate(env.DB,'create:'+ipHash,20,3600000);
    }
    const rawToken=request.headers.get('Authorization')?.replace(/^Bearer /,'')||'';
    if(rawToken.length>200)fail('Invalid session.',401);
    const who=rawToken?await hash(rawToken):'';
    if(parts[1]!=='rooms')fail('Not found.',404);
    if(parts.length===2&&request.method==='POST'){
      const b=await body(request);
      if(![10,25,30,50,75,100].includes(b.budget)||![3,30].includes(b.kappa))fail('Choose a supported budget and κ.');
      const secret=token(), id=crypto.randomUUID().replaceAll('-','').slice(0,12).toUpperCase();
      const s={id,title:label(b.title),budget:b.budget,kappa:b.kappa,host:await hash(secret),status:'registration',
        participants:[],events:[],created:now(),expires:Date.now()+TTL,version:1};
      event(s,'Room created. Invite institutions to register.');
      await env.DB.prepare('INSERT INTO rooms(id,state,version,expires_at) VALUES(?,?,1,?)').bind(id,JSON.stringify(s),s.expires).run();
      ctx.waitUntil(env.DB.batch([env.DB.prepare('DELETE FROM rate_limits WHERE expires_at < ?').bind(Date.now()),env.DB.prepare('DELETE FROM rooms WHERE expires_at < ?').bind(Date.now())]));
      return json({token:secret,room:publicRoom(s,s.host)},201);
    }
    const id=(parts[2]||'').toUpperCase();
    if(!/^[A-F0-9]{12}$/.test(id))fail('Enter the 12-character room code.',404);
    const action=parts[3];
    const input=request.method==='POST'?await body(request):null;
    const joinSecret=action==='join'&&!who?token():null;
    const joiningKey=joinSecret?await hash(joinSecret):who;
    for(let retry=0;retry<8;retry++){
      const row=await env.DB.prepare('SELECT state,version,expires_at FROM rooms WHERE id=?').bind(id).first();
      if(!row||row.expires_at<Date.now())fail('Room not found or expired.',404);
      const s=JSON.parse(row.state);
      if(request.method==='GET')return json({room:publicRoom(s,who)});
      const host=who===s.host, own=s.participants.find(p=>p.key===who);
      if(['open','clear','cancel','verify'].includes(action)&&!host)fail('Only the room host can do this.',403);
      if(action==='join'){
        if(s.status!=='registration')fail('Registration is closed.',409);
        if(s.participants.some(p=>p.key===joiningKey))return json({room:publicRoom(s,joiningKey)});
        if(s.participants.length>=10)fail('This room already has 10 institutions.',409);
        const name=label(input.name), records=number(input.records,10,500);
        if(!Number.isInteger(records))fail('Record count must be an integer.');
        if(s.participants.some(p=>p.name.toLowerCase()===name.toLowerCase()))fail('This institution name is already registered.',409);
        s.participants.push({id:crypto.randomUUID(),key:joiningKey,name,records,verified:false});
        event(s,`${name} registered. Awaiting host verification.`);
      }else if(action==='verify'){
        if(s.status!=='registration')fail('Registration is closed.',409);
        const p=s.participants.find(p=>p.id===input.id);if(!p)fail('Participant not found.',404);
        p.verified=true;event(s,`${p.name}: record count verified by host.`);
      }else if(action==='open'){
        if(s.status!=='registration')fail('Bidding already opened or room closed.',409);
        if(s.participants.length<2||s.participants.some(p=>!p.verified))fail('Verify at least two institutions and all registered record counts first.',409);
        s.status='bidding';event(s,'Registration frozen. Sealed bidding is open.');
      }else if(action==='bid'){
        if(s.status!=='bidding')fail('Bidding is not open.',409);
        if(!own||!own.verified)fail('Join and verify your institution first.',403);
        own.bid={value:number(input.value,0,1),cap:number(input.cap,.1,5)};
        event(s,`${own.name} submitted a sealed bid.`);
      }else if(action==='clear'){
        if(s.status==='awarded'||s.status==='settled')return json({room:publicRoom(s,who)});
        if(s.status!=='bidding'||s.participants.some(p=>!p.bid))fail('Wait for every verified participant to submit a bid.',409);
        const start=performance.now();
        const result=clearMarket(s.participants.map(p=>[p.bid.value,p.bid.cap,p.records]),s.budget,s.kappa);
        s.clearing={S:result.S,spent:result.spent,mean:result.mean,tau:result.tau,checkpoint:result.checkpoint,
          durationMs:performance.now()-start,clearedAt:now(),gridPoints:65};
        s.participants.forEach((p,i)=>{p.outcome={epsilon:result.e[i],payment:result.p[i],dividend:result.h[i],
          share:result.shares[i],selected:result.selected[i]};p.decision='pending';});
        s.status='awarded';event(s,'Auction cleared with the frozen MFG-RegretNet model. Participants can inspect and accept their contracts.');
      }else if(action==='accept'||action==='reject'){
        if(!own)fail('Participant session required.',403);
        if(!['awarded','settled'].includes(s.status))fail('No contract is available.',409);
        const decision=action==='accept'?'accepted':'rejected';
        if(own.decision===decision)return json({room:publicRoom(s,who)});
        if(own.decision!=='pending')fail('This contract has already been resolved.',409);
        if(action==='accept'&&input.confirm!==true)fail('Confirm the synthetic contribution to receive demo credits.');
        own.decision=decision;event(s,`${own.name} ${decision} the contract${action==='accept'?' · demo credits recorded':''}.`);
        if(s.participants.every(p=>p.decision!=='pending')){s.status='settled';event(s,'All contracts resolved. Final receipts are available.');}
      }else if(action==='cancel'){
        if(!['registration','bidding'].includes(s.status))fail('Awarded contracts cannot be cancelled by the host.',409);
        s.status='cancelled';event(s,'Host cancelled the room before award. No credits were transferred.');
      }else fail('Action not found.',404);
      s.version=row.version+1;
      const result=await env.DB.prepare('UPDATE rooms SET state=?,version=? WHERE id=? AND version=?').bind(JSON.stringify(s),s.version,id,row.version).run();
      if(result.meta.changes===1)return json({...joinSecret?{token:joinSecret}:{},room:publicRoom(s,action==='join'?joiningKey:who)});
    }
    fail('The room changed concurrently. Please retry.',409);
  }catch(e){return json({error:e instanceof RequestError?e.message:'The auction service could not complete this request.'},e.status||500);}
}
