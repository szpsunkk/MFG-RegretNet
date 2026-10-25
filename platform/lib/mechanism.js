import model from './model.json' with { type: 'json' };

const sum = a => a.reduce((s,x)=>s+x,0);
function embed(r) {
  let a=[1-r[0],r[1]/5,Math.log(r[2]/10)/Math.log(50)];
  for(const {weight,bias} of model.phi) {
    const out=new Array(bias.length);
    for(let i=0;i<out.length;i++) {
      let s=bias[i]; const w=weight[i];
      for(let j=0;j<a.length;j++) s+=w[j]*a[j];
      out[i]=Math.tanh(s);
    }
    a=out;
  }
  return a;
}
function project(total,shares,caps) {
  total=Math.min(total,sum(caps));
  if(total<=0)return caps.map(()=>0);
  if(total>=sum(caps))return caps.map(x=>x>=.01?x:0);
  const target=shares.map(x=>x*total), breaks=caps.map((x,i)=>x-target[i]).sort((a,b)=>a-b);
  let prefix=0,lambda=0;
  for(let j=0;j<caps.length;j++) {
    lambda=(total-sum(target)-prefix)/(caps.length-j);
    if(lambda<=breaks[j])break;
    prefix+=breaks[j];
  }
  return caps.map((c,i)=>{const e=Math.min(c,Math.max(0,target[i]+lambda));return e>=.01?e:0;});
}

export function clearMarket(reports,budget=50,kappa=3) {
  if(reports.length<2||reports.length>10)throw new Error('A room supports 2–10 participants.');
  const cfg=model.configs[`${budget}:${kappa}`];
  if(!cfg)throw new Error('Unsupported budget or κ.');
  if(reports.some(r=>r.length!==3||r.some(x=>!Number.isFinite(x))||r[0]<0||r[0]>1||r[1]<.1||r[1]>5||r[2]<10||r[2]>500))throw new Error('Report outside the registered domain.');
  const n=reports.length, sizes=reports.map(r=>r[2]), shares=sizes.map(x=>x/sum(sizes));
  const mask=shares.map(x=>x>=cfg.tau);
  if(!mask.some(Boolean))mask[shares.indexOf(Math.max(...shares))]=true;
  const selected=shares.map((x,i)=>mask[i]?x:0), denom=sum(selected);
  const pi=selected.map(x=>x/denom), caps=reports.map((r,i)=>mask[i]?r[1]:0), capSum=sum(caps);
  const embs=reports.map(embed), embSum=Array.from({length:64},(_,j)=>sum(embs.map(e=>e[j])));
  function purchase(z,cap=capSum) {
    let score=cfg.bias;for(let j=0;j<64;j++)score+=cfg.coef[j]*z[j];
    return Math.min(50/(1+Math.exp(-score)),budget,cap);
  }
  const mean=embSum.map(x=>x/n), S=purchase(mean), e=project(S,pi,caps);
  const p=[],h=[];
  for(let i=0;i<n;i++) {
    let prev=0,integral=0;
    for(let g=0;g<=64;g++) {
      const a=[reports[i][0]+(1-reports[i][0])*g/64,reports[i][1],reports[i][2]], z=embed(a);
      for(let j=0;j<64;j++)z[j]=(embSum[j]-embs[i][j]+z[j])/n;
      const own=project(purchase(z),pi,caps)[i];
      if(g>0)integral+=(prev+own)*.5;
      prev=own;
    }
    p[i]=reports[i][0]*e[i]+integral*(1-reports[i][0])/64;
    const z=embed([0,5,reports[i][2]]);
    for(let j=0;j<64;j++)z[j]=(embSum[j]-embs[i][j]+z[j])/n;
    const sbar=purchase(z,capSum-caps[i]+(mask[i]?5:0));
    h[i]=shares[i]*(budget-sbar);
  }
  if(sum(p)+sum(h)>budget+1e-7||e.some((x,i)=>x>reports[i][1]+1e-8)||p.some((x,i)=>x<reports[i][0]*e[i]-1e-8))throw new Error('Settlement invariant failed.');
  return {e,p,h,S,mean,shares,selected:mask,tau:cfg.tau,spent:sum(p)+sum(h),checkpoint:model.checkpoint_sha256};
}
