export const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

export function toast(message){
  const el=document.querySelector('#toast');
  el.textContent=message;
  el.style.display='block';
  clearTimeout(window.toastTimer);
  window.toastTimer=setTimeout(()=>el.style.display='none',4500);
}

export function download(name,data,type='application/json'){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([typeof data==='string'?data:JSON.stringify(data,null,2)],{type}));
  a.download=name;
  a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}

export function enableScrollRegions(){
  document.querySelectorAll('.table-scroll,.audit').forEach(el=>{
    el.tabIndex=0;
    el.setAttribute('role','region');
    el.setAttribute('aria-label',el.classList.contains('audit')?'Market activity':'Scrollable results table');
  });
}
