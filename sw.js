const C='xiaoji-v1';
self.addEventListener('install',e=>{e.waitUntil(caches.open(C).then(c=>c.addAll(['./','index.html'])).then(()=>self.skipWaiting()));});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==C).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));});
self.addEventListener('fetch',e=>{
  const req=e.request;
  if(req.method!=='GET')return; // 不攔截 API POST（Anthropic / Ollama）
  e.respondWith(
    fetch(req).then(res=>{
      if(res&&res.status===200&&req.url.startsWith(self.location.origin)){
        const cp=res.clone();caches.open(C).then(c=>c.put(req,cp));
      }
      return res;
    }).catch(()=>caches.match(req).then(m=>m||caches.match('index.html')))
  );
});
