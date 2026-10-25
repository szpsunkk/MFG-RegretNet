import {api} from '../lib/api.js';
export default {
  async fetch(request,env,ctx){
    if(new URL(request.url).pathname.startsWith('/api/'))return api(request,env,ctx);
    return env.ASSETS.fetch(request);
  }
};
