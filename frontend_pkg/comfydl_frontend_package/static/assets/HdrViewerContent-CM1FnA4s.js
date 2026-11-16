import"./rolldown-runtime-CM1DJQSe.js";import{Dt as e,G as t,Ht as n,L as r,Ot as i,Tt as a,W as o,et as s,f as c,ht as l,lt as u,ot as d,p as f,st as p,tt as m,vt as h,x as g}from"./vendor-three-bxGCyHx3.js";import{C as _,F as v,Ft as y,G as b,H as x,Kt as S,L as C,O as w,P as T,R as E,Rt as D,S as O,Tt as k,U as A,Wt as j,Yt as M,at as N,b as P,bt as ee,ct as F,ot as te,ut as I,wt as L,yt as R}from"./vendor-vue-core-DyDHSBJn.js";import{r as z}from"./vendor-i18n-D_kNcftw.js";import{t as B}from"./Button-DlyjHT-O.js";import{n as V}from"./downloadUtil-DD6nnwkf.js";import{r as H,s as U}from"./hdrFormatUtil-DsHyljsH.js";import{t as W}from"./histogramUtil-ChFnB6_u.js";var G=[.3127,.329],K={sRGB:{red:[.64,.33],green:[.3,.6],blue:[.15,.06],white:G},"Rec.2020":{red:[.708,.292],green:[.17,.797],blue:[.131,.046],white:G}},q=Object.keys(K),J=[1,0,0,0,1,0,0,0,1];function rgbToXyz(e){let[t,n]=e.red,[r,i]=e.green,[a,o]=e.blue,[s,c]=e.white,l=s/c,u=(1-s-c)/c,d=t*(o-i)+a*(i-n)+r*(n-o),f=l*(o-i)-r*(o-1+o*(l+u))+a*(i-1+i*(l+u)),p=l*(n-o)+t*(o-1+o*(l+u))-a*(n-1+n*(l+u)),m=l*(i-n)-t*(i-1+i*(l+u))+r*(n-1+n*(l+u)),h=f/d,g=p/d,_=m/d;return[h*t,g*r,_*a,h*n,g*i,_*o,h*(1-t-n),g*(1-r-i),_*(1-a-o)]}function multiply(e,t){let n=Array.from({length:9},()=>0);for(let r=0;r<3;r++)for(let i=0;i<3;i++){let a=0;for(let n=0;n<3;n++)a+=e[r*3+n]*t[n*3+i];n[r*3+i]=a}return n}function invert(e){let[t,n,r,i,a,o,s,c,l]=e,u=t*(a*l-o*c)-n*(i*l-o*s)+r*(i*c-a*s);if(u===0)return J;let d=1/u;return[(a*l-o*c)*d,(r*c-n*l)*d,(n*o-r*a)*d,(o*s-i*l)*d,(t*l-r*s)*d,(r*i-t*o)*d,(i*c-a*s)*d,(n*s-t*c)*d,(t*a-n*i)*d]}var Y=invert(rgbToXyz(K.sRGB));function gamutToSrgbMatrix(e){return e===`sRGB`?J:multiply(Y,rgbToXyz(K[e]))}function matchesGamut(e,t){let n=K[t],r=.01;return Math.abs(e.redX-n.red[0])<r&&Math.abs(e.redY-n.red[1])<r&&Math.abs(e.greenX-n.green[0])<r&&Math.abs(e.greenY-n.green[1])<r&&Math.abs(e.blueX-n.blue[0])<r&&Math.abs(e.blueY-n.blue[1])<r&&Math.abs(e.whiteX-n.white[0])<r&&Math.abs(e.whiteY-n.white[1])<r}function detectGamutFromChromaticities(e){return e?q.find(t=>matchesGamut(e,t))??`sRGB`:`sRGB`}var X=`
out vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`,Z=`
in vec2 vUv;
out vec4 frag_color;

uniform sampler2D uImage;
uniform mat3 uGamutToSRGB;
uniform float uGain;
uniform int uChannel;
uniform bool uDither;
uniform bool uClipWarnings;
uniform vec2 uClipRange;
uniform bool uCheckerboard;
uniform float uCheckerSize;

const float CHECK_DARK_LINEAR = 0.1329;
const float CHECK_LIGHT_LINEAR = 0.3185;

float linearToS(float a) {
  float s = sign(a);
  a = abs(a);
  return s * (a < 0.0031308 ? 12.92 * a : 1.055 * pow(a, 1.0 / 2.4) - 0.055);
}

vec3 linearToSRGB(vec3 c) {
  return vec3(linearToS(c.r), linearToS(c.g), linearToS(c.b));
}

float ign(vec2 p) {
  return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

float tent(float r) {
  float rp = sqrt(2.0 * r);
  float rn = sqrt(2.0 * r + 1.0) - 1.0;
  return (r < 0.0) ? 0.5 * rn : 0.5 * rp;
}

void main() {
  vec4 texel = texture(uImage, vUv);
  vec3 mapped = uGamutToSRGB * texel.rgb;

  vec3 selected;
  if (uChannel == 1) selected = vec3(mapped.r);
  else if (uChannel == 2) selected = vec3(mapped.g);
  else if (uChannel == 3) selected = vec3(mapped.b);
  else if (uChannel == 4) selected = vec3(texel.a);
  else if (uChannel == 5)
    selected = vec3(dot(mapped, vec3(0.2126, 0.7152, 0.0722)));
  else selected = mapped;

  vec3 exposed = selected * uGain;

  vec3 composited = exposed;
  if (uCheckerboard && uChannel == 0) {
    vec2 cell = floor(gl_FragCoord.xy / uCheckerSize);
    float check = mod(cell.x + cell.y, 2.0) == 0.0
      ? CHECK_DARK_LINEAR
      : CHECK_LIGHT_LINEAR;
    composited += (1.0 - clamp(texel.a, 0.0, 1.0)) * vec3(check);
  }

  vec3 display = linearToSRGB(composited);

  if (uDither) {
    float r = ign(gl_FragCoord.xy) - 0.5;
    display += vec3(tent(r) / 255.0);
  }

  display = clamp(display, 0.0, 1.0);

  if (uClipWarnings) {
    float zebra1 =
      mod(floor((gl_FragCoord.x + gl_FragCoord.y) / 8.0), 2.0) == 0.0 ? 0.0 : 1.0;
    float zebra2 =
      mod(floor((gl_FragCoord.x - gl_FragCoord.y) / 8.0), 2.0) == 0.0 ? 0.0 : 1.0;
    bvec3 over = greaterThan(exposed, vec3(uClipRange.y));
    bvec3 under = lessThan(exposed, vec3(uClipRange.x));
    display = mix(display, vec3(zebra1), vec3(over));
    display = mix(display, vec3(zebra2), vec3(under));
  }

  frag_color = vec4(display, 1.0);
}
`;function computeImageStats(e,t,n){let r=1/0,i=-1/0,a=0,o=0,s=0,c=0,l=0;for(let u=0;u<t;u++){let t=e(u);if(Number.isNaN(t)){c++;continue}if(!Number.isFinite(t)){l++;continue}(n!==4||u%n!==3)&&(t<r&&(r=t),t>i&&(i=t),a+=t,o+=t*t,s++)}if(s===0)return{min:0,max:0,mean:0,stdDev:0,nanCount:c,infCount:l};let u=a/s,d=Math.max(0,o/s-u*u);return{min:r,max:i,mean:u,stdDev:Math.sqrt(d),nanCount:c,infCount:l}}function computeChannelHistograms(e,t,n,r=256){let i=r-1,a=new Uint32Array(r),o=new Uint32Array(r),s=new Uint32Array(r),c=new Uint32Array(r),l=n===4?new Uint32Array(r):null,accumulate=(e,t)=>{if(Number.isNaN(t))return;let n=Math.floor(Math.max(0,t)*r);e[n>i?i:n]++};for(let r=0;r+n-1<t;r+=n){let t=e(r),i=n>=3?e(r+1):t,u=n>=3?e(r+2):t;accumulate(a,t),accumulate(o,i),accumulate(s,u),l&&accumulate(l,e(r+3)),accumulate(c,n>=3?.2126*t+.7152*i+.0722*u:t)}return{r:a,g:o,b:s,a:l,luminance:c}}var WebGLViewport=class{renderer;resizeObserver=null;constructor(e){this.renderer=e}observeResize(e,t){typeof ResizeObserver>`u`||(this.resizeObserver?.disconnect(),this.resizeObserver=new ResizeObserver(()=>t()),this.resizeObserver.observe(e))}disposeRenderer(){this.resizeObserver?.disconnect(),this.resizeObserver=null,this.renderer.forceContextLoss(),this.renderer.domElement.dispatchEvent(new Event(`webglcontextlost`,{bubbles:!0,cancelable:!0})),this.renderer.dispose(),this.renderer.domElement.remove()}},Q=.05,ne=64,re=8,ie=[`rgb`,`r`,`g`,`b`,`a`,`luminance`],ae={rgb:0,r:1,g:2,b:3,a:4,luminance:5};function createLoader(e){if(H(e)?.toLowerCase().endsWith(`.hdr`))return new c;let t=new f;return t.setDataType(o),t}function makeReader(e,t){return t===1016?t=>r.fromHalfFloat(e[t]):t=>e[t]}function loadHdrTexture(e){return new Promise((t,n)=>{createLoader(e).load(e,(e,n)=>{let r=n.header?.chromaticities;t({texture:e,gamut:detectGamutFromChromaticities(r)})},void 0,n)})}function useHdrViewer(){let r=y(0),o=y(!0),c=y(!1),f=y(!0),_=y(`sRGB`),v=y(`rgb`),b=y(!0),x=y(null),S=y(null),C=y(null),w=D(null),E=y(null),O=T(()=>{let e=w.value;if(!e)return null;switch(v.value){case`r`:return e.r;case`g`:return e.g;case`b`:return e.b;case`a`:return e.a;default:return e.luminance}}),k=D(null),A=null,j=null,M=null,N=null,P=null,F=null,I=null,L=1,R=!1,z=null,B=0,V=0,H=4,U=new a,W=new n;function requestRender(){!A||R||(R=!0,requestAnimationFrame(()=>{R=!1,A&&M&&N&&A.render(M,N)}))}function containerSize(){let e=k.value;return{width:e?.clientWidth||1,height:e?.clientHeight||1}}function updateProjection(){if(!N)return;let{width:e,height:t}=containerSize(),n=.5*e/t;N.left=-n,N.right=n,N.top=.5,N.bottom=-.5,N.updateProjectionMatrix()}function fitView(){if(!N)return;let{width:e,height:t}=containerSize(),n=e/t;N.zoom=Math.min(1,n/L),N.position.set(0,0,1),N.updateProjectionMatrix(),requestRender()}function applyUniforms(){if(!P)return;P.uniforms.uGain.value=2**r.value,P.uniforms.uDither.value=o.value,P.uniforms.uClipWarnings.value=c.value,P.uniforms.uCheckerboard.value=f.value,P.uniforms.uChannel.value=ae[v.value];let e=gamutToSrgbMatrix(_.value);P.uniforms.uGamutToSRGB.value.set(e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[8]),requestRender()}function buildScene(){A=new g({antialias:!1,alpha:!1}),j=new WebGLViewport(A),A.outputColorSpace=m,A.setPixelRatio(window.devicePixelRatio),A.setClearColor(657930,1),M=new e,N=new l(-1,1,1,-1,.1,10),N.position.set(0,0,1),P=new i({glslVersion:t,vertexShader:X,fragmentShader:Z,uniforms:{uImage:{value:null},uGamutToSRGB:{value:new p},uGain:{value:1},uChannel:{value:0},uDither:{value:!0},uClipWarnings:{value:!1},uClipRange:{value:new n(0,1)},uCheckerboard:{value:!0},uCheckerSize:{value:re*A.getPixelRatio()}}}),F=new u(new h(1,1),P),M.add(F)}function resize(){if(!A)return;let{width:e,height:t}=containerSize();A.setSize(e,t,!1),updateProjection(),requestRender()}function setTexture(e){if(!P||!F)return;e.colorSpace=m,e.minFilter=s,e.magFilter=s,e.needsUpdate=!0;let{width:t,height:n,data:r}=e.image;I=e,L=t/n,F.scale.set(L,1,1),P.uniforms.uImage.value=e,S.value=`${t} x ${n}`,r&&(B=t,V=n,H=r.length/(t*n),z=makeReader(r,e.type),C.value=computeImageStats(z,r.length,H),w.value=computeChannelHistograms(z,r.length,H))}async function mount(e,t){k.value=e,b.value=!0,x.value=null;try{buildScene(),e.appendChild(A.domElement),A.domElement.classList.add(`block`,`size-full`),resize(),applyUniforms(),attachInteractions(A.domElement),j.observeResize(e,resize);let{texture:n,gamut:r}=await loadHdrTexture(t);if(!P||!F){n.dispose();return}_.value=r,setTexture(n),applyUniforms(),fitView()}catch(e){x.value=e instanceof Error?e.message:String(e),dispose()}finally{b.value=!1}}function normalizeExposure(){let e=C.value?.max??0;r.value=e>0?-Math.log2(e):0}function attachInteractions(e){e.addEventListener(`wheel`,onWheel,{passive:!1}),e.addEventListener(`pointerdown`,onPointerDown),e.addEventListener(`pointermove`,onHoverMove),e.addEventListener(`pointerleave`,onHoverLeave)}function onWheel(e){if(!N)return;e.preventDefault();let t=Math.exp(-e.deltaY*.001),n=d.clamp(N.zoom*t,Q,ne);N.zoom=n,N.updateProjectionMatrix(),requestRender()}let G=null;function onPointerDown(e){N&&(G={x:e.clientX,y:e.clientY,camX:N.position.x,camY:N.position.y},window.addEventListener(`pointermove`,onPointerMove),window.addEventListener(`pointerup`,onPointerUp))}function onPointerMove(e){if(!N||!G)return;let{height:t}=containerSize(),n=1/(t*N.zoom);N.position.x=G.camX-(e.clientX-G.x)*n,N.position.y=G.camY+(e.clientY-G.y)*n,requestRender()}function onPointerUp(){G=null,window.removeEventListener(`pointermove`,onPointerMove),window.removeEventListener(`pointerup`,onPointerUp)}function onHoverMove(e){if(!N||!F||!A||G||!z)return;let t=A.domElement.getBoundingClientRect();W.x=(e.clientX-t.left)/t.width*2-1,W.y=-((e.clientY-t.top)/t.height*2-1),U.setFromCamera(W,N);let n=U.intersectObject(F).at(0);if(!n?.uv){E.value=null;return}let r=d.clamp(Math.floor(n.uv.x*B),0,B-1),i=d.clamp(Math.floor(n.uv.y*V),0,V-1),a=(i*B+r)*H;E.value={x:r,y:V-1-i,r:z(a),g:z(a+1),b:z(a+2),a:H===4?z(a+3):null}}function onHoverLeave(){E.value=null}function dispose(){window.removeEventListener(`pointermove`,onPointerMove),window.removeEventListener(`pointerup`,onPointerUp),A&&(A.domElement.removeEventListener(`wheel`,onWheel),A.domElement.removeEventListener(`pointerdown`,onPointerDown),A.domElement.removeEventListener(`pointermove`,onHoverMove),A.domElement.removeEventListener(`pointerleave`,onHoverLeave)),j?.disposeRenderer(),I?.dispose(),P?.dispose(),F?.geometry.dispose(),A=null,j=null,M=null,N=null,P=null,F=null,I=null,z=null}return ee([r,o,c,f,_,v],applyUniforms),te(dispose),{exposureStops:r,dither:o,clipWarnings:c,checkerboard:f,gamut:_,channel:v,loading:b,error:x,dimensions:S,stats:C,histogram:O,pixel:E,mount,dispose,fitView,normalizeExposure}}var oe={class:`flex size-full bg-background`},se={class:`relative flex-1`},ce={key:0,class:`absolute inset-0 flex items-center justify-center`},le={key:1,role:`alert`,class:`absolute inset-0 flex flex-col items-center justify-center gap-2`},ue={class:`text-sm`},de={key:2,class:`absolute top-2 left-2 rounded-sm bg-background/80 px-2 py-1 font-mono text-xs`,"data-testid":`hdr-pixel-readout`},fe={class:`flex w-72 flex-col`,"data-testid":`hdr-viewer-sidebar`},pe={class:`flex-1 overflow-y-auto p-4`},me={class:`space-y-2`},he={class:`space-y-4 p-2`},ge={class:`flex flex-col gap-2`},_e=[`aria-label`],ve={class:`space-y-4 p-2`},ye={class:`flex flex-col gap-2`},be=[`aria-label`],xe=[`value`],Se={class:`flex flex-col gap-2`},Ce=[`aria-label`],we=[`value`],Te={class:`space-y-4 p-2`},Ee={class:`flex items-center gap-2`},De={for:`hdr-dither`,class:`cursor-pointer`},Oe={class:`flex items-center gap-2`},ke={for:`hdr-clip`,class:`cursor-pointer`},Ae={class:`flex items-center gap-2`},je={for:`hdr-checkerboard`,class:`cursor-pointer`},Me={key:0,class:`space-y-2 p-2`},Ne={viewBox:`0 0 1 1`,preserveAspectRatio:`none`,class:`aspect-3/2 w-full rounded-sm bg-secondary`},Pe=[`d`],$={key:1,class:`space-y-1 p-2 text-xs tabular-nums`},Fe={key:0,class:`flex justify-between`},Ie={class:`flex justify-between`},Le={class:`flex justify-between`},Re={class:`flex justify-between`},ze={class:`flex justify-between`},Be={key:1,class:`flex justify-between text-destructive`},Ve={key:2,class:`flex justify-between text-destructive`},He={class:`p-4`},Ue={class:`flex gap-2`},We=b({__name:`HdrViewerContent`,props:{imageUrl:{}},setup(e){let{t}=z(),n=useHdrViewer(),r=q,i=ie,a=R(`containerRef`),o=T(()=>{let e=n.exposureStops.value;return`${e>0?`+`:``}${e.toFixed(1)}`}),s=T(()=>n.histogram.value?W(n.histogram.value):``),c=T(()=>{switch(n.channel.value){case`r`:return`text-red-500`;case`g`:return`text-green-500`;case`b`:return`text-blue-500`;default:return`text-foreground`}}),l=T(()=>({rgb:t(`hdrViewer.channels.rgb`),r:t(`hdrViewer.channels.r`),g:t(`hdrViewer.channels.g`),b:t(`hdrViewer.channels.b`),a:t(`hdrViewer.channels.a`),luminance:t(`hdrViewer.channels.luminance`)}));function formatNum(e){return Number.isFinite(e)?Math.abs(e)>=1e3||e!==0&&Math.abs(e)<.001?e.toExponential(3):e.toFixed(4):String(e)}function handleDownload(){V(U(e.imageUrl))}return N(()=>{a.value&&n.mount(a.value,e.imageUrl)}),(e,t)=>(F(),E(`div`,oe,[v(`div`,se,[v(`div`,{ref_key:`containerRef`,ref:a,class:`absolute size-full`,"data-testid":`hdr-viewer-canvas`},null,512),j(n).loading.value?(F(),E(`div`,ce,M(e.$t(`g.loading`))+`... `,1)):j(n).error.value?(F(),E(`div`,le,[t[6]||=v(`i`,{class:`icon-[lucide--image-off] size-12`},null,-1),v(`p`,ue,M(e.$t(`hdrViewer.failedToLoad`)),1)])):C(``,!0),j(n).pixel.value?(F(),E(`div`,de,[v(`div`,null,M(j(n).pixel.value.x)+`, `+M(j(n).pixel.value.y),1),v(`div`,null,[x(M(formatNum(j(n).pixel.value.r))+` `+M(formatNum(j(n).pixel.value.g))+` `+M(formatNum(j(n).pixel.value.b))+` `,1),j(n).pixel.value.a===null?C(``,!0):(F(),E(w,{key:0},[x(M(formatNum(j(n).pixel.value.a)),1)],64))])])):C(``,!0)]),v(`div`,fe,[v(`div`,pe,[v(`div`,me,[v(`div`,he,[v(`div`,ge,[v(`label`,null,M(e.$t(`hdrViewer.exposure`))+`: `+M(o.value),1),k(v(`input`,{"onUpdate:modelValue":t[0]||=e=>j(n).exposureStops.value=e,type:`range`,min:`-10`,max:`10`,step:`0.1`,class:`w-full`,"aria-label":e.$t(`hdrViewer.exposure`)},null,8,_e),[[_,j(n).exposureStops.value,void 0,{number:!0}]])]),A(B,{variant:`secondary`,class:`w-full`,onClick:j(n).normalizeExposure},{default:L(()=>[x(M(e.$t(`hdrViewer.normalizeExposure`)),1)]),_:1},8,[`onClick`])]),v(`div`,ve,[v(`div`,ye,[v(`label`,null,M(e.$t(`hdrViewer.channel`)),1),k(v(`select`,{"onUpdate:modelValue":t[1]||=e=>j(n).channel.value=e,class:`w-full rounded-sm bg-secondary px-2 py-1`,"aria-label":e.$t(`hdrViewer.channel`)},[(F(!0),E(w,null,I(j(i),e=>(F(),E(`option`,{key:e,value:e},M(l.value[e]),9,xe))),128))],8,be),[[O,j(n).channel.value]])]),v(`div`,Se,[v(`label`,null,M(e.$t(`hdrViewer.sourceGamut`)),1),k(v(`select`,{"onUpdate:modelValue":t[2]||=e=>j(n).gamut.value=e,class:`w-full rounded-sm bg-secondary px-2 py-1`,"aria-label":e.$t(`hdrViewer.sourceGamut`)},[(F(!0),E(w,null,I(j(r),e=>(F(),E(`option`,{key:e,value:e},M(e),9,we))),128))],8,Ce),[[O,j(n).gamut.value]])])]),v(`div`,Te,[v(`div`,Ee,[k(v(`input`,{id:`hdr-dither`,"onUpdate:modelValue":t[3]||=e=>j(n).dither.value=e,type:`checkbox`,class:`size-4 cursor-pointer accent-muted-foreground`},null,512),[[P,j(n).dither.value]]),v(`label`,De,M(e.$t(`hdrViewer.dither`)),1)]),v(`div`,Oe,[k(v(`input`,{id:`hdr-clip`,"onUpdate:modelValue":t[4]||=e=>j(n).clipWarnings.value=e,type:`checkbox`,class:`size-4 cursor-pointer accent-muted-foreground`},null,512),[[P,j(n).clipWarnings.value]]),v(`label`,ke,M(e.$t(`hdrViewer.clipWarnings`)),1)]),v(`div`,Ae,[k(v(`input`,{id:`hdr-checkerboard`,"onUpdate:modelValue":t[5]||=e=>j(n).checkerboard.value=e,type:`checkbox`,class:`size-4 cursor-pointer accent-muted-foreground`},null,512),[[P,j(n).checkerboard.value]]),v(`label`,je,M(e.$t(`hdrViewer.checkerboard`)),1)])]),s.value?(F(),E(`div`,Me,[v(`label`,null,M(e.$t(`hdrViewer.histogram`)),1),(F(),E(`svg`,Ne,[v(`path`,{d:s.value,class:S(c.value),fill:`currentColor`,"fill-opacity":`0.5`,stroke:`none`},null,10,Pe)]))])):C(``,!0),j(n).stats.value?(F(),E(`div`,$,[j(n).dimensions.value?(F(),E(`div`,Fe,[v(`span`,null,M(e.$t(`hdrViewer.resolution`)),1),v(`span`,null,M(j(n).dimensions.value),1)])):C(``,!0),v(`div`,Ie,[v(`span`,null,M(e.$t(`hdrViewer.min`)),1),v(`span`,null,M(formatNum(j(n).stats.value.min)),1)]),v(`div`,Le,[v(`span`,null,M(e.$t(`hdrViewer.max`)),1),v(`span`,null,M(formatNum(j(n).stats.value.max)),1)]),v(`div`,Re,[v(`span`,null,M(e.$t(`hdrViewer.mean`)),1),v(`span`,null,M(formatNum(j(n).stats.value.mean)),1)]),v(`div`,ze,[v(`span`,null,M(e.$t(`hdrViewer.stdDev`)),1),v(`span`,null,M(formatNum(j(n).stats.value.stdDev)),1)]),j(n).stats.value.nanCount?(F(),E(`div`,Be,[v(`span`,null,M(e.$t(`hdrViewer.nan`)),1),v(`span`,null,M(j(n).stats.value.nanCount),1)])):C(``,!0),j(n).stats.value.infCount?(F(),E(`div`,Ve,[v(`span`,null,M(e.$t(`hdrViewer.inf`)),1),v(`span`,null,M(j(n).stats.value.infCount),1)])):C(``,!0)])):C(``,!0)])]),v(`div`,He,[v(`div`,Ue,[A(B,{variant:`secondary`,class:`flex-1`,onClick:j(n).fitView},{default:L(()=>[x(M(e.$t(`hdrViewer.fitView`)),1)]),_:1},8,[`onClick`]),A(B,{variant:`secondary`,class:`flex-1`,onClick:handleDownload},{default:L(()=>[x(M(e.$t(`g.downloadImage`)),1)]),_:1})])])])]))}});export{We as default};