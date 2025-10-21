// generator.js
export const CLASSES = ["Leaf", "Plastic", "Stone"];
export const IMG = { W: 96, H: 96 };

const off = document.createElement('canvas');
off.width = IMG.W; off.height = IMG.H;
const ctx = off.getContext('2d');

const rand = (a,b) => Math.random()*(b-a)+a;
const rint = (a,b) => Math.floor(rand(a,b+1));
const clamp = (v,a,b)=> Math.max(a, Math.min(b,v));

function reset(bg="#dbeafe"){ // лёгкий «уличный» фон (сине-серый)
  ctx.fillStyle = bg;
  ctx.fillRect(0,0,IMG.W,IMG.H);
}

function addBrightness(v){
  // v in [-50..50] ~ add constant to RGB
  if (v===0) return;
  const img = ctx.getImageData(0,0,IMG.W,IMG.H);
  const d = img.data;
  for (let i=0;i<d.length;i+=4){
    d[i]   = clamp(d[i]  + v, 0, 255);
    d[i+1] = clamp(d[i+1]+ v, 0, 255);
    d[i+2] = clamp(d[i+2]+ v, 0, 255);
  }
  ctx.putImageData(img,0,0);
}

function addNoise(level=8){
  const n = level*80;
  for (let i=0;i<n;i++){
    const x=rint(0,IMG.W-1), y=rint(0,IMG.H-1);
    const c=rint(0,35);
    ctx.fillStyle = `rgba(0,0,0,${c/255})`;
    ctx.fillRect(x,y,1,1);
  }
}

function addShadow(intensity=20){
  if (intensity<=0) return;
  const grad = ctx.createRadialGradient(rand(10,86), rand(10,86), rand(5,15), IMG.W/2, IMG.H/2, rand(70,110));
  grad.addColorStop(0, `rgba(0,0,0,${intensity/600})`);
  grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0,0,IMG.W,IMG.H);
}

/* -------- CLASS DRAWERS -------- */
function drawLeaf(){
  // базовый зелёный контур «листа» с небольшой зубчатостью + прожилки
  const cx = rand(32,64), cy = rand(32,64);
  const len = rand(24,34), wid = rand(14,22);
  const angle = rand(0,Math.PI*2);

  ctx.save();
  ctx.translate(cx,cy);
  ctx.rotate(angle);

  // контур
  ctx.fillStyle = `hsl(${rand(90,140)}, ${rint(40,70)}%, ${rint(25,40)}%)`;
  ctx.beginPath();
  ctx.moveTo(0, -len);
  for(let t=-Math.PI/2;t<=Math.PI/2;t+=0.2){
    const r = wid*(1-Math.abs(Math.sin(t))*0.3) + rand(-1.5,1.5);
    ctx.lineTo(Math.cos(t)*r, Math.sin(t)*len);
  }
  ctx.closePath(); ctx.fill();

  // прожилки
  ctx.strokeStyle = "rgba(255,255,255,0.6)";
  ctx.lineWidth = 1;
  for(let i=0;i<6;i++){
    ctx.beginPath();
    ctx.moveTo(0, -len*0.9 + i*(len*0.3));
    ctx.lineTo(rand(-wid*0.6, -wid*0.2), -len*0.5 + i*(len*0.3));
    ctx.moveTo(0, -len*0.9 + i*(len*0.3));
    ctx.lineTo(rand(wid*0.2, wid*0.6), -len*0.5 + i*(len*0.3));
    ctx.stroke();
  }
  ctx.restore();
}

function drawPlastic(){
  // яркое пятно с бликом: произвольный многоугольник + белый highlight
  const n = rint(4,7);
  const pts = [];
  const cx = rand(28,68), cy = rand(28,68);
  for(let i=0;i<n;i++){
    pts.push([ clamp(cx+rand(-24,24),0,IMG.W), clamp(cy+rand(-24,24),0,IMG.H) ]);
  }
  ctx.fillStyle = `hsl(${rint(0,360)}, ${rint(60,90)}%, ${rint(45,70)}%)`;
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for(let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath(); ctx.fill();

  // блик
  const gx = rand(10,86), gy = rand(10,86);
  const grad = ctx.createRadialGradient(gx, gy, 2, gx, gy, 28);
  grad.addColorStop(0,"rgba(255,255,255,0.7)");
  grad.addColorStop(1,"rgba(255,255,255,0)");
  ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(gx,gy,30,0,Math.PI*2); ctx.fill();
}

function drawStone(){
  // серо-бурый овал с «зерном» и неровным краем
  const cx = rand(32,64), cy = rand(32,64);
  const rx = rand(16,26), ry = rand(12,22);
  const rot = rand(0,Math.PI*2);
  ctx.save(); ctx.translate(cx,cy); ctx.rotate(rot);

  // базовая форма
  ctx.fillStyle = `hsl(${rint(20,50)}, ${rint(10,25)}%, ${rint(30,45)}%)`;
  ctx.beginPath();
  for(let a=0;a<Math.PI*2;a+=0.25){
    const rrx = rx + rand(-3,3), rry = ry + rand(-3,3);
    const x = Math.cos(a)*rrx, y = Math.sin(a)*rry;
    if (a===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.closePath(); ctx.fill();

  // зерно
  for(let i=0;i<220;i++){
    ctx.fillStyle = `rgba(255,255,255,${rand(0.02,0.08)})`;
    ctx.fillRect(rand(-rx,rx)+cx - cx, rand(-ry,ry)+cy - cy, 1,1);
  }
  ctx.restore();
}

/* -------- PIPELINE -------- */
function toTensor() {
  const img = ctx.getImageData(0,0,IMG.W,IMG.H).data;
  const arr = new Float32Array(IMG.W*IMG.H*3);
  for (let i=0,j=0; i<img.length; i+=4){
    arr[j++] = img[i]/255;     // R
    arr[j++] = img[i+1]/255;   // G
    arr[j++] = img[i+2]/255;   // B
  }
  return tf.tensor4d(arr, [1, IMG.W, IMG.H, 3]);
}
function oneHot(idx,n){ const a=new Array(n).fill(0); a[idx]=1; return a; }

export function generateOne(clsIdx, {brightness=0, shadow=20, noise=8}={}){
  reset(); // фон
  if (clsIdx===0) drawLeaf();
  else if (clsIdx===1) drawPlastic();
  else drawStone();

  addBrightness(brightness);
  addShadow(shadow);
  addNoise(noise);

  const x = toTensor();
  const y = tf.tensor2d([oneHot(clsIdx, CLASSES.length)], [1, CLASSES.length]);
  return { x, y, preview: off.toDataURL() };
}

export function generateDataset(perClass=300, opts={brightness:0, shadow:20, noise:8}, testPct=0.2){
  // используем tidy, но вернём тензоры наружу — они не очищаются tidy
  return tf.tidy(()=>{
    const xs=[], ys=[], previews=[];
    for (let c=0;c<CLASSES.length;c++){
      for (let i=0;i<perClass;i++){
        const {x,y,preview} = generateOne(c, opts);
        xs.push(x); ys.push(y); previews.push({cls:c, data:preview});
      }
    }
    const X = tf.concat(xs,0);
    const Y = tf.concat(ys,0);
    xs.forEach(t=>t.dispose()); ys.forEach(t=>t.dispose());

    const N = X.shape[0];
    const idx = tf.util.createShuffledIndices(N);
    const split = Math.floor(N*(1-testPct));
    const Xsh = tf.gather(X, idx), Ysh = tf.gather(Y, idx);
    const Xtrain = Xsh.slice([0,0,0,0],[split,IMG.W,IMG.H,3]);
    const Ytrain = Ysh.slice([0,0],[split,CLASSES.length]);
    const Xtest  = Xsh.slice([split,0,0,0],[N-split,IMG.W,IMG.H,3]);
    const Ytest  = Ysh.slice([split,0],[N-split,CLASSES.length]);

    return { Xtrain, Ytrain, Xtest, Ytest, previews };
  });
}
