// generator.js
// Генерация синтетических изображений 96×96×3 для классов: Leaf / Plastic / Stone

export const CLASSES = ["Leaf", "Plastic", "Stone"];
export const IMG = { W: 96, H: 96 };
// 👉 Алиас для обратной совместимости (исправляет ошибку импорта)
export const IMG_SIZE = IMG.W;

const off = document.createElement('canvas');
off.width = IMG.W; off.height = IMG.H;
const ctx = off.getContext('2d');

const rand  = (a,b) => Math.random()*(b-a)+a;
const rint  = (a,b) => Math.floor(rand(a,b+1));
const clamp = (v,a,b)=> Math.max(a, Math.min(b,v));

/* ==================== базовые эффекты / утилиты ==================== */
function reset(bg="#dbeafe"){ // «уличный» серо-синий фон
  ctx.fillStyle = bg;
  ctx.fillRect(0,0,IMG.W,IMG.H);
}

function addBrightness(v){ // v ∈ [-50..50]
  if (!v) return;
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
  const n = level * 80;
  for (let i=0;i<n;i++){
    const x=rint(0,IMG.W-1), y=rint(0,IMG.H-1);
    const a = Math.random()*0.15;
    ctx.fillStyle = `rgba(0,0,0,${a})`;
    ctx.fillRect(x,y,1,1);
  }
}

function addShadow(intensity=20){
  if (intensity<=0) return;
  const gx = rand(10, IMG.W-10), gy = rand(10, IMG.H-10);
  const outer = Math.max(IMG.W, IMG.H) * rand(0.7, 1.2);
  const grad = ctx.createRadialGradient(gx, gy, rand(4,12), IMG.W/2, IMG.H/2, outer);
  grad.addColorStop(0, `rgba(0,0,0,${clamp(intensity/600, 0, 0.25)})`);
  grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0,0,IMG.W,IMG.H);
}

/* ==================== рисовалки классов ==================== */
function drawLeaf(){
  const cx = rand(30,66), cy = rand(30,66);
  const len = rand(24,34), wid = rand(14,22);
  const ang = rand(0, Math.PI*2);

  ctx.save(); ctx.translate(cx,cy); ctx.rotate(ang);

  // тело листа
  ctx.fillStyle = `hsl(${rand(90,140)}, ${rint(45,75)}%, ${rint(25,40)}%)`;
  ctx.beginPath();
  ctx.moveTo(0, -len);
  for (let t=-Math.PI/2; t<=Math.PI/2; t+=0.18){
    const r = wid * (1 - Math.abs(Math.sin(t))*0.35) + rand(-1.5,1.5);
    ctx.lineTo(Math.cos(t)*r, Math.sin(t)*len);
  }
  ctx.closePath(); ctx.fill();

  // прожилки
  ctx.strokeStyle = "rgba(255,255,255,0.65)";
  ctx.lineWidth = 1;
  for (let i=0;i<6;i++){
    ctx.beginPath();
    const y = -len*0.9 + i*(len*0.32);
    ctx.moveTo(0, y);
    ctx.lineTo(rand(-wid*0.6, -wid*0.25), y + rand(-4,4));
    ctx.moveTo(0, y);
    ctx.lineTo(rand( wid*0.25,  wid*0.6),  y + rand(-4,4));
    ctx.stroke();
  }
  ctx.restore();
}

function drawPlastic(){
  // цветной многоугольник + блик
  const n = rint(4,7);
  const cx = rand(26,70), cy = rand(26,70);
  const pts = [];
  for (let i=0;i<n;i++){
    pts.push([ clamp(cx+rand(-26,26),0,IMG.W), clamp(cy+rand(-26,26),0,IMG.H) ]);
  }
  ctx.fillStyle = `hsl(${rint(0,360)}, ${rint(60,90)}%, ${rint(45,70)}%)`;
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath(); ctx.fill();

  // блик
  const gx = clamp(cx + rand(-10,10), 0, IMG.W);
  const gy = clamp(cy + rand(-10,10), 0, IMG.H);
  const grad = ctx.createRadialGradient(gx, gy, 2, gx, gy, 28);
  grad.addColorStop(0, "rgba(255,255,255,0.75)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(gx, gy, 30, 0, Math.PI*2); ctx.fill();
}

function drawStone(){
  // неровный «овал» со зерном
  const cx = rand(30,66), cy = rand(30,66);
  const rx = rand(16,26), ry = rand(12,22);
  const rot = rand(0, Math.PI*2);

  ctx.save(); ctx.translate(cx,cy); ctx.rotate(rot);

  ctx.fillStyle = `hsl(${rint(20,50)}, ${rint(10,25)}%, ${rint(30,45)}%)`;
  ctx.beginPath();
  for (let a=0; a<Math.PI*2; a+=0.22){
    const rrx = rx + rand(-3,3), rry = ry + rand(-3,3);
    const x = Math.cos(a)*rrx, y = Math.sin(a)*rry;
    if (a===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.closePath(); ctx.fill();

  // зернистость
  for (let i=0;i<220;i++){
    ctx.fillStyle = `rgba(255,255,255,${rand(0.02,0.08)})`;
    ctx.fillRect(rand(-rx,rx), rand(-ry,ry), 1, 1);
  }
  ctx.restore();
}

/* ==================== конвертация в тензор ==================== */
function toTensorRGB() {
  const img = ctx.getImageData(0,0,IMG.W,IMG.H).data;
  const arr = new Float32Array(IMG.W * IMG.H * 3);
  for (let i=0,j=0; i<img.length; i+=4){
    arr[j++] = img[i]   / 255; // R
    arr[j++] = img[i+1] / 255; // G
    arr[j++] = img[i+2] / 255; // B
  }
  return tf.tensor4d(arr, [1, IMG.W, IMG.H, 3]);
}
function oneHot(idx,n){ const a=new Array(n).fill(0); a[idx]=1; return a; }

/* ==================== публичные API ==================== */
export function generateOne(clsIdx, {brightness=0, shadow=20, noise=8} = {}){
  // не оборачиваем весь блок в tidy, чтобы вернуть тензоры наружу
  reset();
  if (clsIdx===0)      drawLeaf();
  else if (clsIdx===1) drawPlastic();
  else                 drawStone();

  addBrightness(brightness);
  addShadow(shadow);
  addNoise(noise);

  const x = toTensorRGB();
  const y = tf.tensor2d([oneHot(clsIdx, CLASSES.length)], [1, CLASSES.length]);
  return { x, y, preview: off.toDataURL() };
}

export function generateDataset(perClass=300, opts={brightness:0, shadow:20, noise:8}, testPct=0.2){
  // В tidy создаём временные тензоры; итоговые train/test возвращаем наружу
  return tf.tidy(() => {
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
    const split = Math.floor(N * (1 - testPct));

    const Xsh = tf.gather(X, idx);
    const Ysh = tf.gather(Y, idx);

    const Xtrain = Xsh.slice([0,0,0,0], [split, IMG.W, IMG.H, 3]);
    const Ytrain = Ysh.slice([0,0],     [split, CLASSES.length]);
    const Xtest  = Xsh.slice([split,0,0,0], [N - split, IMG.W, IMG.H, 3]);
    const Ytest  = Ysh.slice([split,0],     [N - split, CLASSES.length]);

    // X/Y/Xsh/Ysh будут удалены tidy; Xtrain/Ytrain/Xtest/Ytest вернутся живыми
    return { Xtrain, Ytrain, Xtest, Ytest, previews };
  });
}
