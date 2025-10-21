// generator.js
// Генерация синтетических изображений мазков 64x64 (grayscale) для 4 классов

export const CLASSES = ["Impressionism", "Cubism", "Abstract", "Minimalism"];
export const IMG_SIZE = 64;

const off = document.createElement('canvas');
off.width = IMG_SIZE; off.height = IMG_SIZE;
const ctx = off.getContext('2d');

function resetCanvas(bg="#ffffff") {
  ctx.clearRect(0,0,IMG_SIZE,IMG_SIZE);
  ctx.fillStyle = bg;
  ctx.fillRect(0,0,IMG_SIZE,IMG_SIZE);
}

function rand(a,b){ return Math.random()*(b-a)+a; }
function rint(a,b){ return Math.floor(rand(a,b+1)); }
function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }

function addNoise(amount=8){
  // белый фон уже есть; рисуем серые точки шума
  const n = amount*50; // масштаб
  for (let i=0;i<n;i++){
    const x = rint(0,IMG_SIZE-1), y=rint(0,IMG_SIZE-1);
    const c = rint(200,255);
    ctx.fillStyle = `rgb(${c},${c},${c})`;
    ctx.fillRect(x,y,1,1);
  }
}

function strokeLine(x1,y1,x2,y2,th=2,color="#000"){
  ctx.strokeStyle = color;
  ctx.lineWidth = th;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x1,y1);
  ctx.lineTo(x2,y2);
  ctx.stroke();
}

function strokePoly(points, fill=false, th=2, color="#000"){
  ctx.strokeStyle = color;
  ctx.lineWidth = th;
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i=1;i<points.length;i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.closePath();
  if (fill){
    ctx.fillStyle = color; ctx.globalAlpha = 0.08;
    ctx.fill(); ctx.globalAlpha = 1.0;
  }
  ctx.stroke();
}

function strokeCurve(cx,cy,rad,thetaLen,th=2){
  ctx.strokeStyle = "#000";
  ctx.lineWidth = th;
  ctx.beginPath();
  for(let t=0;t<thetaLen;t+=0.15){
    const x = cx + Math.cos(t)*rad + rand(-2,2);
    const y = cy + Math.sin(t)*rad + rand(-2,2);
    if(t===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
}

function drawImpressionism(thickness){
  // много коротких штрихов по сцене
  const n = rint(12,20);
  for(let i=0;i<n;i++){
    const x = rand(6, IMG_SIZE-6), y = rand(6, IMG_SIZE-6);
    const len = rand(6, 16);
    const ang = rand(0, Math.PI*2);
    const x2 = clamp(x + Math.cos(ang)*len, 0, IMG_SIZE);
    const y2 = clamp(y + Math.sin(ang)*len, 0, IMG_SIZE);
    strokeLine(x,y,x2,y2, thickness, "#000");
  }
}

function drawCubism(thickness){
  // 2-3 угловатых многоугольника
  const k = rint(2,3);
  for(let i=0;i<k;i++){
    const cx = rand(16, IMG_SIZE-16), cy = rand(16, IMG_SIZE-16);
    const sides = rint(4,7);
    const rad = rand(8, 18);
    const pts = [];
    const base = rand(0, Math.PI*2);
    for(let s=0;s<sides;s++){
      const ang = base + s*(Math.PI*2/sides) + rand(-0.2,0.2);
      const rr = rad + rand(-4,4);
      pts.push([ clamp(cx+Math.cos(ang)*rr,0,IMG_SIZE), clamp(cy+Math.sin(ang)*rr,0,IMG_SIZE) ]);
    }
    strokePoly(pts, true, thickness, "#000");
  }
}

function drawAbstract(thickness){
  // дуги/кривые + брызги
  const arcs = rint(2,4);
  for(let i=0;i<arcs;i++){
    const cx = rand(16, IMG_SIZE-16), cy = rand(16, IMG_SIZE-16);
    const rad = rand(10, 22);
    const theta = rand(Math.PI*0.8, Math.PI*1.8);
    strokeCurve(cx, cy, rad, theta, thickness);
  }
  // "splatter"
  const s = rint(25,45);
  for(let i=0;i<s;i++){
    const x = rint(0,IMG_SIZE-1), y = rint(0,IMG_SIZE-1);
    ctx.fillStyle = (Math.random()<0.6)?"#000":"#111";
    ctx.fillRect(x,y,1,1);
  }
}

function drawMinimalism(thickness){
  // 1-2 тонкие линии по диагонали/горизонту
  const n = rint(1,2);
  for(let i=0;i<n;i++){
    const x1 = rand(6, IMG_SIZE-6), y1 = rand(6, IMG_SIZE-6);
    const x2 = rand(6, IMG_SIZE-6), y2 = rand(6, IMG_SIZE-6);
    strokeLine(x1,y1,x2,y2, Math.max(1, thickness-1), "#000");
  }
}

function toTensorGray() {
  // берём один канал (grayscale), нормализация в [0,1]
  const img = ctx.getImageData(0,0,IMG_SIZE,IMG_SIZE).data;
  const arr = new Float32Array(IMG_SIZE*IMG_SIZE);
  for (let i=0, j=0; i<img.length; i+=4, j++){
    // инвертируем фон (чёрное = 1), чтобы линии были яркостью 1
    const r=img[i], g=img[i+1], b=img[i+2];
    const gray = (r+g+b)/3; // 0..255
    arr[j] = (255-gray)/255;
  }
  return tf.tensor4d(arr, [1, IMG_SIZE, IMG_SIZE, 1]);
}

export function generateOne(labelIndex, noise=8, thickness=3){
  resetCanvas("#ffffff");
  switch(labelIndex){
    case 0: drawImpressionism(thickness); break;
    case 1: drawCubism(thickness); break;
    case 2: drawAbstract(thickness); break;
    case 3: drawMinimalism(thickness); break;
  }
  if (noise>0) addNoise(noise);
  const x = toTensorGray();
  const y = tf.tensor2d([oneHot(labelIndex, CLASSES.length)], [1, CLASSES.length]);
  return {x, y, preview: off.toDataURL()};
}

function oneHot(idx, n){
  const a = new Array(n).fill(0); a[idx]=1; return a;
}

export async function generateDataset(perClass=300, noise=8, thickness=3, testPct=0.2){
  const xs = [], ys = [], previews = [];
  for (let c=0;c<CLASSES.length;c++){
    for (let i=0;i<perClass;i++){
      const {x, y, preview} = generateOne(c, noise, thickness);
      xs.push(x); ys.push(y); previews.push({cls:c, data:preview});
    }
  }
  const X = tf.concat(xs, 0);
  const Y = tf.concat(ys, 0);
  xs.forEach(t => t.dispose()); ys.forEach(t => t.dispose());

  // Перемешиваем и делим
  const N = X.shape[0];
  const idx = tf.util.createShuffledIndices(N);
  const split = Math.floor(N*(1-testPct));
  const Xsh = tf.gather(X, idx), Ysh=tf.gather(Y, idx);
  const Xtrain = Xsh.slice([0,0,0,0],[split,IMG_SIZE,IMG_SIZE,1]);
  const Ytrain = Ysh.slice([0,0],[split,CLASSES.length]);
  const Xtest  = Xsh.slice([split,0,0,0],[N-split,IMG_SIZE,IMG_SIZE,1]);
  const Ytest  = Ysh.slice([split,0],[N-split,CLASSES.length]);

  X.dispose(); Y.dispose(); Xsh.dispose(); Ysh.dispose();

  return {Xtrain, Ytrain, Xtest, Ytest, previews};
}
