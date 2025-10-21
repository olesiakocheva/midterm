// app.js
import { CLASSES, IMG_SIZE, generateDataset, generateOne } from './generator.js';
import { buildModel, trainModel, evaluate, predictProbs } from './model.js';

const ui = {
  perClass: document.getElementById('perClass'),
  noise: document.getElementById('noise'),
  noiseVal: document.getElementById('noiseVal'),
  thickness: document.getElementById('thickness'),
  thickVal: document.getElementById('thickVal'),
  btnGen: document.getElementById('btnGen'),
  btnBuild: document.getElementById('btnBuild'),
  btnTrain: document.getElementById('btnTrain'),
  btnEval: document.getElementById('btnEval'),
  preview: document.getElementById('preview'),
  draw: document.getElementById('draw'),
  clearDraw: document.getElementById('clearDraw'),
  predictDraw: document.getElementById('predictDraw'),
  probs: document.getElementById('probs'),
  log: document.getElementById('log'),
  trainChart: document.getElementById('trainChart'),
  cmChart: document.getElementById('cmChart'),
  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  status: null
};

let DATA = null; // {Xtrain,Ytrain,Xtest,Ytest, previews}
let MODEL = null;
let trainChart, cmChart;
let isDrawing=false, lastXY=null;

function logln(s){ ui.log.textContent += s + "\n"; ui.log.scrollTop = ui.log.scrollHeight; }

ui.noise.addEventListener('input', ()=> ui.noiseVal.textContent = ui.noise.value);
ui.thickness.addEventListener('input', ()=> ui.thickVal.textContent = ui.thickness.value);

ui.btnGen.onclick = async ()=>{
  disableAll(true);
  ui.log.textContent = "";
  const n = +ui.perClass.value, noise=+ui.noise.value, th=+ui.thickness.value;
  logln(`Генерация: perClass=${n}, noise=${noise}, thickness=${th}`);
  DATA = await generateDataset(n, noise, th, 0.2);
  logln(`Готово. Train: ${DATA.Xtrain.shape[0]}  Test: ${DATA.Xtest.shape[0]}`);
  renderRandomPreview();
  ui.btnTrain.disabled = !MODEL;
  ui.btnEval.disabled = true;
  ui.predictDraw.disabled = !MODEL;
  disableAll(false);
};

ui.btnBuild.onclick = ()=>{
  if (MODEL) MODEL.dispose();
  MODEL = buildModel();
  logln("Модель построена: CNN 64x64x1 → 16-32-64 → Dense64 → 4 (softmax)");
  ui.btnTrain.disabled = !DATA;
  ui.btnEval.disabled = true;
  ui.predictDraw.disabled = !DATA;
};

ui.btnTrain.onclick = async ()=>{
  if (!MODEL || !DATA) return;
  disableAll(true);
  const epochs = +ui.epochs.value, batch=+ui.batch.value;
  logln(`Обучение: epochs=${epochs}, batch=${batch}`);
  initTrainChart();
  await trainModel(MODEL, DATA.Xtrain, DATA.Ytrain, epochs, batch, (ep, logs)=>{
    addTrainPoint(ep+1, logs.loss, logs.val_loss, logs.acc ?? logs.accuracy, logs.val_acc ?? logs.val_accuracy);
  });
  logln("Обучение завершено.");
  ui.btnEval.disabled = false;
  ui.predictDraw.disabled = false;
  disableAll(false);
};

ui.btnEval.onclick = async ()=>{
  if (!MODEL || !DATA) return;
  disableAll(true);
  const {loss, acc} = await evaluate(MODEL, DATA.Xtest, DATA.Ytest);
  logln(`Оценка на test: loss=${loss.toFixed(4)}  acc=${(acc*100).toFixed(1)}%`);
  // Confusion Matrix
  const cm = await confusionMatrix(MODEL, DATA.Xtest, DATA.Ytest);
  renderCM(cm);
  disableAll(false);
};

function renderRandomPreview(){
  const pv = DATA.previews[Math.floor(Math.random()*DATA.previews.length)];
  const img = new Image();
  img.onload = ()=>{
    const c = ui.preview.getContext('2d');
    c.clearRect(0,0,ui.preview.width, ui.preview.height);
    c.drawImage(img,0,0,ui.preview.width, ui.preview.height);
    c.fillStyle="#111";
    c.fillText(CLASSES[pv.cls], 6, 12);
  };
  img.src = pv.data;
}

// ============ Drawing pad ============
(function setupDraw(){
  const c = ui.draw, g = c.getContext('2d');
  resetDraw();
  c.addEventListener('mousedown', e=>{isDrawing=true; lastXY=getXY(c,e);});
  c.addEventListener('mousemove', e=>{
    if(!isDrawing) return;
    const xy = getXY(c,e);
    g.strokeStyle = "#000"; g.lineCap="round"; g.lineWidth = 6;
    g.beginPath(); g.moveTo(lastXY.x, lastXY.y); g.lineTo(xy.x, xy.y); g.stroke();
    lastXY = xy;
  });
  window.addEventListener('mouseup', ()=> isDrawing=false);
  ui.clearDraw.onclick = resetDraw;
  ui.predictDraw.onclick = predictFromDraw;
})();

function resetDraw(){
  const g = ui.draw.getContext('2d');
  g.fillStyle="#fff"; g.fillRect(0,0,ui.draw.width, ui.draw.height);
}

function getXY(canvas, ev){
  const r = canvas.getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
}

async function predictFromDraw(){
  if (!MODEL) return;
  // превратить 128x128 в 64x64 grayscale inverted
  const src = ui.draw.getContext('2d').getImageData(0,0,ui.draw.width, ui.draw.height).data;
  const tmp = document.createElement('canvas');
  tmp.width = IMG_SIZE; tmp.height = IMG_SIZE;
  const tctx = tmp.getContext('2d');
  // scale down
  tctx.drawImage(ui.draw, 0,0,IMG_SIZE,IMG_SIZE);
  const img = tctx.getImageData(0,0,IMG_SIZE,IMG_SIZE).data;
  const arr = new Float32Array(IMG_SIZE*IMG_SIZE);
  for (let i=0,j=0;i<img.length;i+=4,j++){
    const gray = (img[i]+img[i+1]+img[i+2])/3;
    arr[j] = (255-gray)/255;
  }
  const x = tf.tensor4d(arr, [1, IMG_SIZE, IMG_SIZE, 1]);
  const probs = await predictProbs(MODEL, x);
  x.dispose();
  renderProbs(probs);
}

// ============ Charts ============
function initTrainChart(){
  if (trainChart) trainChart.destroy();
  trainChart = new Chart(ui.trainChart.getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [
      {label:'loss', data:[], tension:.2},
      {label:'val_loss', data:[], tension:.2},
      {label:'acc', data:[], tension:.2},
      {label:'val_acc', data:[], tension:.2},
    ]},
    options: { responsive:true, scales:{ y:{ beginAtZero:true }}, plugins:{legend:{position:'bottom'}} }
  });
}
function addTrainPoint(epoch, loss, vloss, acc, vacc){
  trainChart.data.labels.push(String(epoch));
  trainChart.data.datasets[0].data.push(loss ?? null);
  trainChart.data.datasets[1].data.push(vloss ?? null);
  trainChart.data.datasets[2].data.push(acc ?? null);
  trainChart.data.datasets[3].data.push(vacc ?? null);
  trainChart.update();
}

function renderProbs(probs){
  const rows = CLASSES.map((c,i)=> `<tr><td>${c}</td><td>${(probs[i]*100).toFixed(1)}%</td></tr>`).join("");
  ui.probs.innerHTML = rows;
}

// Confusion Matrix
async function confusionMatrix(model, Xtest, Ytest){
  const preds = model.predict(Xtest);
  const p = await preds.array();
  preds.dispose();
  const y = await Ytest.array();
  const n = CLASSES.length;
  const cm = Array.from({length:n}, ()=> Array(n).fill(0));
  for (let i=0;i<p.length;i++){
    const yi = argmax(y[i]), pi = argmax(p[i]);
    cm[yi][pi] += 1;
  }
  return cm;
}
function argmax(arr){ let mi=0,mv=-Infinity; for(let i=0;i<arr.length;i++){ if(arr[i]>mv){mv=arr[i]; mi=i;} } return mi; }

function renderCM(cm){
  if (cmChart) cmChart.destroy();
  cmChart = new Chart(ui.cmChart.getContext('2d'), {
    type: 'bar',
    data: {
      labels: CLASSES.map((c,i)=> `${i}:${c}`),
      datasets: cm.map((row,ri)=>({
        label:`True ${CLASSES[ri]}`,
        data: row,
      }))
    },
    options:{ responsive:true, scales:{ x:{stacked:true}, y:{stacked:true, beginAtZero:true}}}
  });
}

function disableAll(b){
  ui.btnGen.disabled = b;
  ui.btnBuild.disabled = b;
  ui.btnTrain.disabled = b || !DATA;
  ui.btnEval.disabled = b || !MODEL;
  ui.predictDraw.disabled = b || !MODEL;
}

// авто-превью при старте (пустой)
resetDraw();
