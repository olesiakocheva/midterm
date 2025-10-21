// --- Safe TF backend helpers ---
// вызывать ПЕРЕД обучением
export async function useCpuForTraining() {
  try {
    // иногда помогает полностью выключить упаковку шейдеров
    tf.env().set('WEBGL_PACK', false);
    tf.env().set('WEBGL_VERSION', 1);
  } catch {}
  await tf.setBackend('cpu');
  await tf.ready();
  console.log('TF backend (train):', tf.getBackend());
}

// вызывать ПЕРЕД инференсом/оценкой
export async function useWebglForInference() {
  try {
    tf.env().set('WEBGL_VERSION', 1);
    tf.env().set('WEBGL_PACK', false); // оставим false — совместимее
    await tf.setBackend('webgl');
    await tf.ready();
  } catch {
    await tf.setBackend('cpu');
    await tf.ready();
  }
  console.log('TF backend (infer):', tf.getBackend());
}

// app.js
import { CLASSES, IMG, generateDataset } from './generator.js';
import {
  buildSmallCNN, trainSmall, evaluateSmall, predictProbsSmall,
  buildTransferModel, trainTransfer, evaluateTransfer, predictProbsTransfer
} from './model.js';
// --- SAFE TF INIT (добавить в app.js один раз) ---
async function initTFSafe(logln){
  try {
    // Безопасные флаги до ready(): меньше «сложных» шейдеров
    tf.env().set('WEBGL_VERSION', 1);             // форсим WebGL1 (часто надёжнее)
    tf.env().set('WEBGL_PACK', false);            // отключаем packed-шейдеры
    tf.env().set('WEBGL_FORCE_F16_TEXTURES', true); // half-float текстуры

    await tf.setBackend('webgl');
  } catch(_) { /* игнор */ }

  try {
    await tf.ready();
  } catch(e) {
    // если даже ready спотыкается — уходим на CPU
    await tf.setBackend('cpu');
    await tf.ready();
  }
  logln?.(`TF backend: ${tf.getBackend()}`);
}

const ui = {
  perClass: document.getElementById('perClass'),
  brightness: document.getElementById('brightness'),
  shadow: document.getElementById('shadow'),
  noise: document.getElementById('noise'),
  bVal: document.getElementById('bVal'),
  sVal: document.getElementById('sVal'),
  nVal: document.getElementById('nVal'),
  btnGen: document.getElementById('btnGen'),

  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  btnBuild: document.getElementById('btnBuild'),
  btnTrain: document.getElementById('btnTrain'),
  btnEval: document.getElementById('btnEval'),

  trainChart: document.getElementById('trainChart'),
  cmChart: document.getElementById('cmChart'),
  preview: document.getElementById('preview'),

  file: document.getElementById('fileInput'),
  uploadPreview: document.getElementById('uploadPreview'),
  btnPredict: document.getElementById('btnPredict'),
  probs: document.getElementById('probs'),

  log: document.getElementById('log'),
};

let DATA=null; // {Xtrain,Ytrain,Xtest,Ytest, previews}
let modelType = 'cnn'; // 'cnn' | 'tl'
let small=null;        // small CNN
let tl=null;           // {head, featuresBatch}
let trainChart=null, cmChart=null;

function logln(s){ ui.log.textContent += s + "\n"; ui.log.scrollTop = ui.log.scrollHeight; }
function disableAll(b){
  ui.btnGen.disabled=b; ui.btnBuild.disabled=b;
  ui.btnTrain.disabled=b || !(DATA && (small||tl));
  ui.btnEval.disabled=b || !(DATA && (small||tl));
  ui.btnPredict.disabled=b || !(small||tl);
}

// reflect slider values
ui.bVal.textContent = ui.brightness.value;
ui.sVal.textContent = ui.shadow.value;
ui.nVal.textContent = ui.noise.value;
ui.brightness.oninput=()=> ui.bVal.textContent = ui.brightness.value;
ui.shadow.oninput=()=> ui.sVal.textContent = ui.shadow.value;
ui.noise.oninput=()=> ui.nVal.textContent = ui.noise.value;

// choose model type
[...document.querySelectorAll('input[name="mdl"]')].forEach(r=>{
  r.addEventListener('change', ()=> { modelType = r.value; logln("Модель: " + (modelType==='cnn'?'Small CNN':'Transfer (MobileNet)')); });
});

// dataset generation
ui.btnGen.onclick = async ()=>{
  try{
    disableAll(true);
    ui.log.textContent="";
    await tf.setBackend('webgl').catch(()=>{});
    await tf.ready();
    logln(`TF backend: ${tf.getBackend()}`);

    const perClass = Math.max(30, Math.min(+ui.perClass.value||300, 1500));
    const opts = {
      brightness: +ui.brightness.value||0,
      shadow: +ui.shadow.value||0,
      noise: +ui.noise.value||0
    };
    logln(`Генерация: perClass=${perClass}, brightness=${opts.brightness}, shadow=${opts.shadow}, noise=${opts.noise}`);
    DATA = generateDataset(perClass, opts, 0.2);
    logln(`Датасет готов. Train=${DATA.Xtrain.shape[0]} Test=${DATA.Xtest.shape[0]}`);

    renderRandomPreview();
    ui.btnTrain.disabled = !(small||tl);
    ui.btnEval.disabled  = true;
  } catch(e){
    logln("❌ Ошибка генерации: " + (e.message||e));
  } finally { disableAll(false); }
};

// build model
ui.btnBuild.onclick = async ()=>{
  try{
    disableAll(true);
    if (small){ small.dispose?.(); small=null; }
    tl=null;
    if (modelType==='cnn'){
      small = buildSmallCNN();
      logln("Модель собрана: Small CNN (96×96×3 → 16-32-64 → Dense)");
    } else {
      logln("Загрузка MobileNet…");
      tl = await buildTransferModel();
      logln("Transfer Learning: MobileNet-features → Dense-голова");
    }
    ui.btnTrain.disabled = !DATA;
    ui.btnEval.disabled  = true;
    ui.btnPredict.disabled = !(small||tl);
  } catch(e){
    logln("❌ Ошибка сборки модели: " + (e.message||e));
  } finally { disableAll(false); }
};

// train
ui.btnTrain.onclick = async ()=>{
  if (!DATA) return;
  disableAll(true);
  const epochs = +ui.epochs.value||12, batch=+ui.batch.value||64;
  initTrainChart();
  try{
    if (small){
      logln(`Обучение Small CNN: epochs=${epochs} batch=${batch}`);
      await trainSmall(small, DATA.Xtrain, DATA.Ytrain, epochs, batch, (ep,logs)=> addTrainPoint(ep+1, logs.loss, logs.val_loss, logs.acc??logs.accuracy, logs.val_acc??logs.val_accuracy));
    } else if (tl){
      logln(`Обучение Transfer (голова): epochs=${Math.max(6,Math.min(epochs,20))} batch=${Math.min(64,batch)}`);
      await trainTransfer(tl, DATA.Xtrain, DATA.Ytrain, Math.max(6,Math.min(epochs,20)), Math.min(64,batch), (ep,logs)=> addTrainPoint(ep+1, logs.loss, logs.val_loss, logs.acc??logs.accuracy, logs.val_acc??logs.val_accuracy));
    }
    logln("Обучение завершено.");
    ui.btnEval.disabled = false;
    ui.btnPredict.disabled = false;
  } catch(e){
    logln("❌ Ошибка обучения: " + (e.message||e));
  } finally { disableAll(false); }
};

// eval
ui.btnEval.onclick = async ()=>{
  if (!DATA) return;
  disableAll(true);
  try{
    let res;
    if (small) res = await evaluateSmall(small, DATA.Xtest, DATA.Ytest);
    else       res = await evaluateTransfer(tl, DATA.Xtest, DATA.Ytest);
    logln(`Test: loss=${res.loss.toFixed(4)} acc=${(res.acc*100).toFixed(1)}%`);
    const cm = await confusionMatrix(DATA);
    renderCM(cm);
  } catch(e){
    logln("❌ Ошибка оценки: " + (e.message||e));
  } finally { disableAll(false); }
};

// upload predict
ui.file.onchange = ()=>{
  const f = ui.file.files?.[0]; if(!f) return;
  const url = URL.createObjectURL(f);
  ui.uploadPreview.src = url;
};
ui.btnPredict.onclick = async ()=>{
  try{
    const img = await toInputTensor(ui.uploadPreview);
    const probs = small ? await predictProbsSmall(small, img)
                        : await predictProbsTransfer(tl, img);
    img.dispose();
    renderProbs(probs);
  } catch(e){
    logln("❌ Ошибка инференса: " + (e.message||e));
  }
};

/* ---------- helpers ---------- */
function renderRandomPreview(){
  const pv = DATA.previews[Math.floor(Math.random()*DATA.previews.length)];
  const img = new Image();
  img.onload = ()=>{
    const c = ui.preview.getContext('2d');
    c.clearRect(0,0,ui.preview.width, ui.preview.height);
    c.drawImage(img,0,0,ui.preview.width, ui.preview.height);
    c.fillStyle="#111"; c.fillText(CLASSES[pv.cls], 6, 14);
  };
  img.src = pv.data;
}

async function toInputTensor(htmlImg){
  // рисуем на скрытом canvas и нормализуем в [0,1]
  const can = document.createElement('canvas');
  can.width = IMG.W; can.height = IMG.H;
  const g = can.getContext('2d');
  g.drawImage(htmlImg, 0,0, IMG.W, IMG.H);
  const img = g.getImageData(0,0,IMG.W,IMG.H).data;
  const arr = new Float32Array(IMG.W*IMG.H*3);
  for (let i=0,j=0;i<img.length;i+=4){
    arr[j++] = img[i]/255; arr[j++] = img[i+1]/255; arr[j++] = img[i+2]/255;
  }
  return tf.tensor4d(arr, [1, IMG.W, IMG.H, 3]);
}

// training charts
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

async function confusionMatrix(DATA){
  const N = DATA.Xtest.shape[0];
  const bs = 128;
  const steps = Math.ceil(N/bs);
  const cm = Array.from({length:CLASSES.length}, ()=> Array(CLASSES.length).fill(0));

  for (let s=0;s<steps;s++){
    const start = s*bs, size=Math.min(bs, N-start);
    const x = DATA.Xtest.slice([start,0,0,0],[size, IMG.W, IMG.H, 3]);
    const y = await DATA.Ytest.slice([start,0],[size, CLASSES.length]).array();
    let p;
    if (small){
      const out = small.predict(x);
      p = await out.array(); out.dispose();
    } else {
      // TL: прогоним кусочками через predictProbsTransfer аналогично — здесь для простоты:
      const img = x; // features считаются внутри predict… (в нашей реализации — в model.js они считаются отдельно)
      const probs = [];
      const arr = await img.array();
      for (let i=0;i<size;i++){
        const t = tf.tensor4d(arr[i], [1, IMG.W, IMG.H, 3]);
        const pr = await predictProbsTransfer(tl, t);
        probs.push(pr);
        t.dispose();
      }
      p = probs;
      img.dispose();
    }
    for (let i=0;i<size;i++){
      const yi = argmax(y[i]), pi = argmax(p[i]);
      cm[yi][pi] += 1;
    }
    x.dispose();
  }
  return cm;
}
function argmax(a){ let mi=0,mv=-Infinity; for(let i=0;i<a.length;i++){ if(a[i]>mv){mv=a[i]; mi=i;} } return mi; }

let cmChartRef=null;
function renderCM(cm){
  if (cmChartRef) cmChartRef.destroy();
  cmChartRef = new Chart(ui.cmChart.getContext('2d'), {
    type: 'bar',
    data: {
      labels: CLASSES.map((c,i)=> `${i}:${c}`),
      datasets: cm.map((row,ri)=>({ label:`True ${CLASSES[ri]}`, data: row }))
    },
    options:{ responsive:true, scales:{ x:{stacked:true}, y:{stacked:true, beginAtZero:true}}}
  });
}

// enable predict only when image chosen
ui.uploadPreview.onload = ()=> { ui.btnPredict.disabled = !(small||tl); };
