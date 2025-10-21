// infer.js
let MODEL = null;
let LABELS = null;

const ui = {
  btnLoad: document.getElementById('btnLoad'),
  modelInfo: document.getElementById('modelInfo'),
  file: document.getElementById('file'),
  img: document.getElementById('img'),
  btnPredict: document.getElementById('btnPredict'),
  probs: document.getElementById('probs'),
  log: document.getElementById('log'),
};

function logln(s){ ui.log.textContent += s + "\n"; ui.log.scrollTop = ui.log.scrollHeight; }

ui.btnLoad.onclick = async () => {
  try{
    ui.log.textContent = "";
    // безопасная инициализация TF (WebGL → CPU fallback)
    try {
      await tf.setBackend('webgl'); 
      await tf.ready();
    } catch {
      await tf.setBackend('cpu');
      await tf.ready();
    }
    logln("TF backend: " + tf.getBackend());

    // грузим модель и labels
    MODEL = await tf.loadGraphModel('./web_model/model.json');
    const res = await fetch('./web_model/labels.json');
    LABELS = await res.json();

    // выводим инфо
    const sig = MODEL.executor.graph.signature?.outputs || {};
    ui.modelInfo.innerHTML = `Модель загружена. Классов: ${LABELS.length}`;
    ui.btnPredict.disabled = false;
  } catch(e){
    logln("❌ Ошибка загрузки модели: " + (e.message||e));
  }
};

ui.file.onchange = () => {
  const f = ui.file.files?.[0]; if(!f) return;
  const url = URL.createObjectURL(f);
  ui.img.src = url;
};

ui.btnPredict.onclick = async () => {
  if (!MODEL || !ui.img.src) return;
  try{
    // препроцесс: [H,W,3] RGB → [1,224,224,3], нормализация 0..1
    const W=224, H=224;
    const can = document.createElement('canvas');
    can.width = W; can.height = H;
    const g = can.getContext('2d');
    g.drawImage(ui.img, 0, 0, W, H);
    const data = g.getImageData(0,0,W,H).data;
    const arr = new Float32Array(W*H*3);
    for (let i=0,j=0;i<data.length;i+=4){
      arr[j++] = data[i]   / 255;
      arr[j++] = data[i+1] / 255;
      arr[j++] = data[i+2] / 255;
    }
    const x = tf.tensor4d(arr, [1,W,H,3]);

    // predict
    const out = MODEL.execute(x);
    const probs = await (Array.isArray(out) ? out[0] : out).data();
    x.dispose(); if (Array.isArray(out)) out.forEach(t=>t.dispose()); else out.dispose();

    // визуализация
    const pairs = LABELS.map((name,i)=>({name, p: probs[i] || 0}));
    pairs.sort((a,b)=> b.p - a.p);
    ui.probs.innerHTML = pairs.map(r => `<tr><td>${r.name}</td><td>${(r.p*100).toFixed(2)}%</td></tr>`).join("");
  } catch(e){
    logln("❌ Ошибка инференса: " + (e.message||e));
  }
};
