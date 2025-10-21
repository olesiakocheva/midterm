// model.js
import { CLASSES, IMG } from './generator.js';

export function buildSmallCNN(){
  const m = tf.sequential();
  m.add(tf.layers.conv2d({inputShape:[IMG.W, IMG.H, 3], filters:16, kernelSize:3, activation:'relu', padding:'same'}));
  m.add(tf.layers.maxPool2d({poolSize:2}));
  m.add(tf.layers.conv2d({filters:32, kernelSize:3, activation:'relu', padding:'same'}));
  m.add(tf.layers.maxPool2d({poolSize:2}));
  m.add(tf.layers.conv2d({filters:64, kernelSize:3, activation:'relu', padding:'same'}));
  m.add(tf.layers.maxPool2d({poolSize:2}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dropout({rate:0.25}));
  m.add(tf.layers.dense({units:96, activation:'relu'}));
  m.add(tf.layers.dense({units:CLASSES.length, activation:'softmax'}));
  m.compile({optimizer: tf.train.adam(), loss:'categoricalCrossentropy', metrics:['accuracy']});
  return m;
}

// Transfer Learning на MobileNet (заморозим фичи, обучим «голову»)
export async function buildTransferModel(){
  // загружаем mobilenet (по умолчанию 224x224); подадим ресайз 96→224 внутри predict
  const mn = await mobilenet.load({version:2, alpha:0.5}); // лёгкая
  // берём внутреннюю модель фичей
  const featureExtractor = mn; // у mobilenet есть метод .infer для фичей

  // создаём «голову»
  const input = tf.input({shape:[1024]}); // у v2 alpha=0.5 будет 1024 фичей
  let x = tf.layers.dropout({rate:0.2}).apply(input);
  x = tf.layers.dense({units:256, activation:'relu'}).apply(x);
  const out = tf.layers.dense({units:CLASSES.length, activation:'softmax'}).apply(x);
  const head = tf.model({inputs:input, outputs:out});
  head.compile({optimizer: tf.train.adam(1e-3), loss:'categoricalCrossentropy', metrics:['accuracy']});

  // Обёртка для инференса: превращаем 96→224 и прогоняем через mn.infer
  function toMobileNetTensor(x96){
    // x96: [N,96,96,3] 0..1
    const resized = tf.image.resizeBilinear(x96, [224,224]);
    return resized;
  }
  async function featuresBatch(x96){
    const r = toMobileNetTensor(x96);
    const f = featureExtractor.infer(r, {embedding:true}); // [N,1024]
    r.dispose();
    return f;
  }

  return { head, featuresBatch };
}

export async function trainSmall(model, Xtrain, Ytrain, epochs=12, batch=64, onEpoch){
  return model.fit(Xtrain, Ytrain, {
    epochs, batchSize: batch, shuffle:true, validationSplit:0.1,
    callbacks:{ onEpochEnd: async(ep,logs)=> onEpoch?.(ep,logs) }
  });
}

// обучение головы в TL
export async function trainTransfer(tl, Xtrain, Ytrain, epochs=8, batch=64, onEpoch){
  const {head, featuresBatch} = tl;
  // прогоняем train фичи батчами (чтобы не умирать памятью)
  const N = Xtrain.shape[0];
  const steps = Math.ceil(N / batch);
  const feats = [];
  const labels = [];

  for (let s=0;s<steps;s++){
    const start = s*batch;
    const size  = Math.min(batch, N-start);
    const x = Xtrain.slice([start,0,0,0],[size, IMG.W, IMG.H, 3]);
    const y = Ytrain.slice([start,0],[size, CLASSES.length]);
    const f = await featuresBatch(x);
    feats.push(f); labels.push(y);
    x.dispose();
  }
  const F = tf.concat(feats,0);
  const Y = tf.concat(labels,0);
  feats.forEach(t=>t.dispose());

  const h = await head.fit(F, Y, {
    epochs, batchSize: Math.min(batch,64), shuffle:true, validationSplit:0.1,
    callbacks:{ onEpochEnd: async(ep,logs)=> onEpoch?.(ep,logs) }
  });

  F.dispose(); Y.dispose();
  return h;
}

export async function evaluateSmall(model, Xtest, Ytest){
  const out = model.evaluate(Xtest, Ytest, {batchSize:128, verbose:0});
  const [lossT, accT] = await Promise.all(out.map(t=>t.data()));
  return {loss: lossT[0], acc: accT[0]};
}

export async function evaluateTransfer(tl, Xtest, Ytest){
  const {head, featuresBatch} = tl;
  const N = Xtest.shape[0];
  const batch = 128;
  const steps = Math.ceil(N / batch);
  let correct=0, total=0, lossSum=0;

  for (let s=0;s<steps;s++){
    const start = s*batch;
    const size  = Math.min(batch, N-start);
    const x = Xtest.slice([start,0,0,0],[size, IMG.W, IMG.H, 3]);
    const y = Ytest.slice([start,0],[size, CLASSES.length]);
    const f = await featuresBatch(x);
    const evalOut = head.evaluate(f, y, {batchSize:size, verbose:0});
    const [lossT, accT] = await Promise.all(evalOut.map(t=>t.data()));
    lossSum += lossT[0]*size;
    correct += Math.round(accT[0]*size);
    total   += size;
    x.dispose(); y.dispose(); f.dispose();
  }
  return {loss: lossSum/total, acc: correct/total};
}

export async function predictProbsSmall(model, x1){
  const p = model.predict(x1);
  const probs = await p.data();
  p.dispose();
  return Array.from(probs);
}

export async function predictProbsTransfer(tl, x1){
  const {head, featuresBatch} = tl;
  const f = await featuresBatch(x1);
  const p = head.predict(f);
  const probs = await p.data();
  f.dispose(); p.dispose();
  return Array.from(probs);
}
