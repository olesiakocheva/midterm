// model.js
// Небольшая CNN для 64x64x1

import { IMG_SIZE, CLASSES } from './generator.js';

export function buildModel(){
  const m = tf.sequential();
  m.add(tf.layers.conv2d({
    inputShape: [IMG_SIZE, IMG_SIZE, 1],
    filters: 16, kernelSize: 3, activation: 'relu', padding: 'same'
  }));
  m.add(tf.layers.maxPool2d({poolSize: 2}));
  m.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', padding:'same'}));
  m.add(tf.layers.maxPool2d({poolSize: 2}));
  m.add(tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'relu', padding:'same'}));
  m.add(tf.layers.maxPool2d({poolSize: 2}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dropout({rate: 0.25}));
  m.add(tf.layers.dense({units: 64, activation: 'relu'}));
  m.add(tf.layers.dense({units: CLASSES.length, activation: 'softmax'}));

  m.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return m;
}

export async function trainModel(model, Xtrain, Ytrain, epochs=15, batch=64, onEpoch){
  const h = await model.fit(Xtrain, Ytrain, {
    epochs, batchSize: batch, shuffle: true, validationSplit: 0.1,
    callbacks: {
      onEpochEnd: async (ep, logs) => onEpoch?.(ep, logs)
    }
  });
  return h;
}

export async function evaluate(model, Xtest, Ytest){
  const out = model.evaluate(Xtest, Ytest, {batchSize: 256, verbose: 0});
  const [lossT, accT] = await Promise.all(out.map(t=>t.data()));
  return {loss: lossT[0], acc: accT[0]};
}

export async function predictProbs(model, x1){
  const p = model.predict(x1);
  const probs = await p.data();
  p.dispose();
  return Array.from(probs);
}
