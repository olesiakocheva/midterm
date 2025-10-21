/* app.js — безопасное обучение на CPU и быстрый инференс на WebGL.
   Можно вызывать напрямую из твоих обработчиков кнопок.
   Ничего больше не нужно менять по месту — просто импортируй/используй. */

// ---------- Backend helpers ----------
export async function useCpuForTraining() {
  try {
    tf.env().set('WEBGL_PACK', false);
    tf.env().set('WEBGL_VERSION', 1);
  } catch {}
  await tf.setBackend('cpu');
  await tf.ready();
  console.log('TF backend (train):', tf.getBackend());
}

export async function useWebglForInference() {
  try {
    tf.env().set('WEBGL_VERSION', 1);
    tf.env().set('WEBGL_PACK', false);
    tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
    await tf.setBackend('webgl');
    await tf.ready();
  } catch {
    await tf.setBackend('cpu');
    await tf.ready();
  }
  console.log('TF backend (infer):', tf.getBackend());
}

// ---------- Training / Eval wrappers ----------
/** Обучение с авто-подстраховкой (CPU), маленьким batch, проверкой NaN. */
export async function trainSafe(model, Xtrain, Ytrain, {
  epochs = 20,
  batchSize = 32,
  validationSplit = 0.1,
  shuffle = true,
  onEpochEnd = null
} = {}) {
  await useCpuForTraining();

  // аккуратный оптимайзер
  const lr = 1e-3;
  const opt = tf.train.adam(lr);

  // если compile уже сделан — не трогаем; иначе попробуем угадать задачу
  if (!model.optimizer) {
    // эвристика: классификация, если размер Ytrain[1] > 1
    const isClf = (Ytrain.shape[1] ?? 1) > 1;
    model.compile({
      optimizer: opt,
      loss: isClf ? 'categoricalCrossentropy' : 'meanSquaredError',
      metrics: [isClf ? 'accuracy' : 'mae']
    });
  }

  const history = await model.fit(Xtrain, Ytrain, {
    epochs,
    batchSize: Math.max(8, Math.min(32, batchSize)), // ≤ 32 — меньше шансов упасть
    validationSplit,
    shuffle,
    callbacks: {
      onEpochEnd: async (ep, logs) => {
        if (!Number.isFinite(logs.loss)) {
          throw new Error('Loss стал NaN — уменьшите learning rate/batch и проверьте препроцесс.');
        }
        onEpochEnd && onEpochEnd(ep, logs);
        // даем браузеру отрисоваться и WebGL/GC подышать
        await tf.nextFrame();
      }
    }
  });

  // вернёмся на WebGL для инференса
  await useWebglForInference();
  return history;
}

/** Оценка модели (переключает бэкенд на WebGL для скорости). */
export async function evaluateSafe(model, Xtest, Ytest) {
  await useWebglForInference();
  const out = model.evaluate(Xtest, Ytest, { verbose: 0 });
  const vals = Array.isArray(out) ? await Promise.all(out.map(t => t.data())) : [await out.data()];
  return vals.map(v => v[0]);
}

/** Предсказание (переключает бэкенд на WebGL). */
export async function predictSafe(model, input) {
  await useWebglForInference();
  const y = model.predict(input);
  const data = Array.isArray(y) ? await Promise.all(y.map(t => t.array())) : await y.array();
  tf.dispose(y);
  return data;
}

// ---------- (Необязательная) авто-привязка к кнопкам, если они есть ----------
// Если у тебя есть такие id — получишь «из коробки». Если нет — просто игнорируется.
(function attachOptionalUI() {
  const log = (msg) => {
    const el = document.getElementById('log');
    if (el) { el.textContent += msg + '\n'; el.scrollTop = el.scrollHeight; }
    console.log(msg);
  };

  const btnTrain = document.getElementById('btnTrain');
  const btnEval  = document.getElementById('btnEval');

  // Ожидаем, что где-то глобально доступны model, Xtrain, Ytrain, Xtest, Ytest.
  if (btnTrain) {
    btnTrain.addEventListener('click', async () => {
      try {
        log('Начинаю обучение (CPU)…');
        const epochs = +document.getElementById('epochs')?.value || 20;
        const batch  = +document.getElementById('batch')?.value  || 32;
        await trainSafe(window.model, window.Xtrain, window.Ytrain, {
          epochs, batchSize: batch,
          onEpochEnd: (ep, logs) => log(`epoch ${ep+1}: loss=${logs.loss.toFixed(4)} val_loss=${(logs.val_loss??0).toFixed(4)}`)
        });
        log('Обучение завершено. Переключилась на WebGL для инференса.');
      } catch (e) {
        log('❌ Ошибка обучения: ' + (e.message || e));
      }
    });
  }

  if (btnEval) {
    btnEval.addEventListener('click', async () => {
      try {
        log('Оцениваю (WebGL)…');
        const vals = await evaluateSafe(window.model, window.Xtest, window.Ytest);
        log('Eval: ' + vals.map((v,i)=> `m${i}=${v.toFixed(4)}`).join('  '));
      } catch (e) {
        log('❌ Ошибка оценки: ' + (e.message || e));
      }
    });
  }
})();
