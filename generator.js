/* generator.js — безопасные индексы для tf.gather/oneHot и базовые утилиты.
   Заменяет любые Uint32Array/Float32Array индексы на int32-тензоры.
   Экспортирует IMG_SIZE (часто требуется). */

export const IMG_SIZE = 224; // если где-то ожидают этот экспорт — он теперь есть

// ---------- Индексы и сборка ----------
/** Превращает JS-массив индексов в int32-тензор. */
export function int32Indices(indexArray) {
  // indexArray может быть TypedArray или обычным массивом
  const arr = Array.from(indexArray, v => v|0);
  return tf.tensor1d(arr, 'int32');
}

/** Безопасный gather: indices → int32. */
export function safeGather(x, indicesArray, axis = 0) {
  const idx = int32Indices(indicesArray);
  const y = tf.gather(x, idx, axis);
  idx.dispose();
  return y;
}

/** Сделать последовательные индексы [0..n-1] как int32-тензор. */
export function arangeInt32(n) {
  return tf.range(0, n|0, 1, 'int32');
}

/** Перемешка индексов (Fisher–Yates), возвращает обычный массив. */
export function shuffledIndices(n, seed = null) {
  const idx = Array.from({length: n|0}, (_,i)=> i);
  let r = seedRandom(seed);
  for (let i=idx.length-1; i>0; i--) {
    const j = Math.floor(r()* (i+1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}

// простой детерминированный генератор случайных чисел (если нужен seed)
function seedRandom(seed) {
  let s = (seed == null ? (Math.random()*1e9)|0 : (seed|0)) >>> 0;
  return function() {
    // xorshift32
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    return ((s >>> 0) / 4294967296);
  };
}

// ---------- Пример: подготовка батчей из тензора X (samples x features) ----------
/** Возвращает итератор батчей: по shuffled индексам и безопасному gather. */
export function* makeBatches(X, Y = null, batchSize = 32, seed = null) {
  const n = X.shape[0]|0;
  const order = shuffledIndices(n, seed);
  for (let i=0; i<n; i += batchSize) {
    const slice = order.slice(i, i+batchSize);
    const xBatch = safeGather(X, slice, 0);
    const yBatch = Y ? safeGather(Y, slice, 0) : null;
    yield { xBatch, yBatch, idx: slice };
    // ВАЖНО: потребитель должен сам делать dispose() после использования батча
  }
}

// ---------- Пример: преобразование HTMLImageElement → тензор [1,H,W,3], 0..1 ----------
export function imageToInputTensor(img, size = IMG_SIZE) {
  const can = document.createElement('canvas');
  can.width = size; can.height = size;
  const g = can.getContext('2d');
  g.drawImage(img, 0, 0, size, size);
  const data = g.getImageData(0,0,size,size).data;
  const arr = new Float32Array(size*size*3);
  for (let i=0,j=0;i<data.length;i+=4) {
    arr[j++] = data[i]   / 255;
    arr[j++] = data[i+1] / 255;
    arr[j++] = data[i+2] / 255;
  }
  return tf.tensor4d(arr, [1, size, size, 3]);
}

// ---------- Пример: oneHot с безопасными индексами ----------
export function safeOneHot(classIndexArray, depth) {
  const idx = int32Indices(classIndexArray);
  const oh  = tf.oneHot(idx, depth);
  idx.dispose();
  return oh;
}
