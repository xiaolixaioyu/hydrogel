const MODEL_PATH = 'model/hydrogel_temp.tflite';
const INPUT_SIZE = 128;
const THRESHOLDS = {
  normal: 37.5,
  mild: 38.5
};

let interpreter = null;

function loadModel() {
  return new Promise((resolve, reject) => {
    if (interpreter) {
      resolve(true);
      return;
    }

    try {
      const tflite = requirePlugin('tflite');

      tflite.loadModel({
        filePath: MODEL_PATH,
        onSuccess: (res) => {
          interpreter = res;
          console.log('TFLite模型加载成功');
          resolve(true);
        },
        onError: (err) => {
          console.error('TFLite模型加载失败:', err);
          resolve(false);
        }
      });
    } catch (e) {
      console.error('TFLite插件加载失败:', e);
      resolve(false);
    }
  });
}

function predict(inputData) {
  return new Promise((resolve, reject) => {
    if (!interpreter) {
      reject(new Error('模型未加载'));
      return;
    }

    try {
      const inputTensor = interpreter.getInputTensor(0);
      inputTensor.data = inputData;

      interpreter.invoke()
        .then(() => {
          const outputTensor = interpreter.getOutputTensor(0);
          const output = outputTensor.data[0];
          resolve(output);
        })
        .catch(reject);
    } catch (e) {
      reject(e);
    }
  });
}

function getTemperatureStatus(temp) {
  if (temp < THRESHOLDS.normal) {
    return {
      level: 'normal',
      label: '正常',
      color: '#2ecc71',
      message: '体温正常，请继续观察'
    };
  } else if (temp <= THRESHOLDS.mild) {
    return {
      level: 'mild',
      label: '轻度发热',
      color: '#f1c40f',
      message: '轻度发热，建议物理降温并密切观察'
    };
  } else {
    return {
      level: 'high',
      label: '高烧预警',
      color: '#e74c3c',
      message: '高烧预警，请立即就医！'
    };
  }
}

function preprocessImage(imagePath, size = INPUT_SIZE) {
  const canvas = wx.createOffscreenCanvas({
    type: '2d',
    width: size,
    height: size
  });

  const ctx = canvas.getContext('2d');
  const image = canvas.createImage();

  return new Promise((resolve, reject) => {
    image.onload = () => {
      ctx.drawImage(image, 0, 0, size, size);

      const imageData = ctx.getImageData(0, 0, size, size);
      const data = imageData.data;

      const normalizedData = new Float32Array(size * size * 3);
      let offset = 0;

      for (let i = 0; i < data.length; i += 4) {
        normalizedData[offset++] = data[i] / 255.0;
        normalizedData[offset++] = data[i + 1] / 255.0;
        normalizedData[offset++] = data[i + 2] / 255.0;
      }

      resolve(normalizedData);
    };

    image.onerror = reject;
    image.src = imagePath;
  });
}

function releaseModel() {
  if (interpreter) {
    interpreter = null;
    console.log('TFLite模型已释放');
  }
}

module.exports = {
  loadModel,
  predict,
  getTemperatureStatus,
  preprocessImage,
  releaseModel,
  INPUT_SIZE,
  THRESHOLDS
};
