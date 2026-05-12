const predictor = require('../../utils/predictor.js');

Page({
  data: {
    tempImage: null,
    temperature: null,
    status: {
      level: 'normal',
      label: '正常',
      color: '#2ecc71',
      message: '体温正常，请继续观察'
    },
    loading: false,
    modelReady: false
  },

  onLoad() {
    this.initModel();
  },

  async initModel() {
    wx.showLoading({ title: '模型加载中...' });

    try {
      const success = await predictor.loadModel();
      if (success) {
        this.setData({ modelReady: true });
        wx.showToast({
          title: '模型加载成功',
          icon: 'success'
        });
      } else {
        wx.showToast({
          title: '模型加载失败',
          icon: 'none'
        });
      }
    } catch (err) {
      console.error('模型初始化失败:', err);
      wx.showToast({
        title: '模型加载失败',
        icon: 'none'
      });
    } finally {
      wx.hideLoading();
    }
  },

  chooseImage() {
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['camera', 'album'],
      success: (res) => {
        const tempFilePath = res.tempFilePaths[0];
        this.setData({ tempImage: tempFilePath });
        this.predictTemperature(tempFilePath);
      },
      fail: (err) => {
        console.error('选择图片失败:', err);
        wx.showToast({
          title: '请选择图片',
          icon: 'none'
        });
      }
    });
  },

  async predictTemperature(imagePath) {
    if (!this.data.modelReady) {
      wx.showToast({
        title: '模型未就绪',
        icon: 'none'
      });
      return;
    }

    this.setData({ loading: true, temperature: null });

    wx.showLoading({ title: '正在检测...' });

    try {
      const tempImage = wx.env.USER_DATA_PATH + '/temp_input.png';
      await this.downloadFile(imagePath, tempImage);

      const imageData = await this.preprocessImage(tempImage);

      const temperature = await predictor.predict(imageData);

      const status = predictor.getTemperatureStatus(temperature);

      this.setData({
        temperature: temperature.toFixed(1),
        status: status
      });

      if (status.level === 'high') {
        wx.vibrateLong();
        wx.showModal({
          title: '⚠️ 高烧预警',
          content: `当前体温 ${temperature.toFixed(1)}°C，请立即就医！`,
          showCancel: false,
          confirmText: '我知道了'
        });
      }

    } catch (err) {
      console.error('温度预测失败:', err);
      wx.showToast({
        title: '预测失败，请重试',
        icon: 'none'
      });
    } finally {
      this.setData({ loading: false });
      wx.hideLoading();
    }
  },

  downloadFile(tempFilePath, savePath) {
    return new Promise((resolve, reject) => {
      wx.getFileSystemManager().copyFile({
        srcPath: tempFilePath,
        destPath: savePath,
        success: () => resolve(savePath),
        fail: (err) => reject(err)
      });
    });
  },

  preprocessImage(imagePath) {
    return new Promise((resolve, reject) => {
      const canvas = wx.createOffscreenCanvas({
        type: '2d',
        width: 128,
        height: 128
      });

      const ctx = canvas.getContext('2d');
      const image = canvas.createImage();

      image.onload = () => {
        ctx.drawImage(image, 0, 0, 128, 128);

        const imageData = ctx.getImageData(0, 0, 128, 128);
        const data = imageData.data;

        const normalizedData = new Float32Array(128 * 128 * 3);
        let offset = 0;

        for (let i = 0; i < data.length; i += 4) {
          normalizedData[offset++] = data[i] / 255.0;
          normalizedData[offset++] = data[i + 1] / 255.0;
          normalizedData[offset++] = data[i + 2] / 255.0;
        }

        resolve(normalizedData);
      };

      image.onerror = (err) => {
        reject(err);
      };

      image.src = imagePath;
    });
  },

  clearResult() {
    this.setData({
      tempImage: null,
      temperature: null,
      status: {
        level: 'normal',
        label: '正常',
        color: '#2ecc71',
        message: '体温正常，请继续观察'
      }
    });
  },

  onShareAppMessage() {
    return {
      title: '水凝胶温度检测',
      desc: '基于CNN的智能体温识别系统',
      path: '/pages/index/index'
    };
  }
});
