App({
  globalData: {
    modelReady: false,
    temperatureThreshold: {
      normal: 37.5,
      mild: 38.5
    }
  },
  onLaunch() {
    console.log('水凝胶温度智能识别系统启动');
  },
  onShow() {
    console.log('应用显示');
  },
  onHide() {
    console.log('应用隐藏');
  }
});
