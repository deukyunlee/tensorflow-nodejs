const tf = require("@tensorflow/tfjs");

// 1. 과거의 데이터를 준비합니다.

var open = [
  0.7614, 0.7591, 0.8253, 0.8697, 0.8763, 0.8314, 0.6825, 0.6656, 0.6555,
  0.6102, 0.6006, 0.6284, 0.6174, 0.6013, 0.6174, 0.6111, 0.6088, 0.6193,
];
var close = [
  0.7591, 0.8253, 0.8697, 0.8763, 0.8314, 0.6825, 0.6656, 0.6555, 0.6102,
  0.6006, 0.6284, 0.6174, 0.6013, 0.6174, 0.6111, 0.6088, 0.6193, 0.618,
];
var 원인 = tf.tensor(open);
var 결과 = tf.tensor(close);

// 2. 모델의 모양을 만듭니다.
var X = tf.input({ shape: [1] });
var Y = tf.layers.dense({ units: 1 }).apply(X);
var model = tf.model({ inputs: X, outputs: Y });
var compileParam = {
  optimizer: tf.train.adam(),
  loss: tf.losses.meanSquaredError,
};
model.compile(compileParam);

// 3. 데이터로 모델을 학습시킵니다.
// var fitParam = { epochs: 1000000 };
var fitParam = {
  epochs: 10000,
  callbacks: {
    onEpochEnd: function (epoch, logs) {
      console.log("epoch", epoch, logs);
    },
  },
}; // loss 추가 예제
model.fit(원인, 결과, fitParam).then(function (result) {
  // 4. 모델을 이용합니다.
  // 4.1 기존의 데이터를 이용
  var 예측한결과 = model.predict(원인);
  var 다음주온도 = [
    0.6118, 0.6283, 0.5966, 0.6368, 0.7199, 0.7381, 0.7509, 0.7617, 0.7784,
    0.7791,
  ];
  var 다음주원인 = tf.tensor(다음주온도);
  var 다음주결과 = model.predict(다음주원인);
  다음주결과.print();
  //   예측한결과.print();
});

// 4.2 새로운 데이터를 이용
