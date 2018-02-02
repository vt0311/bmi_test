import pandas as pd
import numpy as np
import tensorflow as tf

# 키, 몸무게, 레이블이 적힌 CSV 파일 읽어 들이기 --- (※1)
csv = pd.read_csv("bmi.csv")

# 데이터 정규화 --- (※2) : 
# 데이터를 0 이상, 1미만으로 지정한다.
# 키의 최대값은 200cm, 몸무게의 최대값은 100kg 로 정규화한다.
csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 100

# 레이블을 배열로 변환하기 --- (※3)
# - thin(저체중)=(1,0,0) / normal(정상)=(0,1,0) / fat(비만)=(0,0,1)
# 
bclass = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}
csv["label_pat"] = csv["label"].apply(lambda x : np.array(bclass[x]))

# 테스트를 위한 데이터 분류 --- (※4)
test_csv = csv[15000:20000] # 2만개의 데이터 중 뒤의 5000개를 테스트 데이터로 지정
test_pat = test_csv[["weight","height"]]
test_ans = list(test_csv["label_pat"])

# 데이터 플로우 그래프 구출하기 --- (※5)
# 소프트맥스 회귀 사용함.

# 플레이스홀더 선언하기 
x  = tf.placeholder(tf.float32, [None, 2]) # 키와 몸무게 데이터 넣기
y_ = tf.placeholder(tf.float32, [None, 3]) # 정답 레이블 넣기

# 변수 선언하기 --- (※6)
W = tf.Variable(tf.zeros([2, 3])); # 가중치
b = tf.Variable(tf.zeros([3])); # 바이어스 - 0으로 초기화됨. 이 변수는 모델을 학습하면서 자동으로 조정됨.

# 소프트맥스 회귀 정의하기 --- (※7)
# x가 입력, w가 가중치, b가 바이어스
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 모델 훈련하기 --- (※8)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

# 정답률 구하기
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

# 세션 시작하기
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화하기

# 학습시키기 : 100개씩 3500번 학습시킴.
for step in range(3500):
    i = (step * 100) % 14000
    rows = csv[1 + i : 1 + i + 100]
    x_pat = rows[["weight","height"]]
    y_ans = list(rows["label_pat"])
    fd = {x: x_pat, y_: y_ans}
    sess.run(train, feed_dict=fd)
    if step % 500 == 0:
        cre = sess.run(cross_entropy, feed_dict=fd)
        acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
        print("step=", step, "cre=", cre, "acc=", acc)

# 최종적인 정답률 구하기
acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
print("정답률 =", acc)

# 결과 :
'''
2018-02-02 14:16:53.789106: I C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\
36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instruct
ions that this TensorFlow binary was not compiled to use: AVX AVX2
step= 0 cre= 108.663 acc= 0.3242
step= 500 cre= 57.5887 acc= 0.8904
step= 1000 cre= 45.0209 acc= 0.898
step= 1500 cre= 41.6543 acc= 0.9566
step= 2000 cre= 34.664 acc= 0.943
step= 2500 cre= 34.287 acc= 0.9674
step= 3000 cre= 26.8808 acc= 0.9726
정답률 = 0.9712
'''