import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

((trainX, trainY), (testX, testY)) = load_data("mnist.npz")

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# 生成trainX1k数据集（小数据）
trainX1k_x = []
trainX1k_y = []

for i in range(0,10):
    index_y = 0        #用于记录当前label的位置，并且完成一次最外层循环后，回到第一个数据
    for label in trainY:    # 在所有数据中找i
        if label == i:             # 标签值
            trainX1k_x.append(trainX[index_y])    # 加入相应二维数组（图像）
            trainX1k_y.append(label)
            if len(trainX1k_y) >= (i+1) * 100:             # 每次加入100个
                break                            # 够100个就跳出
        index_y = index_y + 1                    # 索引值+1

trainX1k_x = np.array(trainX1k_x)
trainX1k_y = np.array(trainX1k_y)           #转换为numpy类型

#划分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(trainX1k_x, trainX1k_y, test_size=0.2, random_state=666)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#训练并记录
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/大数据模型")
model.fit(trainX, trainY, epochs=50,callbacks=[tensorboard_callback])

#测试
print("用小数据集验证")
model.evaluate(X_test, y_test, verbose=2)
print("用大数据集验证")
model.evaluate(testX, testY, verbose=2)