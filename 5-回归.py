import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 首先下载数据集
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")


# 使用 pandas 导入数据集。
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
print(type(dataset))
print(dataset.tail())
print("************************************")

# 数据清洗: 数据集中包括一些未知值。
print(dataset.isna().sum())
# 为了保证这个初始示例的简单性，删除这些行。
dataset = dataset.dropna()
# "Origin" 列实际上代表分类，而不仅仅是一个数字。所以把它转换为独热码 （one-hot）:
origin = dataset.pop("Origin")
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print("************************************")
print(dataset.tail())
print("************************************")

# 拆分训练数据集和测试数据集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 查看训练数据集中的平均数、中位数和方差等
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)
print("************************************")

# 从标签中分离特征
# 将特征值从目标值或者"标签"中分离。 这个标签是你使用训练模型进行预测的值。
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")


# 数据归一化处理
def norm(x):
    # 使用均值和标准差来处理数据
    return (x - train_stats["mean"]) / train_stats["std"]


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# 模型的构建
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=["mae", "mse"])

    return model


model = build_model()


# 检查模型
model.summary()

# 现在试用下这个模型。从训练数据中批量获取‘10’条例子并对这些例子调用 model.predict
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)


# 对模型进行训练
# 对模型进行1000个周期的训练，并在 history 对象中记录训练和验证的准确性。
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 1000 == 0:
            print('')
        print('=>', end='')


EPOCHES = 500

# history = model.fit(
#     normed_train_data,
#     train_labels,
#     epochs=EPOCHES,
#     validation_split=0.2,
#     verbose=0,
#     callbacks=[PrintDot()]
# )
#
# # 使用 history 对象中存储的统计信息可视化模型的训练进度。
# hist = pd.DataFrame(history.history)
# # keys都有Index(['loss', 'mae', 'mse', 'val_loss', 'val_mae', 'val_mse', 'epoch'], dtype='object')
# hist["epoch"] = history.epoch # 得到一个list:[0, 1, 2,...,999]
# print("\n")
# print(hist.tail())


def plot_hist(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error")
    plt.plot(hist["epoch"], hist["mae"],
             label="Train Error")
    plt.plot(hist["epoch"], hist["val_mae"],
             label="Val Error")
    plt.ylim([0, 5])
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error")
    plt.plot(hist["epoch"], hist["mse"],
             label="Train Error")
    plt.plot(hist["epoch"], hist["val_mse"],
             label="Val Error")
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


# 可以发现，误差没有改进，反而一直在上升，此时可以采用EarlyStopping callback 来测试每个 epoch 的训练条件。
# 如果经过一定数量的 epochs 后没有改进，则自动停止训练。
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHES,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, PrintDot()]
)

plot_hist(history)


# 使用测试集来测试模型的效果
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# 做预测
# 最后，使用测试集中的数据预测 MPG 值:
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# 这看起来我们的模型预测得相当好。我们来看下误差分布。
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

