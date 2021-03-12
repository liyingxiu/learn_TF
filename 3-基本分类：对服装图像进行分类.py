import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# 加载数据集会返回四个 NumPy 数组：
# 图像是 28x28 的 NumPy 数组，像素值介于 0 到 255 之间。标签是整数数组，介于 0 到 9 之间。
fashion_mnist = keras.datasets.fashion_mnist
(train_image, train_labels), (test_image, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_image.shape)
print(train_labels)
print(train_labels.shape)

# 查看第一个图像
plt.figure()
plt.imshow(train_image[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 将这些值缩小至 0 到 1 之间，然后将其馈送到神经网络模型。为此，请将这些值除以 255
train_image = train_image / 255
test_image = test_image / 255

# 显示训练集中的前 25 个图像，并在每个图像下方显示类名称
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_image[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 构建模型
model = keras.Sequential([
    # 该网络的第一层 tf.keras.layers.Flatten 将图像格式从二维数组（28 x 28 像素）转换成一维数组（28 x 28 = 784 像素）。
    # 层没有要学习的参数，它只会重新格式化数据。
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10)
])

# 编译模型
# 以下内容是在模型的编译步骤中添加的：
# 损失函数 - 用于测量模型在训练期间的准确率。您会希望最小化此函数，以便将模型“引导”到正确的方向上。
# 优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
# 指标 - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# 训练模型
model.fit(train_image, train_labels, epochs=10)

# 评估准确率: 训练准确率和测试准确率之间的差距代表过拟合
test_loss, test_acc = model.evaluate(test_image, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 进行预测
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_image)
print(class_names[np.argmax(predictions[0])])


# 使用训练好的模型对单个图像进行预测。
img = test_image[1]
print(img.shape)   # (28, 28)
# tf.keras 模型经过了优化，可同时对一个批或一组样本进行预测。因此，即便您只使用一个图像，您也需要将其添加到列表中：
img = (np.expand_dims(img, 0))
print(img.shape)   # (1, 28, 28)

predictions_single = probability_model.predict(img)
print(class_names[np.argmax(predictions_single[0])])






