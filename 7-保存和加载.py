import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def build_model():
    model = keras.Sequential([
        layers.Dense(512, activation="relu", input_shape=(784, )),
        layers.Dropout(0.2),
        layers.Dense(10),
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

# 创建一个基本的模型实例
model = build_model()

# 显示模型的结构
model.summary()

# 创建一个只在训练期间保存权重的 tf.keras.callbacks.ModelCheckpoint 回调：
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个保存模型权重的回调
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)
# 使用新的回调训练模型
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback]) # 通过回调训练

# 这可能会生成与保存优化程序状态相关的警告。
# 这些警告（以及整个笔记本中的类似警告）
# 是防止过时使用，可以忽略。


# 创建一个基本模型实例
new_model = build_model()

# 评估模型
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


# 然后从 checkpoint 加载权重并重新评估：
# 加载权重
new_model.load_weights(checkpoint_path)

# 重新评估模型
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))



