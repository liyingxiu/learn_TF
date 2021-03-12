import tensorflow as tf
import numpy as np

# create data
x_train_data = np.random.rand(10000).astype(np.float32)
y_train_data = x_train_data * 0.1 + 0.3

x_test_data = np.random.rand(1000).astype(np.float32)
y_test_data = x_test_data * 0.1 + 0.3

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32))

model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_data, y_train_data, epochs=10)
model.evaluate(x_test_data, y_test_data, verbose=2)



