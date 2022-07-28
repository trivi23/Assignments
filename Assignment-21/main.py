import tensorflow as tf
import numpy as np
from tensorflow import keras

x_train = np.array([[1.0, 1.0]])
y_train = np.array([1.0])

for i in range(3, 10000, 2):
    x_train = np.append(x_train, [[i, i]], axis=0)
    y_train = np.append(y_train, [i*i])

print(x_train)
print(y_train)

x_test = np.array([[2.0, 2.0]])
y_test = np.array([4.0])

for i in range(3, 8000, 2):
    x_test = np.append(x_test, [[i, i]], axis=0)
    y_test = np.append(y_test, [i*i])

print(x_test)
print(y_test)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(1)

])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=1)

model.save("h2")
print("Saved model to disk")

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
a = np.array([[2000, 3000], [4, 5]])
print(model.predict(a))
