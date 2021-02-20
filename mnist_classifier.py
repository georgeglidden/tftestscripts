# yanked directly from https://www.tensorflow.org/tutorials/quickstart/beginner
import tensorflow as tf
from time import time
import sys
t = int(sys.argv[1])
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

cpu_dur = 0.0
gpu_dur = 0.0

for i in range(t):
    print(f'{i+1}/{t}')
    try:
        with tf.device('/cpu:0'):
            cpu_start = time()
            model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=5, verbose=0)
            model.evaluate(x_test,  y_test, verbose=0)
            cpu_finish = time()
            cpu_dur += cpu_finish - cpu_start
    except e:
        print('cpu execution failed')
    try:
        with tf.device('/gpu:0'):
            gpu_start = time()
            model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=5, verbose=0)
            model.evaluate(x_test,  y_test, verbose=0)
            gpu_finish = time()
            gpu_dur += gpu_finish - gpu_start
    except e:
        print('gpu execution failed')
print(f'cpu time: {cpu_dur/t}\ngpu time: {gpu_dur/t}')
