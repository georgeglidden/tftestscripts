# yanked directly from https://www.tensorflow.org/tutorials/quickstart/beginner
import tensorflow as tf
from time import time
from statistics import mean
import sys
t = int(sys.argv[1])
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

cpu_dur = []
gpu_dur = []

try:
    with tf.device('/cpu:0'):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
        for i in range(t):
            print(f'cpu {i+1}/{t}')
            cpu_start = time()
            model.fit(x_train, y_train, epochs=5, verbose=0)
            model.evaluate(x_test,  y_test, verbose=0)
            cpu_finish = time()
            cpu_dur.append(cpu_finish - cpu_start)
except e:
    print('cpu execution failed')
try:
    with tf.device('/gpu:0'):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
        for i in range(t):
            print(f'gpu {i+1}/{t}')
            gpu_start = time()
            model.fit(x_train, y_train, epochs=5, verbose=0)
            model.evaluate(x_test,  y_test, verbose=0)
            gpu_finish = time()
            gpu_dur.append(gpu_finish - gpu_start)
except e:
    print('gpu execution failed')
cpu_dur_str = ', '.join([str(x) for x in cpu_dur]) + '\n'
gpu_dur_str = ', '.join([str(x) for x in gpu_dur]) + '\n'
print(f'cpu time: {mean(cpu_dur)}\ngpu time: {mean(gpu_dur)}')
with open('mnist_classifier_results.csv', 'w') as results:
    results.write(cpu_dur_str + gpu_dur_str)
