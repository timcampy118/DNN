import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# importing the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the values to between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Creating the structure of our network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Teaching our network how to evaluate with loss function and optimizer method
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# Displaying stat on model and saving it
print(model.evaluate(x_test, y_test))
model.save('handwriting128.model')
predictions = model.predict([x_test])

# Randomly trying a value for checking manually
randomNum = random.randint(0, len(x_test))
print(np.argmax(predictions[randomNum]))
plt.imshow(x_test[randomNum], plt.cm.binary)
plt.show()

# Testing for showing the value
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()
