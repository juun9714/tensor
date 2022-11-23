import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("끝")
            self.model.stop_training=True

callbacks=mycallback()
data=tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels)=data.load_data()

training_images=training_images/255.0
test_images=test_images/255.0

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

model.evaluate(test_images, test_labels)
classifications=model.predict(test_images)
print(classifications[0])
print(test_labels[0])

# print(tf.config.list_physical_devices('GPU'))