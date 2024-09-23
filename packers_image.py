import numpy as np
import tensorflow as tf
import glob
import os as os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

X_train = []
Y_train = []
X_test = []
Y_test = []

for f in glob.glob("football_photos/*/*/*.jpg"):
    img_data = tf.io.read_file(f)
    img_data = tf.io.decode_jpeg(img_data)
    img_data = tf.image.resize(img_data, [100, 100])
    f = f.replace(os.sep, '/')

    if f.split("/")[1] == "train":
        X_train.append(img_data)
        Y_train.append(int(f.split("/")[2].split("_")[0]))
    elif f.split("/")[1] == "test":
        X_test.append(img_data)
        Y_test.append(int(f.split("/")[2].split("_")[0]))

X_train = np.array(X_train) / 255.0
Y_train = np.array(Y_train)
X_test = np.array(X_test) / 255.0
Y_test = np.array(Y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 3)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=20)

model.evaluate(X_test, Y_test)
