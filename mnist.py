import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import files
from tensorflow import keras
from keras import layers

# Carregar o modelo MNIST pré-treinado
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

# Compilar e treinar o modelo
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

# Fazer upload de uma imagem de dígito
uploaded = files.upload()

for fn in uploaded.keys():
    # Carregar a imagem e prepará-la
    image = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = cv2.bitwise_not(image)  # Inverter cores (se necessário)
    image = image.astype("float32") / 255
    image = np.expand_dims(image, axis=-1)  # Expandir para 3 dimensões
    image = np.expand_dims(image, axis=0)   # Expandir para 4 dimensões (batch)

    # Prever o dígito
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    # Exibir resultado
    plt.imshow(image[0, :, :, 0], cmap='gray')
    plt.title(f'Previsão: {predicted_digit}')
    plt.axis('off')
    plt.show()
