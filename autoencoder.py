import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Carregar o dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizar os dados
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Modificar a entrada para ter 3 canais (RGB)
x_train_rgb = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
x_test_rgb = np.repeat(x_test[..., np.newaxis], 3, axis=-1)

# Criar o autoencoder
input_img = keras.Input(shape=(28, 28, 3))  # Alterar para 3 canais

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Saída com 3 canais

# Definir o modelo do autoencoder
autoencoder = keras.Model(input_img, decoded)

# Compilar o modelo
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Treinar o autoencoder com a entrada RGB
autoencoder.fit(x_train_rgb, x_train_rgb, epochs=50, batch_size=128, shuffle=True, validation_data=(x_test_rgb, x_test_rgb))

# Função para colorir os dígitos de acordo com a paridade
def colorize_output(predictions, labels):
    colorized = np.zeros((predictions.shape[0], 28, 28, 3))
    for i in range(predictions.shape[0]):
        if labels[i] % 2 == 0:  # Par: Azul
            colorized[i, :, :, 2] = predictions[i, :, :, 2]  # Canal Azul
        else:  # Ímpar: Vermelho
            colorized[i, :, :, 0] = predictions[i, :, :, 0]  # Canal Vermelho
    return colorized

# Testar o autoencoder e colorir os dígitos
decoded_imgs = autoencoder.predict(x_test_rgb)

# Selecionar uma amostra de cada dígito de 0 a 9
def get_ordered_digits(x_data, y_data, predictions):
    ordered_digits = []
    for digit in range(10):
        idx = np.where(y_data == digit)[0][0]  # Encontra o primeiro índice do dígito
        ordered_digits.append((x_data[idx], y_data[idx], predictions[idx]))
    return ordered_digits

# Obter dígitos de 0 a 9
ordered_digits = get_ordered_digits(x_test, y_test, decoded_imgs)

# Colorir os dígitos ordenados
ordered_imgs = np.array([img for _, _, img in ordered_digits])
ordered_labels = np.array([label for _, label, _ in ordered_digits])
colorized_imgs = colorize_output(ordered_imgs, ordered_labels)

# Exibir os resultados (amostras de 0 a 9 em ordem)
plt.figure(figsize=(20, 4))
for i in range(10):
    # Mostrar a imagem original
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(ordered_digits[i][0], cmap='gray')
    plt.title(f"Original {ordered_digits[i][1]}")
    plt.gray()

    # Mostrar a imagem colorizada
    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(colorized_imgs[i])
    plt.title(f"Colorized {ordered_digits[i][1]}")

plt.show()

