import os
import cv2
import numpy as np
from tqdm import tqdm
from random import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv2D, BatchNormalization, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense)
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras as keras
import matplotlib.pyplot as plt

class ClasificadorFrutas:
    def __init__(self, ruta_entrenamiento, ruta_prueba):
        """Inicializa las rutas de los conjuntos de datos."""
        self.ruta_entrenamiento = ruta_entrenamiento
        self.ruta_prueba = ruta_prueba
        self.modelo = None

    def cargar_datos(self, tipo="prueba"):
        """Carga y preprocesa los datos de frutas.
        
        Args:
            tipo (str): Especifica si cargar datos de "entrenamiento" o "prueba".

        Returns:
            tuple: Arreglos X (imágenes) y Y (etiquetas).
        """
        calidad = ['apples', 'banana', 'oranges']
        X, Y, datos = [], [], []
        ruta = self.ruta_prueba if tipo == "prueba" else self.ruta_entrenamiento

        for categoria in tqdm(os.listdir(ruta)):
            etiqueta = calidad.index(next((q for q in calidad if q in categoria), None))
            ruta_categoria = os.path.join(ruta, categoria)

            for nombre_imagen in os.listdir(ruta_categoria):
                img = cv2.imread(os.path.join(ruta_categoria, nombre_imagen))
                img = cv2.resize(img, (100, 100))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                datos.append([img, etiqueta])

        print('Barajando los datos...')
        shuffle(datos)

        for imagen, etiqueta in datos:
            X.append(imagen)
            Y.append(etiqueta)

        return np.array(X), np.array(Y)

    def construir_modelo(self):
        """Construye y compila el modelo de clasificación."""
        modelo = Sequential()
        
        modelo.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu', input_shape=(100, 100, 3)))
        modelo.add(BatchNormalization())
        modelo.add(SeparableConv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
        modelo.add(MaxPooling2D((2, 2)))
        modelo.add(BatchNormalization())
        modelo.add(Dropout(0.3))

        modelo.add(SeparableConv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
        modelo.add(SeparableConv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D((2, 2)))
        modelo.add(Dropout(0.4))

        modelo.add(SeparableConv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
        modelo.add(SeparableConv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(SeparableConv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D((2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Flatten())
        modelo.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        modelo.add(Dropout(0.3))
        modelo.add(Dense(3, activation='softmax'))

        modelo.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        self.modelo = modelo

    def entrenar_modelo(self, X, Y, X_val, Y_val, epochs=40, batch_size=20):
        """Entrena el modelo con los datos proporcionados.
        
        Args:
            X (numpy array): Imágenes de entrenamiento.
            Y (numpy array): Etiquetas de entrenamiento.
            X_val (numpy array): Imágenes de validación.
            Y_val (numpy array): Etiquetas de validación.
            epochs (int): Número de épocas para entrenar.
            batch_size (int): Tamaño del batch.

        Returns:
            History: Historial del entrenamiento.
        """
        lr_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1, min_lr=0.00002, cooldown=2)
        check_point = ModelCheckpoint(filepath='/kaggle/working/fruit_cata.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        X, X_val = X / 255.0, X_val / 255.0
        Y, Y_val = to_categorical(Y), to_categorical(Y_val)

        history = self.modelo.fit(X, Y, batch_size=batch_size, validation_data=(X_val, Y_val), epochs=epochs, callbacks=[lr_rate, check_point])
        return history

    def evaluar_modelo(self, X, Y):
        """Evalúa el modelo con los datos proporcionados.

        Args:
            X (numpy array): Imágenes de evaluación.
            Y (numpy array): Etiquetas de evaluación.

        Returns:
            list: Resultados de la evaluación [pérdida, precisión].
        """
        X = X / 255.0
        Y = to_categorical(Y)
        resultados = self.modelo.evaluate(X, Y)
        print("Resultados de la evaluación:", resultados)
        return resultados

    def graficar_historial(self, history):
        """Genera gráficas del historial de entrenamiento.

        Args:
            history (History): Historial del entrenamiento.
        """
        plt.figure(1, figsize=(20, 12))
        plt.subplot(1, 2, 1)
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.plot(history.history["loss"], label="Pérdida de entrenamiento")
        plt.plot(history.history["val_loss"], label="Pérdida de validación")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.xlabel("Épocas")
        plt.ylabel("Precisión")
        plt.plot(history.history["accuracy"], label="Precisión de entrenamiento")
        plt.plot(history.history["val_accuracy"], label="Precisión de validación")
        plt.legend()
        plt.grid(True)
        plt.show()

    def guardar_modelo(self, ruta):
        """Guarda el modelo entrenado en la ruta especificada."""
        self.modelo.save(ruta)
        print(f"Modelo guardado en {ruta}")

# Ejemplo de uso
clasificador = ClasificadorFrutas(
    '1/dataset/dataset/train', 
    '1/dataset/dataset/test'
)

X_val, Y_val = clasificador.cargar_datos(tipo="prueba")
X_train, Y_train = clasificador.cargar_datos(tipo="entrenamiento")
clasificador.construir_modelo()
historia = clasificador.entrenar_modelo(X_val, Y_val, X_val, Y_val, epochs=1)
clasificador.guardar_modelo('modeloTipoFrutasFinal.h5')