import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, MaxPooling2D, Activation,
                                      Flatten, BatchNormalization, SeparableConv2D)
from tensorflow.keras.models import Sequential
import tensorflow as tf

class ClasificadorFrutas:
    def __init__(self, ruta_datos_entrenamiento, ruta_datos_prueba):
        self.ruta_entrenamiento = ruta_datos_entrenamiento
        self.ruta_prueba = ruta_datos_prueba
        self.modelo = None

    def cargar_imagenes_muestra(self, max_imagenes=6):
        """Carga una muestra limitada de imágenes desde el conjunto de entrenamiento."""
        imagenes = []
        for subdirectorio in tqdm(os.listdir(self.ruta_entrenamiento)):
            ruta_subdirectorio = os.path.join(self.ruta_entrenamiento, subdirectorio)
            for i, nombre_imagen in enumerate(os.listdir(ruta_subdirectorio)):
                if i >= max_imagenes:
                    break
                img = cv2.imread(os.path.join(ruta_subdirectorio, nombre_imagen))
                img = cv2.resize(img, (100, 100))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imagenes.append(img)
        return np.array(imagenes)

    def mostrar_imagenes(self, imagenes, titulos=None):
        """Muestra un conjunto de imágenes en una cuadrícula con títulos opcionales."""
        if len(imagenes) == 36:
            fig, axes = plt.subplots(6, 6, figsize=(20, 20))
            for i, img in enumerate(imagenes):
                ax = axes[i // 6, i % 6]
                ax.imshow(img)
                ax.axis('off')
                if titulos is not None:
                    ax.set_title(titulos[i])
            plt.show()
        else:
            print("El número de imágenes no es suficiente para mostrar.")

    def cargar_datos(self, tipo="entrenamiento"):
        """Carga las imágenes y etiquetas del conjunto de datos especificado."""
        calidad = ['fresh', 'rotten']
        imagenes, etiquetas = [], []
        datos = self.ruta_entrenamiento if tipo == "entrenamiento" else self.ruta_prueba

        for categoria in tqdm(os.listdir(datos)):
            etiqueta = 0 if calidad[0] in categoria else 1
            ruta_categoria = os.path.join(datos, categoria)

            for nombre_imagen in os.listdir(ruta_categoria):
                img = cv2.imread(os.path.join(ruta_categoria, nombre_imagen))
                img = cv2.resize(img, (100, 100))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imagenes.append(img)
                etiquetas.append(etiqueta)

        shuffle(imagenes)
        return np.array(imagenes), np.array(etiquetas)

    def construir_modelo(self):
        """Construye y compila el modelo de red neuronal."""
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

        modelo.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
        modelo.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(MaxPooling2D((2, 2)))
        modelo.add(Dropout(0.5))

        modelo.add(Flatten())
        modelo.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        modelo.add(Dropout(0.3))
        modelo.add(Dense(1, activation='sigmoid'))

        modelo.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        self.modelo = modelo
        print(self.modelo.summary())

    def entrenar_modelo(self, X, Y, X_val, Y_val, epochs=50, batch_size=20):
        """Entrena el modelo con los datos proporcionados."""
        lr_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1, mode='max', min_lr=0.00002, cooldown=2)
        check_point = tf.keras.callbacks.ModelCheckpoint(filepath='/modelo/rotten.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        X, X_val = X / 255.0, X_val / 255.0

        history = self.modelo.fit(X, Y, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, callbacks=[lr_rate, check_point])
        return history

    def evaluar_modelo(self, X_val, Y_val):
        """Evalúa el modelo en el conjunto de datos de validación."""
        X_val = X_val / 255.0
        resultados = self.modelo.evaluate(X_val, Y_val)
        print("Resultados de la evaluación:", resultados)
        return resultados

    def guardar_modelo(self, ruta):
        """Guarda el modelo entrenado en la ruta especificada."""
        self.modelo.save(ruta)
        print(f"Modelo guardado en {ruta}")

# Uso de la clase
clasificador = ClasificadorFrutas(
    ruta_datos_entrenamiento='1/dataset/dataset/train', 
    ruta_datos_prueba='1/dataset/dataset/test'
)

X_val, Y_val = clasificador.cargar_datos(tipo="prueba")
clasificador.construir_modelo()
historia = clasificador.entrenar_modelo(X_val, Y_val, X_val, Y_val, epochs=1)
#clasificador.evaluar_modelo(X_val, Y_val)
clasificador.guardar_modelo('modelofinal.h5')
