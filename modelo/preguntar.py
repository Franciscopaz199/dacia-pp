from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFile
from io import BytesIO

class ClasificadorFrutas:
    def __init__(self, modelo_calidad_path, modelo_clasificacion_path):
        """Inicializa los modelos para calidad y clasificación de frutas."""
        self.modelo_calidad = load_model(modelo_calidad_path)
        self.modelo_clasificacion = load_model(modelo_clasificacion_path)

    def cargar_imagen_by_path(self, path):
        """Carga una imagen desde un archivo.

        Args:
            path (str): Ruta de la imagen.

        Returns:
            file-like: Archivo de imagen cargado en formato binario.
        """
        with open(path, 'rb') as archivo:
            return archivo.read()  # Leer los bytes de la imagen

    def preprocesar_imagen(self, archivo):
        """Procesa una imagen para que sea compatible con los modelos.
        - Convierte la imagen cargada a un arreglo.
        - Devuelve la imagen original y la procesada lista para predicciones.

        Args:
            archivo (bytes): Contenido binario de la imagen cargada.

        Returns:
            tuple: Imagen original como arreglo, imagen procesada expandida para el modelo.
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img_original = Image.open(BytesIO(archivo))  # Leer los bytes en una imagen
        img_original.load()
        img_procesada = img_original.resize((100, 100), Image.ANTIALIAS)  # Cambia el tamaño según el modelo

        img_procesada = image.img_to_array(img_procesada)
        img_original = image.img_to_array(img_original)
        return img_original, np.expand_dims(img_procesada, axis=0)

    def verificar_frescura(self, img):
        """Predice la probabilidad de que una fruta esté fresca o podrida.

        Args:
            img (numpy array): Imagen procesada para el modelo.

        Returns:
            list: Probabilidades de estar fresca y podrida [fresca, podrida].
        """
        prob_fresca = round(100 * self.modelo_calidad.predict(img)[0][0], 3)
        prob_podrida = round(100 * (1 - self.modelo_calidad.predict(img)[0][0]), 3)
        return [prob_fresca, prob_podrida]

    def clasificar_fruta(self, img):
        """Clasifica el tipo de fruta y devuelve las probabilidades.

        Args:
            img (numpy array): Imagen procesada para el modelo.

        Returns:
            dict: Diccionario con tipos de frutas como claves y probabilidades como valores.
        """
        dict_frutas = {
            'manzana': round(self.modelo_clasificacion.predict(img)[0][0] * 100, 4),
            'banana': round(self.modelo_clasificacion.predict(img)[0][1] * 100, 4),
            'naranja': round(self.modelo_clasificacion.predict(img)[0][2] * 100, 4)
        }

        # Elimina valores insignificantes
        for fruta, valor in dict_frutas.items():
            if valor <= 0.001:
                dict_frutas[fruta] = 0.00

        return dict_frutas

# Ejemplo de uso
if __name__ == "__main__":
    clasificador = ClasificadorFrutas(modelo_calidad_path='local_rotten_lr2_final.h5',
                                      modelo_clasificacion_path='modeloTipoFrutasFinal.h5')

    # Cargar la imagen en formato binario
    archivo = clasificador.cargar_imagen_by_path('manzana.jpeg')

    # Preprocesar la imagen y clasificar
    img_original, img_procesada = clasificador.preprocesar_imagen(archivo)
    print(clasificador.clasificar_fruta(img_procesada))

    # Verificar la frescura de la fruta
    resultado_calidad = clasificador.verificar_frescura(img_procesada)
    print("Resultado de frescura:", resultado_calidad)
