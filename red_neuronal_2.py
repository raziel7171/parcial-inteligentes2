import tensorflow as tf
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

#componentes de la red neuronal

from keras.models import Sequential
from keras.layers import InputLayer, Input, Conv2D, MaxPool2D, Reshape, Dense, Flatten
#######################################################################

def cargarDatos(rutaOrigen, numeroCategorias, limite, width, height):
    imagenesCargadas = []
    valorEsperado = []
    for categoria in range (0, numeroCategorias):
        for idImagen in range(1, limite[categoria]):
            ruta = rutaOrigen + str(categoria) + "/" + str(categoria) + " ("+ str(idImagen) +")" + ".png"
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (width, height))
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
        print(ruta)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados

########################################################################

ancho = 128
alto = 128
pixeles = ancho * alto
# Imagen RGB -->
numero_canales = 1
forma_imagen = (ancho, alto, numero_canales)
numero_clases = 10

cantidadDatosEntrenamiento = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
cantidadDatosPruebas = [17, 17, 17, 17, 17, 17, 17, 17, 17, 17]

################### cargar imagenes

imagenes, probabilidades = cargarDatos("dataset/train/", numero_clases, cantidadDatosEntrenamiento, ancho, alto)

model = Sequential()
#capas de entradas
model.add(InputLayer(input_shape= (pixeles,)))
model.add(Reshape(forma_imagen))


#capas ocultas
#capas convolucionales

model.add(Conv2D(kernel_size=5,strides=2,filters=16,padding="same",activation="relu",name="capa_1"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=3,strides=1,filters=32,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

# model.add(Conv2D(kernel_size=9,strides=4,filters=64,padding="same",activation="relu",name="capa_3"))
# model.add(MaxPool2D(pool_size=2,strides=2))

#aplanamiento
model.add(Flatten())
model.add(Dense(128,activation="relu"))

#salida
model.add(Dense(numero_clases,activation="softmax"))

#traducir keras a tensorflow

model.compile(optimizer="SGD",loss="Poisson", metrics=["TopKCategoricalAccuracy"])

#entrenamiento
model.fit(x=imagenes, y=probabilidades, epochs=100,batch_size=30)

#prueba de modelo

imagenesPrueba, probabilidadesPruebas = cargarDatos("dataset/test/", numero_clases, cantidadDatosPruebas, ancho, alto)
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPruebas)
print("accuracy=", resultados[1])

#Guardar modelo
ruta = "models/modelo2.h5"
model.save(ruta)

#informe de estructura de la red
model.summary()

informacion = model.compile(optimizer="SGD",loss="Poisson",metrics=['opKCategoricalAccuracy',tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
print(informacion)

scnn_pred = model.predict(imagenesPrueba, batch_size=60, verbose=1)
scnn_predicted = np.argmax(scnn_pred, axis=1)

#Creamos la matriz de confusión
scnn_cm = confusion_matrix(np.argmax(probabilidadesPruebas, axis=1), scnn_predicted)

# Visualiamos la matriz de confusión
scnn_df_cm = pd.DataFrame(scnn_cm, range(10), range(10))
plt.figure(figsize=(20, 14))
sn.set(font_scale=1.4) #for label size
sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12}) # font size
plt.show()

scnn_report = classification_report(np.argmax(probabilidadesPruebas, axis=1), scnn_predicted)
print(scnn_report)