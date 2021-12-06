import numpy as np
import cv2
from os import remove
from os import path
from prediccion import Prediccion

nameWindow = "Calcular"
img_counter = 0
listVertices = []
clases = [1,2,3,4,5,6,7,8,9,10]
acumulador = 0
resultadoVisual = 0

ancho = 128
alto = 128

def nothing(x):
	pass

def constructorVentana():
	cv2.namedWindow(nameWindow)
	cv2.createTrackbar("min",nameWindow, 0, 255, nothing)
	cv2.createTrackbar("max",nameWindow, 100, 255, nothing)
	cv2.createTrackbar("kernel",nameWindow, 1, 100, nothing)
	cv2.createTrackbar("areaMin",nameWindow, 500, 10000, nothing)

def calcularAreas(figuras):
	areas = []
	for figuraActual in figuras:
		areas.append(cv2.contourArea(figuraActual))
	return areas

img_counter = 0

def verificarVertices(vertices):
	verificacion = True
	x, y, w, h = cv2.boundingRect(vertices)
	for item in listVertices:
		xAux, yAux, wAux, hAux = cv2.boundingRect(item)
		if(x==xAux and y==yAux and w==wAux and h==hAux):
			verificacion = False
			return verificacion
	return verificacion

def recortar(vertices):
	global img_counter
	if path.exists("imagen_probar.png"):
		remove('imagen_probar.png')
	img_name = "imagen_probar.png"
	x, y, w, h = cv2.boundingRect(vertices)
	new_img=frame[y:y+h,x:x+w]
	cv2.imwrite(img_name,new_img)
	imageN = cv2.imread('imagen_probar.png')
	anchoNuevo = imageN.shape[1] #columnas
	altoNuevo = imageN.shape[0] # filas
	M = cv2.getRotationMatrix2D((anchoNuevo//2,altoNuevo//2),10,0.5)
	imageOut = cv2.warpAffine(imageN,M,(anchoNuevo,altoNuevo))
	cv2.imwrite(img_name, imageOut)
	img_counter += 1

def resultadoPrediccion():
	miModeloCNN = Prediccion("models/modelo1.h5", ancho, alto)
	imagen = cv2.imread("imagen_probar.png")
	claseResultado = miModeloCNN.predecir(imagen)
	print("el valor es", clases[claseResultado])
	return clases[claseResultado]

def detectarForma(imagen):
	global img_counter
	global acumulador
	global resultadoVisual
	imagenGris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
	min = cv2.getTrackbarPos("min", nameWindow)
	max = cv2.getTrackbarPos("max", nameWindow)
	bordes = cv2.Canny(imagenGris, min, max)
	tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
	kernel = np.ones((tamañoKernel,tamañoKernel), np.uint8)
	bordes = cv2.dilate(bordes, kernel)
	cv2.imshow("Bordes", bordes)
	figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	areas = calcularAreas(figuras)
	i = 0
	areaMin = cv2.getTrackbarPos("areaMin", nameWindow)
	mensaje = "Acumulado "+str(acumulador)
	cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.putText(imagen, "Resultado "+str(resultadoVisual), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
	for figuraActual in figuras:
		if areas[i] >= areaMin:
			vertices = cv2.approxPolyDP(figuraActual, 0.05*cv2.arcLength(figuraActual,True), True)
			if len(vertices) == 4:
				#mensaje = "Cuadrado"+str(areas[i])
				#cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.drawContours(imagen, [figuraActual], 0, (0,0,255), 2)
				if cv2.waitKey(1)%256 == 112:
					recortar(vertices)
					resultado = resultadoPrediccion()
					resultadoVisual = resultado
					acumulador += resultado
					mensaje = "Acumulado "+str(acumulador)
					cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
					mensaje2 = "Resultado "+str(resultado)
					cv2.putText(imagen, mensaje2, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
		i = i + 1
	return imagen


cap= cv2.VideoCapture(1)
constructorVentana()



while(True) :
	ret, frame = cap.read()
	imagen = detectarForma(frame)
	cv2.imshow('Imagen', imagen)
	if cv2.waitKey(1)%256 == 99: #c:
		recortar()
	# if cv2.waitKey(1)%256 == 99: #c:
	# 	img_name = "imagen_{}.png".format(img_counter)
	# 	cv2.imwrite(img_name, frame)
	# 	img_counter += 1
	if cv2.waitKey(1) & 0xFF == ord ('q'):
	 	break

cap.release()
cv2.destroyAllWindows()
