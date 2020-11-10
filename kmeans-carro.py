
from skimage.io import imread
from skimage.io import imsave
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans
import cv2

# Scaling the image pixels values within 0-1
img = imread('coches.jpg') / 255
centroides = 15
metodo = 1
mapa = img.copy()
largo = 0
font = cv2.FONT_HERSHEY_SIMPLEX
ubicacion = (0,0)


# Función de segmentación k-means para diferentes metodo de inicio
def km(centroides,img,metodo):
    values = np.zeros(shape=(centroides,3))
    count = 0
    paso = int(255/(centroides+1))
    nivel = 0
    canal = 0
    
    
    #Matriz para inicializar los centroides de manera acumulados
    while metodo == 0:
        nivel = nivel+paso
        values[count][0] = 125
        values[count][1] = 125
        values[count][2] = 125
        count = count+1
        if count > 7:
            break
    #Matriz para inicializar los centroides de manera distribuida
    while metodo == 2:
        nivel = nivel+paso
        values[count][canal] = nivel
        count = count+1
        canal = canal +1
        if canal > 2:
            canal = 0
        if count > 7:
            break
    
    
    # Prepara la imagen para el proceso
    imagen = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    
    # Selecciona el metodo inicialización de los centroides
    if metodo == 0 or metodo == 2:
        kmeans = KMeans(n_clusters=centroides,init=values,n_init=1).fit(imagen)
    else:
        kmeans = KMeans(n_clusters=centroides,init='random').fit(imagen)
        
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    # Convierte la información en imagen
    imagen_final = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
    return(imagen_final)


# Segmentación por color naranja
imagen_final = km(centroides,img,metodo)
imagen_final = imagen_final*255
imagen_final = np.uint8(imagen_final)
lower_blue = np.array([235,180, 15])
upper_blue = np.array([255,200,40])
mask = cv2.inRange(imagen_final,lower_blue,upper_blue)
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Etiquitado de carros naranja
for cnt in contours:
    area = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    
    if area > 2000:
        f= cv2.boundingRect(cnt)
        print(f)
        cv2.rectangle(mapa,(f[0],f[1]),(f[0]+f[2],f[1]+f[3]+20),(0,0,0),5)
        
        # Determinación del deportivo
        if largo < f[2]:
            largo = f[2]
            ubicacion = (f[0],f[1]-20)
    
cv2.putText(mapa,'Deportivo',ubicacion, font, 2,(0,0,0),2,cv2.LINE_AA)


# Guardado de imagen segmetado por k-means
plt.imshow(imagen_final)
plt.title('Clustered Image')
plt.show()
imsave('coches0.jpg',imagen_final)

# Guardado de imagen con objetos detectados
plt.imshow(mapa)
plt.title('M')
plt.show()
imsave('cochesf.jpg',mapa)