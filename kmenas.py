
import numpy as np
from sklearn.cluster import KMeans


# La función km mecesita las siguientes varaibles
# centroides varaibles tipo int, numero de centroides para el kmeans
# img, imagen que se desea procesar, formato rgb
# metodo, valor int entre 0 a 2, representa el metodo para ubicar los centroides
#       0 acumulados en una zona
#       1 ubicación aleatoria
#       2 distribuidos uniformente en el espacio
# La función devuelve una imagen en formato rgb

def km(centroides,img,metodo):
    values = np.zeros(shape=(centroides,3))
    count = 0
    paso = int(255/(centroides+1))
    nivel = 0
    canal = 0
    
    
    
    while metodo == 0:
        nivel = nivel+paso
        values[count][0] = 125
        values[count][1] = 125
        values[count][2] = 125
        count = count+1
        if count > 7:
            break
    
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