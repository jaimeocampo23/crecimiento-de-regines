import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float
from skimage import exposure
from skimage import color, morphology, io

plt.close('all')

Ima = io.imread('Brain.png')
# gris = color.rgb2gray(Ima)
plt.figure(1)
plt.imshow(Ima, cmap='gray')
plt.ion()

tol = 0.2
f, c = Ima.shape
pos = np.int32(plt.ginput(0, 0))
aux1 = np.zeros([f, c], dtype=np.byte)
aux2 = np.zeros([f, c], dtype=np.byte)

aux1[pos[:, 1], pos[:, 0]] = 1
pixeles = Ima[pos[:, 1], pos[:, 0]]
promedio = np.mean(pixeles)

while (np.sum(aux1) != np.sum(aux2)):
    plt.cla()
    aux2 = np.copy(aux1)
    bordes = morphology.binary_dilation(aux1) - aux1
    pos_borde = np.argwhere(bordes)
    gris_bordes = Ima[pos_borde[:, 0], pos_borde[:, 1]]
    compara = list(np.logical_and([gris_bordes > (promedio - tol)], [gris_bordes < (promedio + tol)]))
    datos = pos_borde[compara]
    aux1[datos[:, 0], datos[:, 1]] = 1
    plt.imshow(aux1, cmap='gray')
    plt.pause(1)



