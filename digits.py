## script para extrair Dummy Features da base de digitos manuscritos
## As imagens sao normalizadas no tamanho indicado nas variavies X e Y
## Aprendizagem de Maquina, Prof. Luiz Eduardo S. Oliveira
##
##

import cv2
import os
import numpy as np
import random


def load_images(path_images, fout, largura, altura):
    print('Loading images...')
    archives = os.listdir(path_images)
    images = []
    arq = open('digits/files.txt')
    lines = arq.readlines()
    print('Extracting dummy features')
    for line in lines:
        aux = line.split('/')[1]
        image_name = aux.split(' ')[0]
        label = line.split(' ')[1]
        label = label.split('\n')

        for archive in archives:
            if archive == image_name:
                image = cv2.imread(path_images + '/' + archive, 0)
                rawpixel(image, label[0], fout, largura, altura)

        # images.append((image, label))

    print('Done. Take a look into file features_' + str(largura) + 'x' + str(altura) + '.txt')
    return images


#########################################################
# Usa o valor dos pixels como caracteristica
#
#########################################################


def rawpixel(image, label, fout, largura, altura):
    # redimensiona a imagem
    image = cv2.resize(image, (largura, altura))
    fout.write(str(label) + " ")

    indice = 0
    for i in range(altura):
        for j in range(largura):
            if (image[i][j] > 128):
                v = 0
            else:
                v = 1

            fout.write(str(indice) + ":" + str(v) + " ")
            indice = indice + 1

    fout.write("\n")


if __name__ == "__main__":

    # inicial e novas hipoteses, após analisar a base
    hipoteses = [
        {'largura': 10, 'altura': 10},
        {'largura': 10, 'altura': 30},
        {'largura': 12, 'altura': 15},
        {'largura': 20, 'altura': 10},
        {'largura': 20, 'altura': 25},
        {'largura': 20, 'altura': 70},
        {'largura': 30, 'altura': 40},
        {'largura': 32, 'altura': 40},
        {'largura': 36, 'altura': 46},
        {'largura': 37, 'altura': 47},
        {'largura': 40, 'altura': 50},
        {'largura': 50, 'altura': 60},
        {'largura': 60, 'altura': 30},
        {'largura': 60, 'altura': 70},
        {'largura': 99, 'altura': 81}
    ]

    for hipotese in hipoteses:
        largura = hipotese['largura']
        altura = hipotese['altura']
        filename = 'features_' + str(largura) + 'x' + str(altura) + '.txt'
        fout = open(filename, "w")
        images = load_images('digits/data', fout, largura, altura)
        fout.close
