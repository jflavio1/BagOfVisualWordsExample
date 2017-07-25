__author__ = "Jose Flavio Quispe Irrazabal"
__copyright__ = "Copyright 2017, Laboratorio 4"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "jflavio90@gmail.com"
__status__ = "Production"

import numpy as np
import cv2
import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


class BOVHelper:
    def __init__(self, numClusters):
        self.numClusters = numClusters
        self.kmeans_obj = KMeans(n_clusters=self.numClusters)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.supportVectorMachine = SVC()
        self.img_helper = ImageHelper()
        self.knn_obj = cv2.ml.KNearest_create()

    def cluster(self):
        """
        clustering usando el algoritmo k means
        con el array de descripciones creado en el metodo formatND
        """
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

    def formatND(self, l):
        """
        se resctructura la lista en un arreglo vstack
        que tendra la forma: M imagenes x N caracteristicas
        y que sea utilizado por el modulo sklearn

        """
        vStack = np.array(l[0])
        for remaining in l:
            vStack = np.vstack((vStack, remaining))
        self.descriptor_vstack = vStack.copy()
        # print(self.descriptor_vstack)
        return vStack

    def developVocabulary(self,n_images, descriptor_list, kmeans_ret = None):

        """
        Cada clase tiene un histograma de palabras visuales que lo haran diferenciarse
        Cada imagen es una combinacion de multiples palabras visuales, entonces
        se crea un mega histograma o vocabulario que contendra la frecuencia
        en que cada palabra visual aparece.

        Entonces, este vocabulario sera el conjunto de todos los histogramas
        que identifican a cada clase

        """

        self.mega_histogram = np.array([np.zeros(self.numClusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            l = len(descriptor_list[i])
            for j in range(l):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_count + j]
                else:
                    idx = kmeans_ret[old_count + j]
                self.mega_histogram[i][idx] += 1
            old_count += l
        print("Vocabulario generado")
        # print(str(self.mega_histogram))
        plt.hist(self.mega_histogram)
        plt.title("Histograma de palabras visuales (todas las clases)")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.show()


    def standardize(self):
        """
        Se normaliza la data del histograma para obtener mejores resultados
        """
        self.scale = StandardScaler().fit(self.mega_histogram)
        self.mega_histogram = self.scale.transform(self.mega_histogram)
        print("Vocabulario normalizado")
        # print(str(self.mega_histogram))
        plt.hist(self.mega_histogram)
        plt.title("Histograma de palabras visuales normalizado")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.show()

    def trainSVC(self, train_labels):
        """
        usando el clasificador SVC (SVM) que ofrece sklearn

        """
        self.supportVectorMachine.fit(self.mega_histogram, train_labels)
        print("Entrenamiento completo")


    def knnTrain(self, train_labels):
        ttr = np.array([np.zeros(1) for i in range(len(train_labels))]).astype(np.float32)
        #print(str(self.mega_histogram))
        #print(str(ttr))
        self.knn_obj.train(self.mega_histogram, cv2.ml.ROW_SAMPLE, ttr)
        print("Se entreno el modelo")
        #ret, results, neighbours ,dist = knn.find_nearest(newcomer, 3)

    def classifyKnn(self, im, neighbors = 7):

        kp, des = self.img_helper.features(im)

        i = np.array([des[i] for i in range(self.numClusters)]).astype(np.float32)

        print(str(i))
        ret, results, neighbours, dist = self.knn_obj.findNearest(i, neighbors)
        print("resultados: ", results, "\n")
        print("vecinos: ", neighbours, "\n")
        print("distancia: ", dist)

        return results

class FileHelper:

    def __init__(self, folder):
        self.folder = folder

    def getFiles(self, path):
        """
        retorna un diccionario de todos los archivos de imagen como nombre -> path
        y retorna tambien el numero total de archivos

        """
        imlist = {}
        count = 0
        for each in glob.glob(path + "/"+self.folder+"/*"):
            word = each.split("/")[-1]
            print ("Leyendo carpeta de entrenamiento ", word)
            imlist[word] = []
            print(path + "/" + word + "/*")
            for imagefile in glob.glob(path+"/"+word+"/*"):
                print ("Leyendo imagen: ", imagefile)
                im = cv2.imread(imagefile, 0)
                imlist[word].append(im)
                count +=1

        return [imlist, count]

    def getTestFiles(self, path):
        """
        retorna un diccionario de todos los archivos de imagen como nombre -> path
        y retorna tambien el numero total de archivos

        """
        imlist = {}
        count = 0
        word = self.folder
        print("Leyendo carpeta de test: ", word)
        imlist[word] = []
        for imagefile in glob.glob(path + "/"+self.folder+"/*"):
            print("Leyendo imagen: ", imagefile)
            im = cv2.imread(imagefile, 0)
            imlist[word].append(im)
            count += 1

        return [imlist, count]

class ImageHelper:
    def __init__(self):
        self.sift_object = cv2.xfeatures2d.SIFT_create()

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]
