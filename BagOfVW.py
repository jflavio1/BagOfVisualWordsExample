import numpy as np
import cv2
from sklearn.svm import SVC
from Helpers import FileHelper
from Helpers import BOVHelper
from Helpers import ImageHelper

from matplotlib import pyplot as plt

class BagOfVW:
    def __init__(self, numClusters, folder):
        self.img_path = "img/"
        self.numClusters = numClusters
        self.file_helper = FileHelper(folder)
        self.img_helper = ImageHelper()
        self.algorithmHelper = BOVHelper(numClusters)
        self.name_dict = {}
        self.descList = []
        self.clf = SVC()

        self.train_labels = np.array([])

    def trainModel(self, trainPath):

        # leyendo imagenes
        self.images, self.numImages = self.file_helper.getFiles(trainPath)

        # extrayendo caracteristicas de cada imagen usando SIFT
        label_count = 0
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print("Obteniendo caracteristicas para ", word)
            for im in imlist:
                #cv2.imshow("im", im)
                #cv2.waitKey()
                self.train_labels = np.append(self.train_labels, label_count)
                keypoints, descriptors = self.img_helper.features(im)
                self.descList.append(np.array(descriptors))

            label_count += 1

        bov_descriptor_stack = self.algorithmHelper.formatND(self.descList)

        self.algorithmHelper.cluster()

        # creando histograma
        self.algorithmHelper.developVocabulary(n_images = self.numImages, descriptor_list=self.descList)

        # normalizacion de caracteristicas
        self.algorithmHelper.standardize()

        # a entrenar el modelo
        self.algorithmHelper.trainSVC(self.train_labels)

    def recognize(self, test_img):

        """
        Metodo para reconocer cada imagen
        """

        kp, des = self.img_helper.features(test_img)

        # generamos un vocabulario para las imagenes
        vocab = np.array([0 for i in range(self.numClusters)])

        # retorna los clusters mas cercanos con las caracteristicas parecidas
        test_ret = self.algorithmHelper.kmeans_obj.predict(des)

        for each in test_ret:
            vocab[each] += 1

        # obtenemos caracteristicas
        vocab = vocab.reshape(1,-1)
        vocab = self.algorithmHelper.scale.transform(vocab)

        # predecimos la clase de la imagen
        lb = self.algorithmHelper.supportVectorMachine.predict(vocab)
        print ("La imagen es de la clase: ", self.name_dict[str(int(lb[0]))])
        #print ("Pertenece a clase entrenada: " + str(lb))
        return lb

    def testModel(self, testPath):
        """
        Para testear el clasificador
        Obtendra todas las imagenes de la carpeta de testing elegida en el metodo init()
        predict() obtiene la clase a la que pertenece cada imagen y la muestra en una interfaz grafica

        """

        self.testImages, self.testImageCount = self.file_helper.getTestFiles(testPath)

        predictions = []

        for word, imlist in self.testImages.items():
            # print("procesando ", word)
            for im in imlist:
                #se intento K nn pero no se logro, puedo ver el codigo del metodo en Helpers.py
                #cl = self.algorithmHelper.classifyKnn(im)
                cl = self.recognize(im)
                predictions.append({
                    'imagen': im,
                    'clase': cl,
                    'nombre': self.name_dict[str(int(cl[0]))]
                })

        #print("Se testearon " + str(len(predictions))+" imagenes")
        for each in predictions:
            plt.imshow(cv2.cvtColor(each['imagen'], cv2.COLOR_GRAY2RGB))
            plt.title(each['nombre'])
            plt.show()