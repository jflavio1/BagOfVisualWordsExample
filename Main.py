__author__ = "Jose Flavio Quispe Irrazabal"
__copyright__ = "Copyright 2017, Laboratorio 4"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "jflavio90@gmail.com"
__status__ = "Production"

from BagOfVW import BagOfVW

def init():
    bov = BagOfVW(4, "t2")
    bov.trainModel("img/training")
    bov.testModel("img/test")

if __name__ == '__main__':
    init()
