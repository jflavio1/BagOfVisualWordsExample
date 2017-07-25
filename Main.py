from BagOfVW import BagOfVW

def init():
    bov = BagOfVW(4, "t2")
    bov.trainModel("img/training")
    bov.testModel("img/test")

if __name__ == '__main__':
    init()