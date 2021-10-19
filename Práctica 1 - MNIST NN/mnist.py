
import idx2numpy
import numpy
import matplotlib.pyplot as plot


def loadDataset():
    train = idx2numpy.convert_from_file("data/train-images.idx3-ubyte")
    train_labels = idx2numpy.convert_from_file("data/train-labels.idx1-ubyte")

    test = idx2numpy.convert_from_file("data/t10k-images.idx3-ubyte")
    test_labels = idx2numpy.convert_from_file("data/t10k-labels.idx1-ubyte")
    return train, train_labels, test, test_labels


def digito(n):
    plot.imshow(train[n], cmap=plot.cm.binary)
    plot.show()
    print("El numero es " + str(train_labels[n]))

def inicializarWeights(train):
    ws = []
    #for i in range(10):
    #    ws.append(numpy.zeros(numpy.shape(train)[1]*numpy.shape(train)[1]))
    return numpy.zeros(train.shape[1]*train.shape[1])

def predecir(w, x):
    res = numpy.inner(w, x.reshape(28*28))
    return funcionActivacion(res)

def funcionActivacion(pred):
    y_aux = []
    #print(pred)
    for i in range(10):
        if pred >= 0:
            y_aux.append(1)
        else:
           y_aux.append(0)
    return y_aux

def perceptron(train, labels, epocas):
    w = inicializarWeights(train)
    ws = []

    for i in range(epocas):
        for x, y in zip(train, labels):
            y_pred = predecir(w, x)
            for j in range(10):
                if y_pred[j] == 1 and y == j:
                    w = w + numpy.reshape(x, 28*28)
                else:
                    w = w - numpy.reshape(x, 28*28)
        ws.append(w)

    return ws

if __name__=="__main__":
    train, train_labels, test, test_labels = loadDataset()

    #digito(0)
    w = perceptron(train, train_labels, 10)
    y = predecir(w, train[0])
    res = funcionActivacion(y)
    print(res)

