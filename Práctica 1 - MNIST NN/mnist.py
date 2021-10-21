
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
    return numpy.zeros(train.shape[1]*train.shape[1])

def obtenerNumero(arr):
    for i in range(10):
        if arr[i] == 1:
            return i
    return -1

def predecir(w, x):
    res = funcionActivacion(numpy.dot(w, x.reshape(28*28)))
    return res

def funcionActivacion(pred):
    y_aux = []
    for i in range(10):
        if pred[i] >= 0:
            y_aux.append(1)
        else:
           y_aux.append(0)

    return y_aux

def perceptron(train, labels, epocas):
    w_aux = []

    for i in range(10):
        w_aux.append(inicializarWeights(train))

    for i in range(epocas):
        for x_i, y_i in zip(train, labels):
            y_pred = predecir(w_aux, x_i)
            for j in range(10):
                if y_pred[j] == 1 and y_i != j:
                    w_aux[j] = w_aux[j] - numpy.reshape(x_i, 28 * 28)
                elif y_pred[j] == 0 and y_i == j:
                    w_aux[j] = w_aux[j] + numpy.reshape(x_i, 28 * 28)

    return w_aux

if __name__=="__main__":
    train, train_labels, test, test_labels = loadDataset()

    #digito(0)
    n = 1
    w = perceptron(train, train_labels, 1)

    for i in range(50):
        y = predecir(w, train[i])
        y_n = obtenerNumero(y)
        print("El nº es " + str(train_labels[i]) + " y el nº predicho es " + str(y_n))
