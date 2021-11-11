import idx2numpy
import numpy
import matplotlib.pyplot as plot
from PIL import Image
import matplotlib.pyplot as plot


class Perceptron:

    def __init__(self):
        self.w = []
        self.images = []
        self.accuracy = []
        self.DIGITS = 10
        self.SIZE = 28
        self.n = 0.005

    def loadDataset(self):
        train = idx2numpy.convert_from_file("data/train-images.idx3-ubyte")
        train_labels = idx2numpy.convert_from_file("data/train-labels.idx1-ubyte")

        test = idx2numpy.convert_from_file("data/t10k-images.idx3-ubyte")
        test_labels = idx2numpy.convert_from_file("data/t10k-labels.idx1-ubyte")
        return train, train_labels, test, test_labels

    def showDigit(self, n):
        plot.imshow(train[n], cmap=plot.cm.binary)
        plot.show()
        print("El numero es " + str(train_labels[n]))

    def inicializarWeights(self, train):
        return numpy.zeros(train.shape[1] * train.shape[1])

    def obtenerNumero(self, arr):
        for i in range(self.DIGITS):
            if arr[i] == 1:
                return i
        return -1

    def saveImage(self, n, size):
        image = Image.fromarray(self.w[n].reshape(self.SIZE, self.SIZE))
        image = image.convert("RGB").resize((size, size))
        self.images.append(image)

    def saveImages(self, n, duration):
        self.images[0].save("gif/" + str(n) + ".gif", format="GIF", save_all=True, append_images=self.images,
                            duration=duration, loop=0)

    def test(self, x_test, y_test):
        good = 0

        for x_i, y_i in zip(x_test, y_test):
            y = self.predict(self.w, x_i)
            y_pred = self.obtenerNumero(y)

            if y_pred == y_i:
                good = good + 1

        good *= 100 / x_test.shape[0]

        return good

    def plot(self):
        plot.plot(self.accuracy, range(len(self.accuracy)), "ro-")
        plot.title("Accuracy vs epochs")
        plot.xlabel("Accuracy (%)")
        plot.ylabel("Epochs")
        plot.show()

    def softmax(self, n):
        e = numpy.exp(n - n.max())
        return e / numpy.sum(e, axis=0)

    def derivadaSigmoid(self, n):
        return n * (1 - n)

    def sigmoid(self, n):
        return 1 / (1 + numpy.exp(-n))

    # Funcion de Activacion
    def predict(self, w, x, tipo="softmax"):
        y_aux = []
        n = numpy.dot(w, x.reshape(self.SIZE * self.SIZE))

        if tipo == "softmax":
            for i in range(self.DIGITS):
                y_aux.append(self.sigmoid(n[i]))
        else:
            for i in range(self.DIGITS):
                if n[i] >= 0:
                    y_aux.append(1)
                else:
                    y_aux.append(0)

        return y_aux

    def perceptron(self, x_train, y_train, epocas, save_image=False, n=0, tipo=""):
        self.w = []

        for i in range(self.DIGITS):
            self.w.append(self.inicializarWeights(x_train))

        for i in range(epocas):
            accuracy_aux = 0

            for x_i, y_i, l in zip(x_train, y_train, range(len(x_train))):
                y_pred = self.predict(self.w, x_i, "softmax")

                for j in range(1):
                    if tipo == "softmax":
                        z = numpy.dot(self.w, x_i.reshape(self.SIZE * self.SIZE))
                        y_i_arr = []
                        for k in range(self.DIGITS):
                            if k == y_i:
                                y_i_arr.append(1)
                            else:
                                y_i_arr.append(0)
                        delta_y = self.derivadaSigmoid(z) * (numpy.array(y_pred) - numpy.array(y_i_arr))
                        self.w -= self.n * numpy.dot(x_i.reshape(self.SIZE * self.SIZE),  numpy.resize(delta_y, (10, 1)))
                    else:
                        if y_pred[j] == 1 and y_i != j:
                            self.w[j] = self.w[j] - numpy.reshape(x_i, self.SIZE * self.SIZE)
                        elif y_pred[j] == 0 and y_i == j:
                            self.w[j] = self.w[j] + numpy.reshape(x_i, self.SIZE * self.SIZE)
                        elif y_pred[j] == 1 and y_i == j:
                            accuracy_aux = accuracy_aux + 1

                if l % 500 == 0 and (i == 0 or i == epocas) and save_image:
                    self.saveImage(n, size=500)

            accuracy_aux *= 100 / x_train.shape[0]
            self.accuracy.append(accuracy_aux)

        if save_image:
            self.saveImages(n, duration=5)

        return self.w


if __name__ == "__main__":
    epocas = 50

    perceptron = Perceptron()
    train, train_labels, test, test_labels = perceptron.loadDataset()

    # showDigit(0)
    w = perceptron.perceptron(train, train_labels, epocas=1, tipo="softmax", save_image=False, n=7)

    #for i in range(epocas):
    #y = perceptron.predict(w, train[i])
    #y_pred = perceptron.obtenerNumero(y)
    #print("Para i=" + str(i) + ", el nº es " + str(train_labels[i]) + " y el nº predicho es " + str(y_pred))

    good = perceptron.test(test, test_labels)
    print("Accurary: " + str(good) + "%")
    perceptron.plot()
