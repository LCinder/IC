import idx2numpy
import numpy
import matplotlib.pyplot as plot
from PIL import Image
import matplotlib.pyplot as plot
from keras import Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, SimpleRNN
from keras.utils.np_utils import to_categorical
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Activation
from numba import cuda
import seaborn
from sklearn.metrics import confusion_matrix


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
        plot.imshow(x_train[n], cmap=plot.cm.binary)
        plot.show()
        print("El numero es " + str(y_train[n]))

    def inicializarWeights(self, train):
        return numpy.random.rand(train.shape[1] * train.shape[1]) #numpy.zeros(train.shape[1] * train.shape[1])

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
            self.w.append(self.inicializarWeights(x_train)) #inicializar pesos aleatoriamente, y normalizar!!!!

        #Normalizar
        for l in range(len(self.w)):
            self.w[l] /= 255

        for i in range(epocas):
            accuracy_aux = 0

            for x_i, y_i, l in zip(x_train, y_train, range(len(x_train))):
                y_pred = self.predict(self.w, x_i, "softmax") # aplicar sigmoide mejor

                if tipo == "softmax":
                    z = numpy.dot(self.w, x_i.reshape(self.SIZE * self.SIZE))
                    y_i_arr = []
                    for k in range(self.DIGITS):
                        if k == y_i:
                            y_i_arr.append(1)
                        else:
                            y_i_arr.append(0) #derviada para propagar hacia atras y  funcion lineal para propagar hacia adelante
                    delta_y = self.sigmoid(z) * (numpy.array(y_pred) - numpy.array(y_i_arr))
                    self.w -= self.n * numpy.resize(numpy.outer(x_i.reshape(self.SIZE * self.SIZE),  numpy.resize(delta_y, (1, 10))), (10, 784))
                else:
                    for j in range(self.DIGITS):
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


class MultiLayer:
    def __init__(self, x, y, hidden, epochs, lr):
        self.x = x
        self.y = y
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.w = [numpy.random.rand(784) for i in range(10)]

    def fit(self, x, y):
        for i in range(self.epochs):
            hidden_input = numpy.dot(x, self.w)
            hidden_output = self.hidden_activation(hidden_input)
            output_input = numpy.dot(hidden_output, numpy.random())
            y_pred = self.output_activation(output_input)

            grad = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_input)
            grad_v = hidden_output.dot(grad)


def plot_results(y_pred, y_test):
    seaborn.heatmap(confusion_matrix(numpy.argmax(y_test, 1), numpy.argmax(y_pred, 1)), annot=True)
    plot.xlabel("Valores verdaderos")
    plot.xlabel("Valores predichos")
    plot.show()


def NN(x_train, y_train, x_test, y_test, type):

    if type == "cnn":
        model = Sequential([
            Input(shape=(28, 28, 1)),
            # Red neuronal convolutiva con mascara 3x3
            # Capa 1
            Conv2D(32, activation="relu", kernel_size=(3, 3), padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            # Capa 2
            Conv2D(64, activation="relu", kernel_size=(3, 3), padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            # Capa 3
            Conv2D(128, activation="relu", kernel_size=(3, 3), padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            # Serializa(tranforma) un tensor(array)
            Flatten(),
            Dropout(0.2),
            # Capa 6
            Dense(10, activation="softmax")
        ])
    elif type == "deep":
        model = Sequential([
            Input(shape=(28, 28, 1)),
            Flatten(),
            Dense(40, activation="relu", input_dim=784),
            # Para evitar el sobreajuste se eliminan nodos aleatoriamente
            Dropout(0.2),
            Dense(80, activation="relu", input_dim=784),
            # Para evitar el sobreajuste se eliminan nodos aleatoriamente
            Dropout(0.2),
            MaxPooling2D(pool_size=(2, 2)),
            Dense(500, activation="relu", input_dim=784),
            # Para evitar el sobreajuste se eliminan nodos aleatoriamente
            Dropout(0.2),
            Dense(1000, activation="relu", input_dim=784),
            # Para evitar el sobreajuste se eliminan nodos aleatoriamente
            Dropout(0.2),
            Dense(2000, activation="relu", input_dim=784),
            # Para evitar el sobreajuste se eliminan nodos aleatoriamente
            Dropout(0.2),
            Dense(10, activation="softmax")
        ])
    elif type == "rnn":
        model = Sequential([
            Input(shape=(28, 28)),
            SimpleRNN(1000, activation="relu", input_shape=(28, 28)),
            Dropout(0.2),
            Dense(10, activation="softmax")
        ])

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    y_train = to_categorical(y_train, 10)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_test = to_categorical(y_test, 10)

    #Normalizacion
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    #sparse_categorical_crossentropy
    model.compile(optimizer=Adam(learning_rate=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1, use_multiprocessing=True, workers=16)
    accuracy = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    plot_results(y_pred, y_test)

    print("Accuracy: " + str(round(accuracy[1], 3)))


if __name__ == "__main__":
    epocas = 5

    perceptron = Perceptron()
    x_train, y_train, x_test, y_test = perceptron.loadDataset()

    # showDigit(0)
    #for i in range(epocas):
    #    y = perceptron.predict(w, x_train[i])
    #    y_pred = perceptron.obtenerNumero(y)
    #    print("Para i=" + str(i) + ", el nº es " + str(y_train[i]) + " y el nº predicho es " + str(y_pred))
    #perceptron.plot()

    #w = perceptron.perceptron(x_train, y_train, epocas=100, tipo="otro", save_image=False, n=7)
    #good = perceptron.test(x_test, y_test)

    #print("Accuracy Train: " + str(perceptron.accuracy[len(perceptron.accuracy)-1]))
    #print("Accurary: " + str(good) + "%")
    print(cuda.gpus)
    NN(x_train, y_train, x_test, y_test, "cnn")
