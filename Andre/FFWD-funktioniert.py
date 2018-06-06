import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

# Seed der Zufallszahlen um Ergebnis reproduzieren zu können
np.random.seed(1)

# Iris Datenset einlesen
iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
iris = np.array(iris)

np.random.shuffle(iris)

# Features aus Datenmatrix und dann normieren
features = iris[:, :-1]
features = (features - features.mean()) / features.std()

# Vektor von Einsen an feature Matrix rechts anhängen
biasPad = np.ones((features.shape[0], 1), dtype=features.dtype)
features = np.concatenate((features, biasPad), axis=1)

# One Hot Encodierung der Label
label = np.array(iris[:, -1], dtype=int).reshape(-1)
label = np.eye(3)[label]

# Datenset aufsplitten in Training und Test Daten
XTrain, XTest, YTrain, YTest = train_test_split(features, label, shuffle=True, test_size=0.8)

# Neuronen definieren
# Input Neuronen = Anzahl von Features je Label
inputCount = features.shape[1]

# Basis
hiddenCount = inputCount+2

# 3 Klassen => 3 OutputNeuronen
outputCount = 3

# Aktivierungsstatus für alle Layer initialisieren
ai = np.ones((inputCount, 1))
ah = np.ones((hiddenCount, 1))
ao = np.ones((outputCount, 1))

# Neuronen Gewichtung
# Zufällige Initialgewichtungen für jede Layerverbindung zwischen 0-1 multipliziert mit Wurzel(2/inputNeuronen)
wih = np.random.rand(inputCount, hiddenCount) * np.sqrt(2. / inputCount)
# Analog:
who = np.random.rand(hiddenCount, outputCount) * np.sqrt(2. / hiddenCount)

# Update Arrays for momentum updates ??
cih = np.zeros((inputCount, hiddenCount))
cho = np.zeros((hiddenCount, outputCount))


# Forwärtslauf des Netzwerks
def feedFwd(featureMat, act_fun):
    global ai, ah, ao, wih, who
    # InputAktivierungsstatus auf InputFeatures
    ai = featureMat

    # Skalarprodukt des Aktivierungsstatus mal Gewichtungsparameter
    ah = np.dot(ai, wih)
    # Aktivierungsstatus der HiddenLayer
    # if act_fun == 'tanh':
    #     ah = np.tanh(ah)
    # else:
    #     ah = sigmoid(ah, beta_sig)

    ah = activation(ah, act_fun, beta_sig)
    # Skalarprodukt des Aktivierungsstatus mal Gewichtungsparameter
    ao = np.dot(ah, who)
    # Aktivierungsstatus der OutputLayer
    # if act_fun == 'sigmoid':
    #     ao = sigmoid(ao, beta_sig)
    # else:
    #     ao = np.tanh(ao)
    ao = activation(ao, act_fun, beta_sig)
    # ao = 1.0/1.0+np.exp(-1*ao)
    return ao


def activation(x, func, beta):
    inp = np.copy(x)
    if func == 'tanh':
        return np.tan(inp)
    elif func == 'sigmoid':
        return sigmoid(inp, beta)
    elif func == 'softmax':
        return np.log10(1+np.exp(inp))
    else:
        sm = inp < 0
        inp[sm] = 0
        return inp


# Backpropagation
def backProp(X, label, output, N, batchSize=1, beta=0.0009):
    # X -> Input
    # output -> Ergebnis
    # N -> Lern Rate
    # BatchSize -> Wie viele Werte werden auf einmal ins Netzwerk eingespeist
    # beta -> Bias Modifikator?

    # Initialisierung der Aktivierungsstatusse, Gewichtungen und Bias
    global ai, ah, ao, wih, who, cih, cho

    Cost = 0.5 * np.abs(output - label)**2
    # Fehler = Ableitung der mean squared error function
    delOut = output - label

    # Output der HiddenLayer multipliziert mit dem Fehler
    dwho = np.dot(ah.T, delOut) / batchSize

    # Fehler multipliziert mit der Gewichtung der OutputLayer
    delHidden = np.dot(delOut, who.T) * (1.0 - ah ** 2)

    # Delta wih berechnen
    dwih = np.dot(X.T, delHidden) / batchSize

    # Gewichtung who anpassen
    who -= N * dwho + beta * cho
    cho[:] = dwho

    # Gewichtung wih anpassen
    wih -= N * dwih + beta * cih
    cih[:] = dwih


# Funktion zum trainieren
# X -> training Data
# Y -> training Targets
# learningRate ->
def train(X, Y, iteration=1000, act_fun = 'tanh', learningRate=0.001, batchSize=1, beta=0.099, decayRate=0.0005):
    errorTimeline = []
    epochList = []

    # train it for iteration number of epoch
    for epoch in range(iteration):

        # for each mini batch
        for i in range(0, X.shape[0], batchSize):
            # split the dataset into mini batches
            batchSplit = min(i + batchSize, X.shape[0])
            XminiBatch = X[i:batchSplit, :]
            YminiBatch = Y[i:batchSplit, :]

            # calculate a forwasd pass through the network
            output = feedFwd(XminiBatch, act_fun)

            # calculate mean squared error
            error = 0.5 * np.sum((YminiBatch - output) ** 2) / batchSize
            # print error

            # backprop and update weights
            backProp(XminiBatch, YminiBatch, output, learningRate, batchSize, beta)

        # after every 50 iteration decrease momentum and learning rate
        # decreasing momentum helps reduce the chances of overshooting a convergence point
        if epoch % 50 == 0 and epoch > 0:
            learningRate *= 1. / (1. + (decayRate * epoch))
            beta *= 1. / (1. + (decayRate * epoch))
            # Store error for ploting graph
            errorTimeline.append(error)
            epochList.append(epoch)
            print('Epoch :', epoch, ', Error :', error, ', alpha :', learningRate)

    return errorTimeline, epochList

def sigmoid(x, beta_sig = 1):
    return (1 / (1 + np.exp(-beta_sig*x))) - 0.5

# Work it, make it, do it,
# Makes us harder, better, faster, stronger!
learningRate = 0.01
beta = 0.099
beta_sig = 3

errorTimeline, epochList_relu = train(XTrain, YTrain, 2000, 'relu', learningRate, features.shape[0], beta)

predOutput_relu = feedFwd(XTest, 'relu')

errorTimeline_sig, epochList_sig = train(XTrain, YTrain, 2000, 'sigmoid', learningRate, features.shape[0], beta)

predOutput_sig = feedFwd(XTest, 'sigmoid')

errorTimeline_soft, epochList_soft = train(XTrain, YTrain, 2000, 'softmax', learningRate, features.shape[0], beta)

predOutput_soft = feedFwd(XTest, 'softmax')

errorTimeline_tanh, epochList_tanh = train(XTrain, YTrain, 2000, 'tanh', learningRate, features.shape[0], beta)

predOutput_tanh = feedFwd(XTest, 'tanh')
# vectorised count compare the indices of output and labels along rows
# add to count if they are same
count_relu = np.sum(np.argmax(predOutput_relu, axis=1) == np.argmax(YTest, axis=1))
count_sig = np.sum(np.argmax(predOutput_sig, axis=1) == np.argmax(YTest, axis=1))
count_tanh = np.sum(np.argmax(predOutput_tanh, axis=1) == np.argmax(YTest, axis=1))
count_soft = np.sum(np.argmax(predOutput_soft, axis=1) == np.argmax(YTest, axis=1))

# print accuracy
perc_relu = (float(count_relu) / float(YTest.shape[0]))*100
perc_sigmoid = (float(count_sig) / float(YTest.shape[0]))*100
perc_soft = (float(count_soft) / float(YTest.shape[0]))*100
perc_tanh = (float(count_tanh) / float(YTest.shape[0]))*100

print('Relu-Accuracy : ', round(perc_relu,1), '%')
print('Sigmoid-Accuracy : ', round(perc_sigmoid,1), '%')
print('Softmax-Accuracy : ', round(perc_soft,1), '%')
print('Tanh-Accuracy : ', round(perc_tanh,1), '%')

# plot graph
x1 = np.linspace(-5,5,100)
y1 = sigmoid(x1)
y2 = np.tanh(x1)
y3 = sigmoid(x1, 2)
fig1 = plt.figure()
# ax1 = fig1.add_subplot(211)
# ax1.plot(x,y1, label='sigmoid')
# ax1.legend()
ax2 = fig1.add_subplot(111)
ax2.plot(x1,y3, label=('tanh - ' + str(round(perc_tanh,1)) + '%'))
ax2.plot(x1,y2, label=('sigmoid_beta - ' + str(round(perc_sigmoid,1)) + '%'))
ax2.plot(x1,activation(x1,'softmax',0), label=('softmax - ' + str(round(perc_soft,1)) + '%'))
ax2.plot(x1,activation(x1,'relu',0), label=('relu - ' + str(round(perc_relu,1)) + '%'))
ax2.legend()
ax2.grid()
