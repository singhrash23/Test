from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


def createModel(totalPlayers):
    cp =[]
    for i in range(totalPlayers):
        model = Sequential()
        model.add(Dense(input_dim=3,units=7))
        model.add(Activation("sigmoid"))
        model.add(Dense(units=1))
        model.add(Activation("sigmoid"))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='mse',optimizer=sgd, metrics=['accuracy'])

        cp.append(model)
    return cp    