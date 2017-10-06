from keras.models import Sequential
from keras.layers import Dense
import numpy

#fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

N=10 #Size of dataset N-dim list of 10-dim lists



X = numpy.random.choice([0, 1], size=(N,10)) #generate matrix of size (N,10) with random entries of the list [0,1]

#CREATE MODEL
#-------------------------------------------------------------------------------
model = Sequential()
model.add(Dense(10, input_dim=10, init='uniform', activation='sigmoid'))
model.add(Dense(3, init='uniform', activation='sigmoid'))
model.add(Dense(10, init='uniform', activation='sigmoid'))

"""
-Sequential() is for a model where Layer comes after layer
 Sequential() opens a empty scratch, where we can add layers
-Dense() gives a fully connected layer
 First entry (here 10) is the output dimension of the layer. aka we have 12 neurons in the first layer
-So the first layer has 10 neurons and expects 10 input variables
-init='uniform' initializes weights from uniform distribution (default value between 0 and 0.05)
 'normal' would be from Gaussian distribution
-activation defines the function that defines the output of the neuron
 'relu' is a rectifier fct f(x) = max(0,x)
 'sigmoid' sigmoid fct for binary outputs
"""
#-------------------------------------------------------------------------------
#COMPILE MODEL
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

"""
Define loss and cost function:
'binary_crossentropy' suggested for binary values
'adam' see adam paper

metrics:
Not entirely clear why this.
Classification problems (spam / not spam, ill / not ill, 0 / 1) are supposed to use this
"""
#--------------------------------------------------------------------------------
#FIT MODEL
model.fit(X, X, nb_epoch=2000, batch_size=10)

#-------------------------------------------------------------------------------
#EVALUATE MODEL
scores = model.evaluate(X, X)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
