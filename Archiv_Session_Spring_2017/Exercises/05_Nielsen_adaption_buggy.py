"""This example is mostly taken from Nielsens book, Neural Networks and Deep Learning
http://neuralnetworksanddeeplearning.com/chap1.html
The aim was to build a Neural Network from the scratch, without Keras.
The algorithm still contains a bug, such that the certainity to find Einstein decreases, instead of increases"""


import numpy as np
import random
import tools

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
    
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):

    def __init__(self, sizes):          #sizes is argument of Network()
        self.num_layers = len(sizes)    #how many layers does list contain
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #sizes[1:] gives the elements of sizes from index 1 to the end
        #randn(a,b) generates an array of size a times b with random numbers
        #therefore biases is a list of (Nx1)-arrays containing the biases for all the nodes of a N-dim layer, except for the first layer
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #zip(a,b) gives a list of pairs of the items in a,b (a_i,b_i) 
        #therefore weights is a list of matrices, that contain the weights. E.g. for sizes = [2,3,1], there is a 2x3 and a 3x1 matrix
        #so matrixelement w_jk connects neuron j from the n+1 layer with the k neuron from the n-th layer.




    def feedforward(self, a):                       #input a has to be same dimension as 1st layer
        """Return the output of the network if "a" is input. Propagates the input through the network"""
        for b, w in zip(self.biases, self.weights): #b, w are the full weight and bias vectors, that contain all parameters
            a = sigmoid(np.dot(w, a)+b)        #dot(a,b) for a = n x m and b = m x l gives n x l
        return a
        #if we define Network([a,b,c,d]) then w contains a x b matrix, b x c matrix and c x d matrix  
        
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): #stochastic gradient descent
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)   #is there testdata?
        n = len(training_data)                  #check length of training data
        for j in range(epochs):                 #repeat for every epoch
            random.shuffle(training_data)       #shuffle training data to choose mini batch
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) #update mini batch
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
                #evaluate is fct in this class, that checks, how often the network predicted
                #the outcome exactly. 
            else:
                print("Epoch {0} complete".format(j))
                
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        for (x,y) in test_data:
            print(x,y,self.feedforward(x))
        return sum(int(x == y) for (x, y) in test_results)
        #e.g. for image recognition, we have the pixels and a variable that says what the picture contains
        #(e.g. x = 'Dog'). This function returns the sum of succesful recognitions, where the output y is
        #identical to the input x.

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

        
training_set, training_labels= tools.load_images("images/train/")
test_set, test_labels = tools.load_images("images/test/")

resize_set, resize_labels= tools.prep_datas(training_set, training_labels)
resize_test_set, resize_test_labels= tools.prep_datas(test_set, test_labels)

training_data = list(zip(resize_set,resize_labels))

test_data = list(zip(resize_test_set,resize_test_labels))

net = Network([1024, 200, 1]) #output is only yes or no. Its marie or albert or not.
net.SGD(training_data, 10000, 25, 0.1, test_data=test_data) #training data, epochs, mini_batch_size, eta ,test data
