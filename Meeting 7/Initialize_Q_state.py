import numpy as np
import qutip as qt
import timeit

"""
This file creates the K matrix from any given dataset vector [a,b,c,...] and creates the quantum state for a given label vector 
[1,-1,-1,1,...]
The idea will be to encode the images into a 4-dim vector and apply the Q-SVM from the paper from week6.
"""

class Oracle(): #construct quantum state for data and label
    def __init__(self, data ,label, **kwargs):
        
        self.M = len(data)            #number of training data
        self.N = len(data[0])         #length of training vectors
        if len(data) != len(label): 
            exit('Error: not same number of data and labels!')
        self.norms = []
        self.qstates = []
        for i in data:
            i = np.array(i)
            norm = np.linalg.norm(i)
            self.state = 1/norm*sum([i[k]*qt.fock(self.N, k) for k in range(0,self.N)])
            self.qstates.append(self.state.unit())      #save quantum states
            self.norms.append(norm)                     #save classical norms
        
        self.qlabels = 1/(np.linalg.norm(label))*sum([label[i]*qt.fock(self.M,i) for i in range(0,self.M)])
        #make qstate for label vector
        
    
def kernel(norm,qstat,m, n): #m is number of datasets, n the length of data vector
    chi = 1/(np.linalg.norm(norm))*sum([norm[i]*qt.tensor(qt.fock(m,i),qstat[i]) for i in range(0,m)])
    #---------------------------------------------------
    #Try partial trace manually.

    def operator(k):
        return qt.tensor(qt.qeye(m),qt.fock(n,k))
    chidens = chi*chi.dag()
    
    trace = sum([operator(i).dag()*chidens*operator(i) for i in range(0,n)])
    #-------------------------------------------------------------------
    # Qutip buildt in function for ptrace
    trace2 = chidens.ptrace(0)  #all other components than the 0-th are traced out!
    
    return trace2
    
    
def matrix_construct(matrix, m, gamma): #constructs the K matrix to the right shape. m is the number of training sets, gamma is the parameter from the paper
    J = np.zeros((m+1,m+1), dtype = complex)
    one = np.ones(m)
    onetransp = np.transpose(one)
    onetransp.shape=(m,1)
    for i in range(1,m+1):
        J[0][i] = 1
    for i in range(1,m+1):
        J[i][0] = 1
    K = np.zeros((m+1,m+1), dtype = complex)
    gam_matrix = np.zeros((m+1,m+1), dtype = complex)    
    for i in range(1,m+1):
        for j in range(1,m+1):
            K[i][j] = matrix[i-1][0][j-1]
        gam_matrix[i][i] = 1/gamma
    
    return  J, K, gam_matrix
    
    
    
        
#--------------------------------------------------------------------------
training_set = [[1,0.3,0.85,1],[1,1,1,0],[0,1,1,1]]  #fictive training set
label_set = [-1,1,1]                            #fictive training labels
a = Oracle(training_set,label_set)
K = kernel(a.norms,a.qstates,a.M, a.N)
j, k , gamma = matrix_construct(K, a.M, 2)
w, v = np.linalg.eig(k)
print(v)

    

