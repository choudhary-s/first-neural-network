#build it
#test it
#train it

#a three layer neural network

#numpy - a library for scientific computing in python
import numpy as np 

def sigmoid(x, deriv):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input data
i_p = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

#output data
o_p = np.array([[0],
                [1],
                [1],
                [0]])

np.random.seed(1)

#synapses weights 
#1 is the bias
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
#print(syn0)

#training step
for i in range(60000):
    #forward propagation
    l0 = i_p
    l1 = sigmoid(np.dot(l0, syn0), False)
    l2 = sigmoid(np.dot(l1, syn1), False)
    #backpropagation
    l2_error = o_p - l2
    
    if(i%10000 == 0):
        print("Error: %s"%str(np.mean(np.abs(l2_error))))
    l2_delta = l2_error * sigmoid(l2,deriv=True)
    
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid(l1,deriv=True)
    
    #update weight using gradient descent
    syn1+=l1.T.dot(l2_delta)
    syn0+=l0.T.dot(l1_delta)
    
print("Output after training")
print (l2)