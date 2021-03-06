
# coding: utf-8

# In[61]:

import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1],
             [1,1,0]])
                
y = np.array([[0],
			[1],
			[1],
			[0],
             [1]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(5000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    #print "l1:" + str(l1)
    #print "l2:" + str(l2)

    # how much did we miss the target value?
    l2_error = y - l2
    #print "l2error:" + str(l2_error)
    
    if (j% 1000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)
    #print "l2 delta:" + str(l2_delta)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    #print "l1 error:" + str(l1_error)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)
    #print "l1 delta:" + str(l1_delta)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    #print "syn1:" + str(syn1)
    #print "syn0:" + str(syn0)

print "Result:" + str(l2)


# In[2]:

import numpy as np

r = np.array([[1],[2]])
s = np.array([[1,3]])
t = r * s

#print t
#print t * t

u = np.array([[1],[2],[3],[4]])
v = np.array([[2, 3]])
#print u * v

listofarrays = []
listofarrays.append(r)
listofarrays.append(s)
listofarrays.append(t)
print listofarrays

for l in range(0,5):
    print l
print 'xxxxxxxxxx'
for l in range(5,1,-1):
    print l
    
print np.multiply([[1,2]], [[2,3]])

print np.random.random((3,4))


# In[5]:

import numpy as np
r = np.array([[-1],[2]])
s = np.array([[1,3]])
print r.dot(s)
print r*s
r[r < 0] = 0
print r


# In[1]:

# examples from CS231 - Andrej Karpathy

import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()


# In[2]:

# examples from CS231 - Andrej Karpathy

#Train a Linear Classifier

# initialize parameters randomly
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in xrange(200):
  
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b 
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  if i % 10 == 0:
    print "iteration %d: loss %f" % (i, loss)
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg*W # regularization gradient
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db


# In[16]:

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))


# In[4]:

# examples from CS231 - Andrej Karpathy

# initialize parameters randomly
h = 150 # size of hidden layer
W = 0.01 * np.random.randn(D,h) # weights input to hidden; D = 2 dimensions
b = np.zeros((1,h)) # bias input to hidden, zero-vector 1 line, h columns
W2 = 0.01 * np.random.randn(h,K) # weights hidden to output, K = 3 classes
b2 = np.zeros((1,K)) # bias hidden to output

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0] # size of array: data matrix
for i in xrange(5000):
  
  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
  scores = np.dot(hidden_layer, W2) + b2 # compute class scores: hidden to output
  
  # compute the class probabilities
  exp_scores = np.exp(scores) # get unnormalized probabilities
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K], normalize them for each example
  # now have an array probs of size [300 x 3], where each row now contains the class probabilities
  
  # compute the loss: average cross-entropy loss and regularization
  correct_logprobs = -np.log(probs[range(num_examples),y]) # correct_logprobs is a 1D array of just
  # the probabilities assigned to the correct classes for each example
  data_loss = np.sum(correct_logprobs)/num_examples # average cross-entropy loss and regularization
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 1000 == 0:
    print "iteration %d: loss %f" % (i, loss)
  
  # compute the gradient on scores and backpropagate
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W,b
  dW = np.dot(X.T, dhidden)
  db = np.sum(dhidden, axis=0, keepdims=True)
  
  # add regularization gradient contribution
  dW2 += reg * W2
  dW += reg * W
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  W2 += -step_size * dW2
  b2 += -step_size * db2


# In[14]:

# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))


# In[5]:

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
#fig.savefig('spiral_linear.png')


# In[58]:

import numpy as np
exp_tst = np.exp([[1,2,3,4,5]])
print exp_tst
probs = exp_tst / np.sum(exp_tst, axis=1, keepdims=True)
print probs
print probs[0,2]
print probs[range(1),[[3]]]


# In[60]:

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
print l2


# In[2]:

import numpy as np

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    if(abs(x).any>10):
        return np.sign(x)
    return 1/(1+np.exp(-x))

wij = np.array([[0.2,0.1],
             [0.1,0.3]])
                
y = np.array([[0.6]])

aj = np.array([[2], [1]])

wi = np.array([[1], [1]])

xi = wij * aj
x = wi * nonlin(xi, False)

E = 1/2 * (nonlin(x, False) - y)**2

print xi
print x

dEdw = (nonlin(x, False) - y)*nonlin(x, True)*wi*nonlin(xi, True)*aj

print E

alpha = 1

dwij = -alpha * dEdw

print dwij


# In[98]:

import numpy as np

# input a_k -- Wkj -- g(in_j) -  hidden a_j -- Wji -- g(in_i) - output a_i - y_i
# hidden to output layer
# Wji = Wji + alpha (yi-ai)g'(ini)aj ini=sum over j Wji aj
# input to hidden layer
# Wkj = Wkj + alpha sum over i (yi-ai) Wji g'(inj) ak inj = sum over k Wkj ak

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

alpha = 1
yi = np.array([[0.6], [0.2]])
ai = np.array([[0.3], [0.2]])
aj = np.array([[0.3], [-0.2]])
wji = np.array([[0.2,0.1],
             [0.1,0.3]]) 
ini = wji * aj
print wji + alpha * (yi-ai) * sigmoid(ini,True) * aj
ak = np.array([[0.4], [-0.1]])
wkj = np.array([[0.1,-0.1],
             [0.1,0.2]]) 
inj = wkj * ak
print wkj + alpha * (yi-ai) * wji * sigmoid(inj,True) * ak

yi = np.array([[0.5], [0.1]])
ai = np.array([[0.3], [0.2]])
aj = np.array([[0.3], [-0.2]]) 
ini = wji * aj
print wji + alpha * (yi-ai) * sigmoid(ini,True) * aj
ak = np.array([[0.4], [-0.1]]) 
inj = wkj * ak
print wkj + alpha * (yi-ai) * wji * sigmoid(inj,True) * ak

yi = np.array([[0.5], [0.1]])
ai = np.array([[0.3], [0.2]])
aj = np.array([[0.3], [-0.2]]) 
ini = wji * aj
print wji + alpha * (yi-ai) * sigmoid(ini,True) * aj
ak = np.array([[0.4], [-0.1]]) 
inj = wkj * ak
print wkj + alpha * (yi-ai) * wji * sigmoid(inj,True) * ak

yi = np.array([[0.5], [0.1]])
ai = np.array([[0.3], [0.2]])
aj = np.array([[0.3], [-0.2]]) 
ini = wji * aj
print wji + alpha * (-yi * np.log(ai)) * sigmoid(ini,True) * aj
ak = np.array([[0.4], [-0.1]]) 
inj = wkj * ak
print wkj + alpha * (-yi * np.log(ai)) * wji * sigmoid(inj,True) * ak


# In[13]:

import numpy as np

# input a_k -- Wkj -- g(in_j) -  hidden a_j -- Wji -- g(in_i) - output a_i - y_i
# hidden to output layer
# Wji = Wji + alpha (yi-ai)g'(ini)aj ini=sum over j Wji aj
# input to hidden layer
# Wkj = Wkj + alpha sum over i (yi-ai) Wji g'(inj) ak inj = sum over k Wkj ak

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    np.putmask(x, x > 100, 100) # limit array values of x (np.putmask(a, a >= m, m - 1))
    np.putmask(x, x < -100, -100)
    return 1/(1+np.exp(-x))

alpha = 1
# solid = 1111, vertical = 1010 or 0101, diagonal = 1001 or 0110, horizontal = 1100 or 0011
# bias as last element
solid = np.array([[1,1,1,1]]) # solid = 1111
vert0 = np.array([[1,0,1,0]]) # vertical = 1010
vert1 = np.array([[0,1,0,1]]) # vertical = 0101
diag0 = np.array([[1,0,0,1]]) # diagonal = 1001
diag1 = np.array([[0,1,1,0]]) # diagonal = 0110
hor0 = np.array([[1,1,0,0]]) # horizontal = 1100
hor1 = np.array([[0,0,1,1]]) # horizontal = 0011

# target outputs
tsol = np.array([[1,0,0,0]]) # solid
tvert = np.array([[0,1,0,0]]) # vertical
tdiag = np.array([[0,0,1,0]]) # diagonal
thor = np.array([[0,0,0,1]]) # horizontal

wji = np.array([[-0.2,0.1,0.1,0.1],
                [0.1,0.1,-0.1,0.1],
                [-0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1]])
wkj = np.array([[0.1,0.1,0.1,0.1],
                [0.1,0.2,-0.1,-0.1],
                [-0.1,-0.1,0.1,0.1],
                [0.2,0.1,0.2,0.1]])

np.set_printoptions(precision=3)

for i in xrange(10):
    ak = solid
    yi = tsol
    inj = np.dot(ak,wkj)
    aj = sigmoid(inj,False)
    ini = np.dot(aj,wji)
    ai = sigmoid(ini,False)
    deltai = (yi-ai) * sigmoid(ini,True)
    wji += alpha * aj.T * deltai # Wji += alpha (yi-ai)g'(ini)aj 
    wkj += alpha * np.dot(ak.T,np.dot(sigmoid(inj,True),np.dot(wji,deltai.T))) # Wkj += alpha ak g'(inj) sum i Wji(yi-ai)

# see if its working: predicted class correct?
ak = solid
inj = np.dot(ak,wkj)
aj = sigmoid(inj,False)
ini = np.dot(aj,wji)
ai = sigmoid(ini,False)
print ai


# In[10]:

a = np.array([[0], [1]])
b = np.array([[0,1], [2,3]])
x1 = a.T.dot(b)
print x1
c = np.array([[1], [2]])
d = np.array([[2,2], [3,2]])
x2 = c.T.dot(d)

print np.dot(b,d)

print b * d

print b
b[1, :] = 3 # Assing to second row
print b
b[:, 1] = 4 # Assing to second column
print b

#[0] #means line 0 of your matrix
#[(0,0)] #means cell at 0,0 of your matrix
#[0:1] #means lines 0 to 1 excluded of your matrix
#[:1] #excluding the first value means all lines until line 1 excluded
#[1:] #excluding the last param mean all lines starting form line 1 included
#[:] #excluding both means all lines
#[::2] #the addition of a second ':' is the sampling. (1 item every 2)
#[::] #exluding it means a sampling of 1
#[:,:] #simply uses a tuple (a single , represents an empty tuple) instead of an index.


# In[3]:

import numpy as np

# input a_k -- Wkj -- g(in_j) -  hidden a_j -- Wji -- g(in_i) - output a_i - y_i
# hidden to output layer
# Wji = Wji + alpha (yi-ai)g'(ini)aj      ini=sum over j Wji aj
# input to hidden layer
# Wkj = Wkj + alpha sum over i (yi-ai) Wji g'(inj) ak      inj = sum over k Wkj ak

np.set_printoptions(precision=4)

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    np.putmask(x, x > 100, 100) # limit array values of x (np.putmask(a, a >= m, m - 1))
    np.putmask(x, x < -100, -100)
    return 1/(1+np.exp(-x))

alpha = 1
# solid = 11, vertical = 10 or 01, horizontal = 11
solid = np.array([[1,1]]) # solid = 11
half0 = np.array([[1,0]]) # 50% = 10
half1 = np.array([[0,1]]) # 50% = 01
empty = np.array([[0,0]]) # empty = 00

# target outputs
tsol = np.array([[1,0]]) # solid
thal = np.array([[0,1]]) # half
tempt = np.array([[0,0]]) # empty

wji = np.array([[0.1,0.1],
                [0.1,0.1]])
wkj = np.array([[0.1,0.1],
                [0.1,0.1]])

ak = solid
yi = tsol
for i in xrange(20):
    inj = np.dot(ak,wkj)
    aj = sigmoid(inj,False)
    ini = np.dot(aj,wji)
    ai = sigmoid(ini,False)
    deltai = (yi-ai) * sigmoid(ini,True)
    wji += alpha * aj.T * deltai # Wji += alpha (yi-ai)g'(ini)aj 
    wkj += alpha * np.dot(ak.T,np.dot(sigmoid(inj,True),np.dot(wji,deltai.T))) # Wkj += alpha ak g'(inj) sum i Wji(yi-ai)

#print wji
#print wkj

# test - see if its working: predicted class correct?
ak = solid
inj = np.dot(ak,wkj)
aj = sigmoid(inj,False)
ini = np.dot(aj,wji)
ai = sigmoid(ini,False)
print ai


# In[14]:

import numpy as np

# input a_k -- Wkj -- g(in_j) -  hidden a_j -- Wji -- g(in_i) - output a_i - y_i
# hidden to output layer
# Wji = Wji + alpha (yi-ai)g'(ini)aj ini=sum over j Wji aj
# input to hidden layer
# Wkj = Wkj + alpha sum over i (yi-ai) Wji g'(inj) ak inj = sum over k Wkj ak

np.set_printoptions(precision=3)

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    np.putmask(x, x > 100, 100) # limit array values of x (np.putmask(a, a >= m, m - 1))
    np.putmask(x, x < -100, -100)
    return 1/(1+np.exp(-x))

alpha = 1
# example 20.15 from Russel/Norvig: 6 examples, 14 inputs, 14 outputs 
I1 = np.array([[1,1,1,1,1,1,1,0,0,0,0,0,0,0]])
I2 = np.array([[0,0,0,1,1,0,0,1,1,0,1,0,1,1]])
I3 = np.array([[1,1,1,0,1,0,0,1,1,0,0,0,1,1]])
I4 = np.array([[0,1,0,0,1,0,0,1,0,1,1,1,0,1]])
I5 = np.array([[0,0,1,1,0,1,1,0,1,1,0,0,1,0]])
I6 = np.array([[0,0,0,1,0,1,0,1,1,0,1,1,1,0]])

tout = np.array([[1,1,1,1,1,1,0,1,0,0,0,0,0,0]])

wji = np.array([[0.2,-0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])
wkj = np.array([[0.2,-0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])

yi=tout

for i in xrange(1):
    ak = I1 # input
    inj = np.dot(ak,wkj)
    aj = sigmoid(inj,False)
    ini = np.dot(aj,wji)
    ai = sigmoid(ini,False)
    deltai = (yi-ai) * sigmoid(ini,True)
    wji += alpha * aj.T * deltai # Wji += alpha (yi-ai)g'(ini)aj 
    wkj += alpha * np.dot(ak.T,np.dot(sigmoid(inj,True),np.dot(wji,deltai.T))) # Wkj += alpha ak g'(inj) sum i Wji(yi-ai)

for i in xrange(1):
    ak = I2 # input
    inj = np.dot(ak,wkj)
    aj = sigmoid(inj,False)
    ini = np.dot(aj,wji)
    ai = sigmoid(ini,False)
    deltai = (yi-ai) * sigmoid(ini,True)
    wji += alpha * aj.T * deltai # Wji += alpha (yi-ai)g'(ini)aj 
    wkj += alpha * np.dot(ak.T,np.dot(sigmoid(inj,True),np.dot(wji,deltai.T))) # Wkj += alpha ak g'(inj) sum i Wji(yi-ai)

for i in xrange(1):
    ak = I3 # input
    inj = np.dot(ak,wkj)
    aj = sigmoid(inj,False)
    ini = np.dot(aj,wji)
    ai = sigmoid(ini,False)
    deltai = (yi-ai) * sigmoid(ini,True)
    wji += alpha * np.dot(aj.T, deltai) # Wji += alpha (yi-ai)g'(ini)aj 
    wkj += alpha * np.dot(ak.T,np.dot(sigmoid(inj,True),np.dot(wji,deltai.T))) # Wkj += alpha ak g'(inj) sum i Wji(yi-ai)

for i in xrange(1):
    ak = I4 # input
    inj = np.dot(ak,wkj)
    aj = sigmoid(inj,False)
    ini = np.dot(aj,wji)
    ai = sigmoid(ini,False)
    deltai = (yi-ai) * sigmoid(ini,True)
    wji += alpha * aj.T * deltai # Wji += alpha (yi-ai)g'(ini)aj 
    wkj += alpha * np.dot(ak.T,np.dot(sigmoid(inj,True),np.dot(wji,deltai.T))) # Wkj += alpha ak g'(inj) sum i Wji(yi-ai)

for i in xrange(1):
    ak = I5 # input
    inj = np.dot(ak,wkj)
    aj = sigmoid(inj,False)
    ini = np.dot(aj,wji)
    ai = sigmoid(ini,False)
    deltai = (yi-ai) * sigmoid(ini,True)
    wji += alpha * aj.T * deltai # Wji += alpha (yi-ai)g'(ini)aj 
    wkj += alpha * np.dot(ak.T,np.dot(sigmoid(inj,True),np.dot(wji,deltai.T))) # Wkj += alpha ak g'(inj) sum i Wji(yi-ai)

for i in xrange(1):
    ak = I6 # input
    inj = np.dot(ak,wkj)
    aj = sigmoid(inj,False)
    ini = np.dot(aj,wji)
    ai = sigmoid(ini,False)
    deltai = (yi-ai) * sigmoid(ini,True)
    wji += alpha * aj.T * deltai # Wji += alpha (yi-ai)g'(ini)aj 
    wkj += alpha * np.dot(ak.T,np.dot(sigmoid(inj,True),np.dot(wji,deltai.T))) # Wkj += alpha ak g'(inj) sum i Wji(yi-ai)

#print wji
#print wkj

# test - see if its working: predicted class correct?
ak = tout # input
inj = np.dot(ak,wkj)
aj = sigmoid(inj,False)
ini = np.dot(aj,wji)
ai = sigmoid(ini,False)
print yi
print ai


# In[43]:

import numpy as np
from math import exp

# input a_k -- Wkj -- g(in_j) -  hidden a_j -- Wji -- g(in_i) - output a_i - y_i
# hidden to output layer
# Wji = Wji + alpha (yi-ai)g'(ini)aj      ini=sum over j Wji aj
# input to hidden layer
# Wkj = Wkj + alpha sum over i (yi-ai) Wji g'(inj) ak      inj = sum over k Wkj ak

np.set_printoptions(precision=4)

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    np.putmask(x, x > 100, 100) # limit array values of x (np.putmask(a, a >= m, m - 1))
    np.putmask(x, x < -100, -100)
    return 1/(1+np.exp(-x))

def sig(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+exp(-x))

alpha = 1
# solid = 11, vertical = 10 or 01, horizontal = 11
solid = np.array([[1,1]]) # solid = 11
half0 = np.array([[1,0]]) # 50% = 10
half1 = np.array([[0,1]]) # 50% = 01
empty = np.array([[0,0]]) # empty = 00

# target outputs
tsol = np.array([[1,0]]) # solid
thal = np.array([[0,1]]) # half
tempt = np.array([[0,0]]) # empty

#wji = np.array([[0.1,0.1],
#                [0.1,0.1]])
#wkj = np.array([[0.1,0.1],
#                [0.1,0.1]])

wji00 = 0.1
wji10 = 0.1
wji01 = 0.1
wji11 = 0.1

wkj00 = 0.1
wkj10 = 0.1
wkj01 = 0.1
wkj11 = 0.1

for i in xrange(20):
    ak0 = 1
    ak1 = 1
    yi0 = 1
    yi1 = 0
    #inj = np.dot(ak,wkj)
    inj0 = ak0 * wkj00 + ak1 * wkj10
    inj1 = ak0 * wkj01 + ak1 * wkj11
    #aj = sigmoid(inj,False)
    aj0 = sig(inj0,False)
    aj1 = sig(inj1,False)
    #ini = np.dot(aj,wji)
    ini0 = aj0 * wji00 + aj1 * wji10
    ini1 = aj0 * wji01 + aj1 * wji11
    #ai = sigmoid(ini,False)
    ai0 = sig(ini0,False)
    ai1 = sig(ini1,False)
    #deltai = (yi-ai) * sigmoid(ini,True)
    deltai0 = (yi0-ai0) * sig(ini0,True)
    deltai1 = (yi1-ai1) * sig(ini1,True)
    # Wji += alpha aj (yi-ai)g'(ini)
    wji00 += alpha * aj0 * deltai0
    wji10 += alpha * aj1 * deltai0
    wji01 += alpha * aj0 * deltai1
    wji11 += alpha * aj1 * deltai1
    # Wkj += alpha ak g'(inj) sum i Wji(yi-ai)
    wkj00 += alpha * ak0 * sig(inj0,True) * (wji00 * deltai0 + wji01 * deltai1) 
    wkj10 += alpha * ak1 * sig(inj0,True) * (wji00 * deltai0 + wji01 * deltai1)
    wkj01 += alpha * ak0 * sig(inj1,True) * (wji10 * deltai0 + wji11 * deltai1)
    wkj11 += alpha * ak1 * sig(inj1,True) * (wji10 * deltai0 + wji11 * deltai1)    
    #print wji
    #print wkj

#print wji
#print wkj

# test - see if its working: predicted class correct?
ak0 = 1
ak1 = 1
#inj = np.dot(ak,wkj)
inj0 = ak0 * wkj00 + ak1 * wkj10
inj1 = ak0 * wkj01 + ak1 * wkj11
#aj = sigmoid(inj,False)
aj0 = sig(inj0,False)
aj1 = sig(inj1,False)
#ini = np.dot(aj,wji)
ini0 = aj0 * wji00 + aj1 * wji10
ini1 = aj0 * wji01 + aj1 * wji11
#ai = sigmoid(ini,False)
ai0 = sig(ini0,False)
ai1 = sig(ini1,False)
print ai0
print ai1


# In[84]:

# pattern recognition deep neural network
# 1 input layer, 3 hidden layer, 1 output layer
# am - wml - al - wlk - ak - wkj - aj - wji - ai : yi target output
# ---------sig---------sig--------sig--------rect----
# 4---------4-----------4----------8----------4------
import numpy as np
np.set_printoptions(precision=4)

def sigmoid(x,deriv=False): # sigmoid
    if(deriv==True):
        return x*(1-x)
    np.putmask(x, x > 100, 100) # limit array values of x (np.putmask(a, a >= m, m - 1))
    np.putmask(x, x < -100, -100)
    return 1/(1+np.exp(-x))

def rect(x,deriv=False): # rectifier, might use smooth version f(x)=ln(1+exp(x)), f'=exp(x)/(1+exp(x))
    if(deriv==True):
        x[x > 0] = 1
    x[x < 0] = 0
    return x

alpha = 1
# solid = 1111, vertical = 1010 or 0101, diagonal = 1001 or 0110, horizontal = 1100 or 0011
# bias as last element
solid = np.array([[1,1,1,1]]) # solid = 1111
vert0 = np.array([[1,0,1,0]]) # vertical = 1010
vert1 = np.array([[0,1,0,1]]) # vertical = 0101
diag0 = np.array([[1,0,0,1]]) # diagonal = 1001
diag1 = np.array([[0,1,1,0]]) # diagonal = 0110
hor0 = np.array([[1,1,0,0]]) # horizontal = 1100
hor1 = np.array([[0,0,1,1]]) # horizontal = 0011

# target outputs
tsol = np.array([[1,0,0,0]]) # solid
tvert = np.array([[0,1,0,0]]) # vertical
tdiag = np.array([[0,0,1,0]]) # diagonal
thor = np.array([[0,0,0,1]]) # horizontal

wji = np.array([[-0.2,0.1,0.1,0.1],
                [0.1,0.1,-0.1,0.1],
                [-0.1,0.1,0.1,0.1],
                [0.1,0.1,-0.1,0.1],
                [-0.1,0.1,0.1,0.1],
                [0.1,0.1,-0.1,0.1],
                [-0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1]])
wkj = np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.2,-0.1,-0.1,0.1,0.1,0.1,0.1],
                [-0.1,-0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.2,0.1,0.2,0.1,0.1,0.1,0.1,0.1]])
wlk = np.array([[0.1,0.1,0.1,0.1],
                [0.1,0.2,-0.1,-0.1],
                [-0.1,-0.1,0.1,0.1],
                [0.2,0.1,0.2,0.1]])
wml = np.array([[0.1,0.1,0.1,0.1],
                [0.1,0.2,-0.1,-0.1],
                [-0.1,-0.1,0.1,0.1],
                [0.2,0.1,0.2,0.1]])

# training
for i in xrange(10):
    am = solid # input
    yi = tsol # target output
    inl = np.dot(am,wml)
    al = sigmoid(inl,False)
    ink = np.dot(al,wlk)
    ak = sigmoid(ink,False)
    inj = np.dot(ak,wkj)
    aj = rect(inj,False)
    ai = np.dot(aj,wji)
    deltai = alpha * (yi-ai)
    wji += deltai * aj.T
    #wkj += alpha * np.dot(ak.T,np.dot(sigmoid(inj,True),np.dot(wji,deltai.T)))
    wkj += (deltai * wji * sigmoid(inj,True).T * ak).T
    #wlk += alpha * np.dot(al.T,np.dot(sigmoid(inj,True),np.dot(wji,deltai.T)))
    #print np.dot(sigmoid(ink,True), wkj)
    #print deltai * wji * sigmoid(inj,True).T
    #print np.dot(np.dot(sigmoid(ink,True), wkj),(deltai * wji * sigmoid(inj,True).T)) * al.T
    wlk += np.dot(np.dot(sigmoid(ink,True), wkj),(deltai * wji * sigmoid(inj,True).T)) * al.T
    #print np.dot(sigmoid(inl,True), wlk)
    #print wkj * sigmoid(ink,True).T
    #print deltai * wji * sigmoid(inj,True).T
    #print np.dot((deltai * wji * sigmoid(inj,True).T).T,(wkj * sigmoid(ink,True).T).T)
    #print np.dot((deltai * wji * sigmoid(inj,True).T).T,(wkj * sigmoid(ink,True).T).T) * am.T
    wml += np.dot((deltai * wji * sigmoid(inj,True).T).T,(wkj * sigmoid(ink,True).T).T) * am.T

# test
am = solid # input
inl = np.dot(am,wml)
al = sigmoid(inl,False)
ink = np.dot(al,wlk)
ak = sigmoid(ink,False)
inj = np.dot(ak,wkj)
aj = rect(inj,False)
ai = np.dot(aj,wji)
print ai


# In[88]:

# pattern recognition deep neural network
# 1 input layer, 3 hidden layer, 1 output layer
# am - wml - al - wlk - ak - wkj - aj - wji - ai : yi target output
# ---------sig---------sig--------sig--------rect----
# 4---------4-----------4----------8----------4------
import numpy as np
np.set_printoptions(precision=4)

def sigmoid(x,deriv=False): # sigmoid
    if(deriv==True):
        return x*(1-x)
    np.putmask(x, x > 100, 100) # limit array values of x (np.putmask(a, a >= m, m - 1))
    np.putmask(x, x < -100, -100)
    return 1/(1+np.exp(-x))

def rect(x,deriv=False): # rectifier, might use smooth version f(x)=ln(1+exp(x)), f'=exp(x)/(1+exp(x))
    if(deriv==True):
        x[x > 0] = 1
    x[x < 0] = 0
    return x

alpha = 1
# solid = 1111, vertical = 1010 or 0101, diagonal = 1001 or 0110, horizontal = 1100 or 0011
# bias as last element
solid = np.array([[1,1,1,1]]) # solid = 1111
vert0 = np.array([[1,0,1,0]]) # vertical = 1010
vert1 = np.array([[0,1,0,1]]) # vertical = 0101
diag0 = np.array([[1,0,0,1]]) # diagonal = 1001
diag1 = np.array([[0,1,1,0]]) # diagonal = 0110
hor0 = np.array([[1,1,0,0]]) # horizontal = 1100
hor1 = np.array([[0,0,1,1]]) # horizontal = 0011

# target outputs
tsol = np.array([[1,0,0,0]]) # solid
tvert = np.array([[0,1,0,0]]) # vertical
tdiag = np.array([[0,0,1,0]]) # diagonal
thor = np.array([[0,0,0,1]]) # horizontal

wji = np.array([[-0.2,0.1,0.1,0.1],
                [0.1,0.1,-0.1,0.1],
                [-0.1,0.1,0.1,0.1],
                [0.1,0.1,-0.1,0.1],
                [-0.1,0.1,0.1,0.1],
                [0.1,0.1,-0.1,0.1],
                [-0.1,0.1,0.1,0.1],
                [0.1,0.1,0.1,0.1]])
wkj = np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.1,0.2,-0.1,-0.1,0.1,0.1,0.1,0.1],
                [-0.1,-0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.2,0.1,0.2,0.1,0.1,0.1,0.1,0.1]])
wlk = np.array([[0.1,0.1,0.1,0.1],
                [0.1,0.2,-0.1,-0.1],
                [-0.1,-0.1,0.1,0.1],
                [0.2,0.1,0.2,0.1]])
wml = np.array([[0.1,0.1,0.1,0.1],
                [0.1,0.2,-0.1,-0.1],
                [-0.1,-0.1,0.1,0.1],
                [0.2,0.1,0.2,0.1]])

# training
for i in xrange(5):
    am = solid # input
    yi = tsol # target output
    inl = np.dot(am,wml)
    al = sigmoid(inl,False)
    ink = np.dot(al,wlk)
    ak = sigmoid(ink,False)
    inj = np.dot(ak,wkj)
    aj = rect(inj,False)
    ai = np.dot(aj,wji)
    deltai = alpha * (yi-ai)
    wji += deltai * aj.T
    wkj += (deltai * wji * sigmoid(inj,True).T * ak).T
    wlk += np.dot(np.dot(sigmoid(ink,True), wkj),(deltai * wji * sigmoid(inj,True).T)) * al.T
    wml += np.dot((deltai * wji * sigmoid(inj,True).T).T,(wkj * sigmoid(ink,True).T).T) * am.T

    am = vert0 # input
    yi = tvert # target output
    inl = np.dot(am,wml)
    al = sigmoid(inl,False)
    ink = np.dot(al,wlk)
    ak = sigmoid(ink,False)
    inj = np.dot(ak,wkj)
    aj = rect(inj,False)
    ai = np.dot(aj,wji)
    deltai = alpha * (yi-ai)
    wji += deltai * aj.T
    wkj += (deltai * wji * sigmoid(inj,True).T * ak).T
    wlk += np.dot(np.dot(sigmoid(ink,True), wkj),(deltai * wji * sigmoid(inj,True).T)) * al.T
    wml += np.dot((deltai * wji * sigmoid(inj,True).T).T,(wkj * sigmoid(ink,True).T).T) * am.T
    
    am = diag0 # input
    yi = tdiag # target output
    inl = np.dot(am,wml)
    al = sigmoid(inl,False)
    ink = np.dot(al,wlk)
    ak = sigmoid(ink,False)
    inj = np.dot(ak,wkj)
    aj = rect(inj,False)
    ai = np.dot(aj,wji)
    deltai = alpha * (yi-ai)
    wji += deltai * aj.T
    wkj += (deltai * wji * sigmoid(inj,True).T * ak).T
    wlk += np.dot(np.dot(sigmoid(ink,True), wkj),(deltai * wji * sigmoid(inj,True).T)) * al.T
    wml += np.dot((deltai * wji * sigmoid(inj,True).T).T,(wkj * sigmoid(ink,True).T).T) * am.T

# test
am = vert0 # input
inl = np.dot(am,wml)
al = sigmoid(inl,False)
ink = np.dot(al,wlk)
ak = sigmoid(ink,False)
inj = np.dot(ak,wkj)
aj = rect(inj,False)
ai = np.dot(aj,wji)
print ai


# In[42]:

# pattern recognition deep neural network
# 1 input layer, 3 hidden layer, 1 output layer
# am - wml - al - wlk - ak - wkj - aj - wji - ai : yi target output
# ---------sig---------sig--------sig--------rect----
# 4---------4-----------4----------8----------4------

# repeat
#   for each e in examples
#      for each node j in input layer (l=0): aj = xj
#      for l=1 to L-1:
#        ini = sum j: wji aj
#        ai = g(ini)
#      for each node i in output layer (l=L-1)
#        deltai = g'(ini) (yi(e)-ai)
#      for l=L-2 to 0:
#        for each node j in layer l:
#          deltaj = g'(inj) sum i: wji deltai
#          for each node i in layer l+1:
#            wji += alpha aj deltai
# until stopping criterion reached

import numpy as np
np.set_printoptions(precision=4)

def sigmoid(x,deriv=False): # sigmoid
    if(deriv==True):
        return x*(1-x)
    np.putmask(x, x > 100, 100) # limit array values of x (np.putmask(a, a >= m, m - 1))
    np.putmask(x, x < -100, -100)
    return 1/(1+np.exp(-x))

def rect(x,deriv=False): # rectifier, might use smooth version f(x)=ln(1+exp(x)), f'=exp(x)/(1+exp(x))
    if(deriv==True):
        x[x > 0] = 1
    x[x < 0] = 0
    return x

L = 5 # number of layers
imax = 100
alpha = -1

def gfunc(x,l,deriv=False): # response function by layer
    if l == L-1:
        x = rect(ini[l],False)
    else:
        x = sigmoid(ini[l],False)
    return x   

# solid = 1111, vertical = 1010 or 0101, diagonal = 1001 or 0110, horizontal = 1100 or 0011
# bias as last element
solid = np.array([[1,1,1,1]]) # solid = 1111
vert0 = np.array([[1,0,1,0]]) # vertical = 1010
vert1 = np.array([[0,1,0,1]]) # vertical = 0101
diag0 = np.array([[1,0,0,1]]) # diagonal = 1001
diag1 = np.array([[0,1,1,0]]) # diagonal = 0110
hor0 = np.array([[1,1,0,0]]) # horizontal = 1100
hor1 = np.array([[0,0,1,1]]) # horizontal = 0011

# target outputs
tsol = np.array([[1,0,0,0]]) # solid
tvert = np.array([[0,1,0,0]]) # vertical
tdiag = np.array([[0,0,1,0]]) # diagonal
thor = np.array([[0,0,0,1]]) # horizontal

wji = []
wji.append(0.2*np.random.random((4,4))-0.05)
wji.append(0.2*np.random.random((4,4))-0.05)
wji.append(0.2*np.random.random((4,8))-0.05)
wji.append(0.2*np.random.random((8,4))-0.05)

ini = []
ai = []
deltai = []

# init ai
ai.append(np.array([[0,0,0,0]])) # 0
ai.append(np.array([[0,0,0,0]])) # 1
ai.append(np.array([[0,0,0,0]])) # 2
ai.append(np.array([[0,0,0,0,0,0,0,0]])) # 3
ai.append(np.array([[0,0,0,0]])) # 4

# init ini
ini.append(np.array([[0,0,0,0]]))
ini.append(np.array([[0,0,0,0]]))
ini.append(np.array([[0,0,0,0]]))
ini.append(np.array([[0,0,0,0,0,0,0,0]]))
ini.append(np.array([[0,0,0,0]]))

# init deltai
deltai.append(np.array([[0,0,0,0]]))
deltai.append(np.array([[0,0,0,0]]))
deltai.append(np.array([[0,0,0,0]]))
deltai.append(np.array([[0,0,0,0,0,0,0,0]]))
deltai.append(np.array([[0,0,0,0]]))

for i in range(imax):
    ai[0] = solid # input
    yi = tsol #target output     
    for l in range(1, L):
        ini[l] = np.dot(ai[l-1],wji[l-1])
        ai[l] = gfunc(ini[l],l,False)
    deltai[L-1] = np.multiply(gfunc(ini[L-1],L-1,True),(ai[L-1] - yi)) # component-wise multiply
    #print("error %s  " % ((yi - ai[L-1])))
    for l in range(L-2, -1, -1):
        deltai[l] = np.multiply(gfunc(ini[l],l,True), np.dot(wji[l], deltai[l+1].T).T)
        wji[l] += alpha * np.multiply(ai[l].T, deltai[l+1])

ai[0] = solid # input
for l in range(1, L):
    ini[l] = np.dot(ai[l-1],wji[l-1])
    ai[l] = gfunc(ini[l],l,False)
print str(ai[L-1])


# In[12]:

# pattern recognition deep neural network
# 1 input layer, 3 hidden layer, 1 output layer
# am - wml - al - wlk - ak - wkj - aj - wji - ai : yi target output
# ---------sig---------sig--------sig--------rect----
# 4---------4-----------4----------8----------4------

# repeat
#   for each e in examples
#      for each node j in input layer (l=0): aj = xj
#      for l=1 to L-1:
#        ini = sum j: wji aj
#        ai = g(ini)
#      for each node i in output layer (l=L-1)
#        deltai = g'(ini) (yi(e)-ai)
#      for l=L-2 to 0:
#        for each node j in layer l:
#          deltaj = g'(inj) sum i: wji deltai
#          for each node i in layer l+1:
#            wji += alpha aj deltai
# until stopping criterion reached

import numpy as np
np.set_printoptions(precision=4)

def sigmoid(x,deriv=False): # sigmoid
    if(deriv==True):
        return x*(1-x)
    np.putmask(x, x > 100, 100) # limit array values of x (np.putmask(a, a >= m, m - 1))
    np.putmask(x, x < -100, -100)
    return 1/(1+np.exp(-x))

def rect(x,deriv=False): # rectifier, might use smooth version f(x)=ln(1+exp(x)), f'=exp(x)/(1+exp(x))
    if(deriv==True):
        x[x > 0] = 1
    x[x < 0] = 0
    return x

L = 5 # number of layers
imax = 3
alpha = 1

def gfunc(x,l,deriv=False): # response function by layer
    if l == L-1:
        x = rect(ini[l],False)
    else:
        x = sigmoid(ini[l],False)
    return x  

def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in xrange(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i

def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i

def L_i_fullvec(X, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  # evaluate loss over all examples in X without using any for loops
  # left as exercise to reader in the assignment

def softmax(y, a):
    return -np.log(np.exp(y) / np.sum(np.exp(a)))

def totloss(y, a, w):
    reg = 0.001
    data_loss = np.sum(softmax(y, a))/len(y)
    reg_loss = 0.5*reg*np.sum(w*w)
    loss = data_loss + reg_loss
    return loss

def dscore(y, a):
    return ((np.exp(y) / np.sum(np.exp(a)))-1) / len(y)

# solid = 1111, vertical = 1010 or 0101, diagonal = 1001 or 0110, horizontal = 1100 or 0011
# bias as last element
solid = np.array([[1,1,1,1]]) # solid = 1111
vert0 = np.array([[1,0,1,0]]) # vertical = 1010
vert1 = np.array([[0,1,0,1]]) # vertical = 0101
diag0 = np.array([[1,0,0,1]]) # diagonal = 1001
diag1 = np.array([[0,1,1,0]]) # diagonal = 0110
hor0 = np.array([[1,1,0,0]]) # horizontal = 1100
hor1 = np.array([[0,0,1,1]]) # horizontal = 0011

# target outputs
tsol = np.array([[1,0,0,0]]) # solid
tvert = np.array([[0,1,0,0]]) # vertical
tdiag = np.array([[0,0,1,0]]) # diagonal
thor = np.array([[0,0,0,1]]) # horizontal

wji = []
wji.append(0.2*np.random.random((4,5))-0.05) # bias on 2nd axis
wji.append(0.2*np.random.random((5,5))-0.05) # bias on 1st+2nd axis
wji.append(0.2*np.random.random((5,9))-0.05) # bias on 1st+2nd axis
wji.append(0.2*np.random.random((9,4))-0.05) # bias on 1st axis
# bias should be init to 0
wji[0][:,4] = 0 # b[1,:]...2nd row, b[:,1]...2nd column
wji[1][:,4] = 0
wji[1][4,:] = 0
wji[2][:,8] = 0
wji[2][4,:] = 0
wji[3][4,:] = 0

ini = []
ai = []
deltai = []

# init ai
ai.append(np.array([[0,0,0,0]])) # 0
ai.append(np.array([[0,0,0,0,0]])) # 1 + bias
ai.append(np.array([[0,0,0,0,0]])) # 2 + bias
ai.append(np.array([[0,0,0,0,0,0,0,0,0]])) # 3 + bias
ai.append(np.array([[0,0,0,0]])) # 4

# init ini
ini.append(np.array([[0,0,0,0]]))
ini.append(np.array([[0,0,0,0,0]]))
ini.append(np.array([[0,0,0,0,0]]))
ini.append(np.array([[0,0,0,0,0,0,0,0,0]]))
ini.append(np.array([[0,0,0,0]]))

# init deltai
deltai.append(np.array([[0,0,0,0]]))
deltai.append(np.array([[0,0,0,0,0]]))
deltai.append(np.array([[0,0,0,0,0]]))
deltai.append(np.array([[0,0,0,0,0,0,0,0,0]]))
deltai.append(np.array([[0,0,0,0]]))

for i in range(imax):
    ai[0] = vert0 # input
    yi = tvert #target output     
    for l in range(1, L):
        ini[l] = np.dot(ai[l-1],wji[l-1])
        ai[l] = gfunc(ini[l],l,False)
    #deltai[L-1] = np.multiply(gfunc(ini[L-1],L-1,True),(ai[L-1] - yi)) # component-wise multiply
    deltai[L-1] = np.multiply(gfunc(ini[L-1],L-1,True),dscore(yi, ai[L-1]))
    #print dscore(yi, ai[L-1])
    for l in range(L-2, -1, -1):
        deltai[l] = np.multiply(gfunc(ini[l],l,True), np.dot(wji[l], deltai[l+1].T).T)
        wji[l] += alpha * np.multiply(ai[l].T, deltai[l+1])
        #print deltai[l]

ai[0] = vert0 # input
for l in range(1, L):
    ini[l] = np.dot(ai[l-1],wji[l-1])
    ai[l] = gfunc(ini[l],l,False)
print str(ai[L-1])


# In[9]:

# forward pass
W = np.random.randn(2, 3)
X = np.random.randn(3, 2)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)

print dX


# In[10]:

# forward-pass of a 3-layer neural network:
f = lambda x: 1.0/(1.0 + np.exp(-x)) # activation function (use sigmoid)
x = np.random.randn(3, 1) # random input vector of three numbers (3x1)
h1 = f(np.dot(W1, x) + b1) # calculate first hidden layer activations (4x1)
h2 = f(np.dot(W2, h1) + b2) # calculate second hidden layer activations (4x1)
out = np.dot(W3, h2) + b3 # output neuron (1x1)


# In[ ]:



