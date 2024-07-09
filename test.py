import os, sys,numpy,scipy,scipy.io

# --------------------------------------------
# Parameters
# --------------------------------------------
seed  = 3453
#split = int(sys.argv[1]) # test split
split = 3
mb    = 25     # size of the minibatch
hist  = 0.1    # fraction of the history to be remembered

# --------------------------------------------
# Load data
# --------------------------------------------
numpy.random.seed(seed)
#if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('qm7.mat')

# --------------------------------------------
# Extract training data
# --------------------------------------------
P = dataset['P'][list(range(0, split)) + list(range(split+1, 5))].flatten()

"""
X = dataset['X'][P]
print("Before reshape : ", X.shape)
print(type(X))
X = X.reshape(X.shape[0], -1)
print("After reshape : ", X.shape)
T = dataset['T'][0,P]


Ptrain = dataset['P'][range(0,split)+range(split+1,5)].flatten()
Ptest  = dataset['P'][split]

for P,name in zip([Ptrain,Ptest],['training','test']):
	# --------------------------------------------
	# Extract test data
	# --------------------------------------------
	X = dataset['X'][P]
	T = dataset['T'][0,P]

	# --------------------------------------------
	# Test the neural network
	# --------------------------------------------
	print('\n%s set:'%name)
	Y = numpy.array([nn.forward(X) for _ in range(10)]).mean(axis=0)
	print('MAE:  %5.2f kcal/mol'%numpy.abs(Y-T).mean(axis=0))
	print('RMSE: %5.2f kcal/mol'%numpy.square(Y-T).mean(axis=0)**.5)
"""
print(dataset['P'])
print(numpy.max(dataset['P'][0:2]))
print((dataset['P'][0:2]).shape)
print(dataset['P'].shape)
print(numpy.max(dataset['P']))
Ptrain = dataset['P'][list(range(0,split))+list(range(split+1,5))].flatten()
print(numpy.max(Ptrain))
print(Ptrain.shape)
print(type(dataset))
print(dataset.keys())

X = dataset['X']
x_0 = X[0]
print(x_0)
x_0_r = x_0.reshape(x_0.shape[0], -1)
print(x_0_r)

import numpy as np
x = np.array([[3,3.1, 3.2], [3.9, 4, 4.1], [5, 6, 8]])
d = np.array([1,2,3,4,5])
print(x[[1,1], [2,2]])
print(d.shape)
d = d.reshape((5,))
print(d.shape)
d = d.reshape(1, -1)
print(d)
pre_res = x.nonzero()
res = np.transpose(x.nonzero())
print(res)
print(res.shape)
print(x[pre_res[0], pre_res[1]])

diagonal = np.array([1,23,25,29, 0, 30, 32, 0])
print(diagonal.nonzero())
diag = diagonal[diagonal.nonzero()]
print(diag)
print(diag.shape)
