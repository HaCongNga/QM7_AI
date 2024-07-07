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