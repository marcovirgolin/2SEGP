# Libraries
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from copy import deepcopy


# Internal imports
from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from ssegp.SKLearnInterface import SSEGPRegressionEstimator as SSEGPRegEst

# Set seed
np.random.seed(42)


# Load regression dataset 
X, y = sklearn.datasets.load_boston( return_X_y=True ) #sklearn.datasets.load_diabetes(return_X_y=True) #
y_std = np.std(y)
X = scale(X)
y = scale(y)

# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )

pop_size = 100
ensemble_size=10
gp = SSEGPRegEst(pop_size=pop_size, ensemble_size=ensemble_size, max_generations=30, verbose=True, max_tree_size=100,
	crossover_rate=0.5, mutation_rate=0.5,
	initialization_max_tree_height=6, use_erc=True,
	use_linear_scaling=True, error_metric='mse',
	functions = [ AddNode(), SubNode(), MulNode(), DivNode() ])

gp.fit(X_train,y_train)

print('Train RMSE:',  y_std * np.sqrt( np.mean(np.square(y_train - gp.predict(X_train)))) )
print('Test RMSE:', y_std * np.sqrt( np.mean(np.square(y_test - gp.predict(X_test)))) )
