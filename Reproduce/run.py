# Libraries
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from ssegp.SKLearnInterface import SSEGPRegressionEstimator as SSEGPRegEst
import time


# Classification dataset names
CLASS_DATASET_NAMES = ['bcw','heart','iono','parks','sonar']

# Allow for deep trees to be deep-copied
sys.setrecursionlimit(100000)

# Read what dataset to tackle
dataset_name = sys.argv[1]
seed = np.random.randint(9999999)


# Load the dataset
Xy = np.genfromtxt('Datasets/'+dataset_name+'.csv', delimiter=',')
X = Xy[:, :-1]
y = Xy[:, -1]   # last column is the label


# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train_std[X_train_std == 0] = 1e-10


# Z-scoring of the features (label not needed if using linear scaling in regression or performing classification)
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# Set up 2SEGP

# Decide the error to be measured
if dataset_name in CLASS_DATASET_NAMES:
    error_metric = 'binary_acc'
else:
    error_metric = 'mse'

# Define extra function nodes if not available in SimpleGP
class SqrtNode(Node):              
    def __init__(self):
        super(SqrtNode,self).__init__()
        self.arity = 1

    def __repr__(self):
        return 'sqrt'

    def GetOutput( self, X ):
        X0 = self._children[0].GetOutput( X )
        result = np.sqrt(np.abs(X0))    # abs for protection
        return result

# Set up all hyper-parameters (approx. as Rodrigues 2020)
ssegp = SSEGPRegEst(pop_size=500, ensemble_size=50, max_generations=100, verbose=True, max_tree_size=500,
            crossover_rate=0.5, mutation_rate=0.5,
            initialization_max_tree_height=6, use_linear_scaling=True, use_erc=True,
            error_metric=error_metric, # for binary classification, use 'binary_acc'
            functions=[AddNode(), SubNode(), MulNode(), DivNode(), LogNode(), SqrtNode()])

# Run
start_time = time.time()
ssegp.fit(X_train, y_train)

# Collect results
elapsed_time = time.time() - start_time

p_train = ssegp.predict(X_train)
p_test = ssegp.predict(X_test)
n_nodes = np.sum([len(x.GetSubtree()) for x in ssegp.get_ensemble_elitists()])


indicators = ['performance_train','performance_test','elapsed_time','n_nodes']
df = pd.DataFrame(columns=indicators)
# If classification
if dataset_name in CLASS_DATASET_NAMES:
    perf_train = np.mean(y_train == p_train)
    perf_test = np.mean(y_test == p_test)
else: # regression
    perf_train = np.sqrt(np.mean(np.square(y_train - p_train)))
    perf_test = np.sqrt(np.mean(np.square(y_test - p_test)))
df.loc[0] = [perf_train, perf_test, elapsed_time, n_nodes]
print(df)
# Save result
df.to_csv('result_' + dataset_name + '_' + str(seed) + '.csv', index=False)