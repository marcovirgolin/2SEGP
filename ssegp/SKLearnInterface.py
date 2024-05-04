from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.stats import mode

import inspect

from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from ssegp.Fitness.FitnessFunction import FitnessFunction
from ssegp.Evolution.Evolution import SSEGP


class SSEGPRegressionEstimator(BaseEstimator, RegressorMixin):

	def __init__(self, pop_size=100, ensemble_size=25,
		max_generations=100,
		max_evaluations=-1,
		max_time=-1,
		functions=[ AddNode(), SubNode(), MulNode(), DivNode() ],
		use_erc=True,
		crossover_rate=0.5,
		mutation_rate=0.5,
		initialization_max_tree_height=6,
		max_tree_size=100,
		error_metric='mse',
		use_linear_scaling=True, 
		verbose=False ):

		if error_metric != 'mse' and error_metric != 'binary_acc':
			raise ValueError('error_metric should be "mse" or "binary_acc"')

		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop('self')
		for arg, val in values.items():
			setattr(self, arg, val)


	def fit(self, X, y):

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		self.X_ = X
		self.y_ = y

		fitness_function = FitnessFunction( X, y, self.ensemble_size, self.use_linear_scaling, error_metric=self.error_metric )

		terminals = []
		if self.use_erc:
			terminals.append( EphemeralRandomConstantNode() )
		n_features = X.shape[1]
		for i in range(n_features):
			terminals.append(FeatureNode(i))

		gp = SSEGP(fitness_function, self.functions, terminals, 
			pop_size=self.pop_size,
			ensemble_size=self.ensemble_size, 
			max_generations=self.max_generations,
			max_time = self.max_time,
			max_evaluations = self.max_evaluations,
			crossover_rate=self.crossover_rate,
			mutation_rate=self.mutation_rate,
			initialization_max_tree_height=self.initialization_max_tree_height,
			max_tree_size=self.max_tree_size,
			verbose=self.verbose)

		gp.Run()

		self.gp_ = gp

		return self

	def predict(self, X):
		# Check fit has been called
		check_is_fitted(self, ['gp_'])

		X = check_array(X)

		predictions = []
		for i, elite in enumerate(self.gp_.fitness_function.ensemble_elites):
			prediction = elite.GetOutput(X)
			if self.use_linear_scaling:
				prediction = elite.bootstrap_as[i] + elite.bootstrap_bs[i] * prediction
			predictions.append(prediction)
		predictions = np.array(predictions)

		if self.error_metric == 'binary_acc':
			predictions[predictions > .5] = 1.0
			predictions[predictions <= .5] = 0.0
			prediction = mode(predictions)[0].squeeze()
		else:
			prediction = np.mean(predictions, axis=0)
   
		assert len(prediction) == len(X)
   
		return prediction

	def score(self, X, y=None):
		if y is None:
			raise ValueError('The ground truth y was not set')

		# Check fit has been called
		prediction = self.predict(X)

		if self.error_metric == 'binary_acc':
			return np.mean(y == prediction)			 # accuracy
		else:
			return -1.0 * np.mean(np.square(y - prediction)) # negative MSE

	def get_params(self, deep=True):
		attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		attributes = [a for a in attributes if not (a[0].endswith('_') or a[0].startswith('_'))]

		dic = {}
		for a in attributes:
			dic[a[0]] = a[1]

		return dic

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_ensemble_elitists(self):
		check_is_fitted(self, ['gp_'])
		result = self.gp_.fitness_function.ensemble_elites
		return result
