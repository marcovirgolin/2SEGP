import numpy as np
from copy import deepcopy

class FitnessFunction:

	def __init__( self, X_train, y_train, ensemble_size, use_linear_scaling=True, error_metric='mse' ):
		self.X_train = X_train
		self.y_train = y_train
		self.use_linear_scaling = use_linear_scaling
		self.evaluations = 0
		self.error_metric = error_metric
		if error_metric == 'binary_acc':
			self.classes = np.unique(y_train)
			if len(self.classes) != 2:
				raise ValueError('error_metric set to binary accuracy but num classes is', len(self.classes))

		self.ensemble_size = ensemble_size
		self.ensemble_elites = [None] * ensemble_size
		self.bootstrap_indices = []
		self.bootstrap_ys = []
		for i in range(ensemble_size):
			self.bootstrap_indices.append( [ j for j in np.random.randint(len(y_train), size=len(y_train)) ] )
			self.bootstrap_ys.append(self.y_train[self.bootstrap_indices[i]])
		self.bootstrap_indices = np.array(self.bootstrap_indices)
		self.bootstrap_ys = np.array(self.bootstrap_ys)

		if use_linear_scaling:
			self.bootstrap_ys_means = np.mean(self.bootstrap_ys, axis=1).reshape((self.ensemble_size, 1))
			self.bootstrap_ys_deviations = self.bootstrap_ys - self.bootstrap_ys_means



	def ComputeError( self, bootstrap_outputs ):
		if self.error_metric == 'mse':
			bootstrap_residuals = self.bootstrap_ys - bootstrap_outputs
			squared_bootstrap_residuals = np.square(bootstrap_residuals)
			bootstrap_errors = np.mean(squared_bootstrap_residuals, axis=1)
		elif self.error_metric == 'binary_acc':
			bootstrap_outputs[ bootstrap_outputs > .5 ] = 1.0
			bootstrap_outputs[ bootstrap_outputs <= .5 ] = 0.0
			bootstrap_errors = np.mean(self.bootstrap_ys != bootstrap_outputs, axis=1 )
		return bootstrap_errors


	def Evaluate( self, individual ):

		if not individual.requires_reevaluation:
			return

		self.evaluations = self.evaluations + 1

		output = individual.GetOutput( self.X_train )

		individual.bootstrap_errors = []
		individual.bootstrap_as = []
		individual.bootstrap_bs = []
		bootstrap_outputs = output[self.bootstrap_indices]

		individual.fitness = np.inf

		if self.use_linear_scaling:

			bootstrap_outputs_means = np.mean(bootstrap_outputs, axis=1).reshape((self.ensemble_size,1))
			bootstrap_outputs_deviations = bootstrap_outputs - bootstrap_outputs_means

			bootstrap_bs = np.sum(np.multiply(self.bootstrap_ys_deviations, bootstrap_outputs_deviations), axis=1).reshape((self.ensemble_size,1)) / ( np.sum(np.square(bootstrap_outputs_deviations), axis=1).reshape((self.ensemble_size,1)) + 1e-10 )
			bootstrap_as = self.bootstrap_ys_means - np.multiply(bootstrap_outputs_means, bootstrap_bs)
			scaled_bootstrap_outputs = np.multiply( bootstrap_outputs, bootstrap_bs )
			scaled_bootstrap_outputs = np.add( scaled_bootstrap_outputs, bootstrap_as )

			#### IF YOU WANNA DOUBLE CHECK #######################
			#covariances = np.array([np.cov(self.bootstrap_ys[i], bootstrap_outputs[i]) for i in range(self.ensemble_size)]) #np.cov uses N-1 instead of N as denom
			#variances = np.var(bootstrap_outputs, axis=1)
			#bootstrap_outputs_means = np.mean(bootstrap_outputs, axis=1)
			#bootstrap_bs = (covariances[:,0,1] / (variances + 1e-10))
			#bootstrap_as = (self.bootstrap_ys_means.reshape((self.ensemble_size,)) - np.multiply(bootstrap_bs, bootstrap_outputs_means))
			#scaled_bootstrap_outputs = np.multiply( bootstrap_outputs.T, bootstrap_bs ).T
			#scaled_bootstrap_outputs = np.add(scaled_bootstrap_outputs.T, bootstrap_as ).T
			######################################################

			individual.bootstrap_as = bootstrap_as
			individual.bootstrap_bs = bootstrap_bs

			bootstrap_outputs = scaled_bootstrap_outputs

		individual.bootstrap_errors = self.ComputeError( bootstrap_outputs )
		individual.fitness = np.max(individual.bootstrap_errors)

		for i in range(self.ensemble_size):
			if not self.ensemble_elites[i] or individual.bootstrap_errors[i] < self.ensemble_elites[i].bootstrap_errors[i]:
				self.ensemble_elites[i] = deepcopy(individual)

		individual.requires_reevaluation = False
