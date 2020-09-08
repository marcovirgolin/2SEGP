import numpy as np
from numpy.random import random, randint
import time
from copy import deepcopy

from simplegp.Variation import Variation
from ssegp.Selection import Selection


class SSEGP:

	def __init__(
		self,
		fitness_function,
		functions,
		terminals,
		pop_size=500,
		crossover_rate=0.5,
		mutation_rate=0.5,
		max_evaluations=-1,
		max_generations=-1,
		max_time=-1,
		initialization_max_tree_height=4,
		max_tree_size=100,
		ensemble_size = 25,
		verbose=False
		):

		self.pop_size = pop_size
		self.fitness_function = fitness_function
		self.functions = functions
		self.terminals = terminals
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate

		self.max_evaluations = max_evaluations
		self.max_generations = max_generations
		self.max_time = max_time

		self.initialization_max_tree_height = initialization_max_tree_height
		self.max_tree_size = max_tree_size
		self.ensemble_size = ensemble_size

		self.generations = 0
		self.verbose = verbose

		self.population = []


	def __ShouldTerminate(self):
		must_terminate = False
		elapsed_time = time.time() - self.start_time
		if self.max_evaluations > 0 and self.fitness_function.evaluations >= self.max_evaluations:
			must_terminate = True
		elif self.max_generations > 0 and self.generations >= self.max_generations:
			must_terminate = True
		elif self.max_time > 0 and elapsed_time >= self.max_time:
			must_terminate = True

		if must_terminate and self.verbose:
			print('Terminating at\n\t',
				self.generations, 'generations\n\t', self.fitness_function.evaluations, 'evaluations\n\t', np.round(elapsed_time,2), 'seconds')

		return must_terminate


	def Run(self):

		self.start_time = time.time()

		# ramped half-n-half initialization w/ rejection of duplicates
		self.population = []

		attempts_duplicate_rejection = 0
		max_attempts_duplicate_rejection = self.pop_size * 10
		already_generated_trees = set()

		half_pop_size = int(self.pop_size/2)
		for j in range(2):

			if j == 0:
				method = 'full'
			else:
				method = 'grow'

			curr_max_depth = 2
			init_depth_interval = self.pop_size / (self.initialization_max_tree_height - 1) / 2
			next_depth_interval = init_depth_interval

			i = 0
			while len(self.population) < (j+1)*half_pop_size:

				if i >= next_depth_interval:
					next_depth_interval += init_depth_interval
					curr_max_depth += 1

				t = Variation.GenerateRandomTree( self.functions, self.terminals, curr_max_depth, curr_height=0, method=method )
				t_as_str = str(t.GetSubtree())
				if t_as_str in already_generated_trees and attempts_duplicate_rejection < max_attempts_duplicate_rejection:
					del t
					attempts_duplicate_rejection += 1
					continue
				else:
					already_generated_trees.add(t_as_str)
					t.requires_reevaluation=True
					self.fitness_function.Evaluate( t )
					self.population.append( t )
					i += 1


		while not self.__ShouldTerminate():

			O = []

			for i in range( self.pop_size ):

				o = deepcopy(self.population[i])

				r = random()

				if ( r < self.crossover_rate + self.mutation_rate):
					if ( r < self.crossover_rate):
						o = Variation.SubtreeCrossover( o, self.population[ randint( self.pop_size ) ] )
						o.requires_reevaluation = True
					else: # ( random() < self.mutation_rate ):
						o = Variation.SubtreeMutation( o, self.functions, self.terminals, max_height=self.initialization_max_tree_height )
						o.requires_reevaluation = True

				invalid_offspring = False
				if (self.max_tree_size > -1 and len(o.GetSubtree()) > self.max_tree_size):
					invalid_offspring = True
				if invalid_offspring:
					del o
					o = deepcopy(self.population[i])
				O.append(o)

			for o in O:
				self.fitness_function.Evaluate(o)

			ens_elite_fitnesses = [self.fitness_function.ensemble_elites[i].bootstrap_errors[i] for i in range(self.ensemble_size)]
			self.population = Selection.BagTruncationSelection( self.population+O, self.pop_size, ensemble_size=self.ensemble_size)

			self.generations = self.generations + 1

			if self.verbose:
				print ('g:',self.generations,'ensemble elite fitnesses:', np.round(ens_elite_fitnesses,3))
