import numpy as np
from copy import deepcopy
from numpy.random import randint
from scipy.special import softmax

def TournamentSelect( population, how_many_to_select, tournament_size=4 ):

	pop_size = len(population)
	selection = []

	while len(selection) < how_many_to_select:

		best = population[randint(pop_size)]
		for i in range(tournament_size - 1):
			contestant = population[randint(pop_size)]

			if contestant.fitness <= best.fitness:
				best = contestant

		survivor = deepcopy(best)
		selection.append(survivor)

	return selection


def BagTruncationSelection( population, how_many_to_select, ensemble_size ):
	selection = []

	probabilities_bag_selection = np.array( [1.0/ensemble_size] * ensemble_size )
	slots_bag_selection = np.round(probabilities_bag_selection*how_many_to_select).astype(int)

	# fix-up
	while(np.sum(slots_bag_selection) > how_many_to_select):
		slots_bag_selection[randint(ensemble_size)] -= 1
	while(np.sum(slots_bag_selection) < how_many_to_select):
		slots_bag_selection[randint(ensemble_size)] += 1


	for i in range(len(slots_bag_selection)):
		chosen_pop = sorted(population, key=lambda x : x.bootstrap_errors[i])
		bests_for_this_bag = chosen_pop[0:slots_bag_selection[i]]
		for b in bests_for_this_bag:
			selection.append(deepcopy(b))

	return selection
