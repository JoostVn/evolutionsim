import numpy as np
from matplotlib import pyplot as plt
import math

class GeneticAlgorithm:
    
    def __init__(self, genomes, fitness, selection, crossover, mutation,
                 elitism, copy_fract, judgement_day_std=0):
        
        # Population 
        self.genomes = genomes
        self.fitness = fitness
        self.indices = np.arange(len(genomes))
        self.popsize = len(self.genomes)
        
        # Evolution parameters and functions
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.elitism = elitism
        self.copy_fract = copy_fract
        self.judgement_day_std = judgement_day_std

    def evolve_population(self):
        
        # Judgement day if standard deviation of population is too low
        mean_std = np.std(self.genomes, axis=0).mean() 
        print(mean_std)
        if mean_std < self.judgement_day_std: 
            new_population_genomes = self.judgement_day()
            return new_population_genomes

        # Fetch the best indivuals as elites
        elite_indices = np.argsort(self.fitness)[-self.elitism:]
        elite_genomes = self.genomes[elite_indices]

        # Copy randomly selected inviduals without mutation or offspring
        self.selection.set_population(self.genomes, self.fitness)
        nr_copy = int(self.copy_fract * self.popsize)
        copy_indices = self.selection.get_n_unique(nr_copy, exclude=elite_indices)
        copy_genomes = self.genomes[copy_indices]
        
        # Selection, Crossover, Mutation
        nr_offspring = self.popsize - nr_copy - self.elitism
        offspring_genomes = []
        while len(offspring_genomes) < nr_offspring:
            parent_genomes = self.genomes[self.selection.get_n_unique(2)]
            offspring = self.crossover.get_offspring(parent_genomes)
            
            # Perform multiple mutations if mutation is given as a list
            for m in self.mutation:
                offspring = m.mutate_genome(offspring)
           
            offspring_genomes.append(offspring)
        offspring_genomes = np.array(offspring_genomes)
        
        # Combining elites, copies and offspring into new population
        new_population_genomes = np.vstack((
            elite_genomes, copy_genomes, offspring_genomes))
        return new_population_genomes
        
    def get_genome_index(self, genome):
        """
        Returns the index of a given genome in self indices.
        """
        return np.where((self.genomes == genome).all(axis=1))[0][0]
        
    def judgement_day(self):
        """
        If the standard deviation is under a given limit, performs extreme
        mutation over the full population with the exception of elites. The
        goals of this is escaping local minima.
        """
        elite_indices = np.argsort(self.fitness)[-self.elitism:]
        elite_genomes = self.genomes[elite_indices]
        mask = np.ones(len(self.genomes), bool)
        mask[elite_indices] = False
        other_genomes = self.genomes[mask]
        super_mutated_genomes = []
        for genome in other_genomes:
            for i in range(3):
                for m in self.mutation:
                    genome = m.mutate_genome(genome)
            super_mutated_genomes.append(genome)
        super_mutated_genomes = np.array(super_mutated_genomes)
        new_population_genomes = np.vstack((elite_genomes, super_mutated_genomes))
        return new_population_genomes
        


class Selection:
    """
    Base class inherited by selection algorithms. The get_singe method is
    implemented differently for the different versions.
    """
    def __init__(self):
        pass
    
    def set_population(self, genomes, fitness):
        self.genomes = genomes
        self.fitness = fitness
        self.indices = np.arange(genomes.shape[0])

    def get_single(self):
        pass
    
    def get_n_unique(self, n, exclude=np.empty(0)):
        selected = []
        while len(selected) < n:
            selected_index = self.get_single()
            if len(exclude) > 0 and np.isin(selected_index, exclude).all():
                continue
            else:
                selected.append(selected_index)
        return selected
                

class SelectionRanked(Selection):
    """
    Selects two unqiue individuals with a probability inversely proportional 
    to their rank.
    """
    def __init__(self):
        super().__init__()

    def get_single(self):
        order = np.argsort(self.fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(self.fitness)) + 1
        p = ranks / ranks.sum()
        selection_index = np.random.choice(self.indices, 1, replace=False, p=p)
        return selection_index[0]


class SelectionRoulette(Selection):
    """
    Selects two unqiue individuals with a probability proportional to fitness.
    """
    def __init__(self):
        super().__init__()

    def get_single(self):
        p = self.fitness / self.fitness.sum()
        selection_index = np.random.choice(self.indices, 1, replace=False, p=p)
        return selection_index[0]


class SelectionTournament(Selection):
    """
    Selects two unqiue individuals based on tournament selection. A larger
    k (tournament size) gives a greater relative selection probability on the
    best fitness individuals. A k of 2 equals rank-based selection.
    """
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def get_single(self):
        tournament_picks = np.random.choice(
            self.indices, self.k, replace=False)
        picks_fitness = self.fitness[tournament_picks]
        selection_index = tournament_picks[np.argmax(picks_fitness)]
        return selection_index
        

class CrossoverMultipoint:
    """
    Produces offspring from two parents by combining their genomes by 
    stiching them together based on n random indices.
    """
    def __init__(self, n):
        self.n = n
    
    def get_offspring(self, parent_genomes):
        genome_size = len(parent_genomes[0])
        indices = np.arange(genome_size)
        cross_indices = sorted(np.random.choice(indices, self.n, replace=False))
        offspring_genome = np.zeros(genome_size)
        current_parent = np.random.randint(0,2)
        for i in range(genome_size):
            if i in cross_indices:
                current_parent = 1 - current_parent
            offspring_genome[i] = parent_genomes[current_parent][i]
        return offspring_genome
        
    
class CrossoverBitmask:
    """
    Produces offspring from two parents by combining their genomes based 
    on a random bit mask that selects a random parent for each genome.
    """
    def __init__(self):
        pass
 
    def get_offspring(self, parent_genomes):
        genome_size = len(parent_genomes[0])
        bitmask = np.random.randint(0,2,genome_size).astype(bool)
        new_genome = np.zeros(genome_size)
        new_genome[bitmask] = parent_genomes[0][bitmask]
        new_genome[~bitmask] = parent_genomes[0][~bitmask]
        return new_genome


class MutationUniformReplacement:
    """
    Mutates each gene in a genome with probability p. Mutated genes are 
    replaced by a uniformly distributed random variable.
    """
    def __init__(self, p):
        self.p = p

    def mutate_genome(self, genome):
        new_genome = genome.copy()
        for i, gene in enumerate(genome):
            if np.random.uniform(0,1) < self.p:
                new_genome[i] = np.random.uniform(-1,1)
        return new_genome


class MutationAdjustment:
    """
    Mutates each gene in a genome with probability p. Mutated genes are 
    adjusted by substracting or adding a value.
    """
    def __init__(self, p):
        self.p = p

    def mutate_genome(self, genome):
        new_genome = genome.copy()
        for i, gene in enumerate(genome):
            if np.random.uniform(0,1) < self.p:
                new_genome[i] = max(-1, min(1, gene + np.random.uniform(-0.1,0.1)))
        return new_genome
                
# _____________________________________________________ Test and tuning 


def equal_to(genomes):
    """
    For a genome of len n, the sum of all genes should be as close to 
    int(n/2) as possible
    """
    goal_sum = int(genomes.shape[1]/2)
    fitness = np.zeros(len(genomes))
    for i, genome in enumerate(genomes):
        diff = abs(goal_sum - genome.sum())
        fitness[i] = - diff**2
    return fitness


def fitness_test_doubling(genomes):
    """
    Every odd element should be double the amount of the previous element,
    and the sum of all abs values should be maximized.
    """
    fitness = np.zeros(len(genomes))
    for i, genome in enumerate(genomes):
        fit = 0
        for j, gen in enumerate(genome):
            if j % 2 == 0:
                target = gen * 2
            else:
                diff = abs(target - gen)
                penalty = (1 + diff)**2
                fit -= penalty
        fit += abs(genome).sum()
        fitness[i] = fit
   
    return fitness
    

def fitness_test_sorting(genomes):
    """
    Each element should be larger than the previous element but smaller than
    the next element. The steps between echt element should be as close 
    to the step mean as possible.
    """
    fitness = np.zeros(len(genomes))
    for i, genome in enumerate(genomes):
        
        # Element-wise difference 
        diff = np.ediff1d(genome)
        
        # Substract negative diff from fitness (square, punish big)
        non_ascending_penalty = (diff[diff<0]).sum()**2 * 4
        
        # Add positive diff to fitness (sqrt, reward many small instead of one big)
        ascending_reward = math.sqrt((diff[diff>0]).sum()) * 2
  
        # Substract diff deviation from diff mean
        non_mean_penalty = abs(diff - diff.mean()).sum()

        # Low start and high end bonus
        ends_reward = genome[-1] - genome[0]

        fitness[i] = ascending_reward + ends_reward - non_ascending_penalty - non_mean_penalty

    return fitness



def test_evolution(genomes, fitness_func, nr_runs, quantiles, selection, 
                    crossover, mutation, elitism, copy_fract, plot=False):
    """
    Takes a set of genomes, a fitness function and parameters for the 
    genetic algorithm as input, and returns the fitness quantiles for each
    generation in nr_runs evolution runs. can be used to test different 
    parameters and compare results in a plot.
    """
    # Run evolution
    quan_fit = np.zeros((nr_runs, len(quantiles)))
    for i in range(nr_runs):
        fitness = fitness_func(genomes)
        ga = GeneticAlgorithm(
            genomes, fitness, selection, crossover, mutation, elitism, 
            copy_fract, judgement_day_std=0.18)
        genomes = ga.evolve_population()
        quan_fit[i] = np.quantile(fitness, quantiles)
        
        # Plotting intermediate genomes
        if plot and i % int(nr_runs / 200)==0 or i==nr_runs-1:
            fitness = fitness_func(genomes)
            plt.figure(figsize=(8,8))
            ordered_indices = np.array(list(reversed(np.argsort(fitness))))
            plt.imshow(genomes[ordered_indices][:80], vmin=-1, vmax=1)
            plt.title(f'Sorted genomes generation {i}')
            plt.show()
        
    # Plot quantile evolution
    if plot:
        plt.figure(figsize=(8,5))
        for val in quan_fit.T:
            plt.plot(val, color='black', linewidth=1, alpha=0.3)
        plt.title('Fitness quantiles per generation')
        plt.show()
    
    return quan_fit, genomes, fitness_func(genomes)




if __name__ == '__main__':
    
    # Create random population
    pop_size = 200
    genome_size = 25
    genomes = np.random.uniform(-1,1,(pop_size, genome_size))
    
    # Test evolution
    quantiles = np.arange(0.1, 1.1, 0.1)
    quan_fit, genomes, fitness = test_evolution(
        genomes = genomes,
        fitness_func = equal_to,
        nr_runs = 200,
        quantiles = quantiles,
        selection = SelectionTournament(k=3), 
        crossover = CrossoverMultipoint(2), 
        mutation = [MutationUniformReplacement(p=0.05), MutationAdjustment(p=0.15)],
        elitism = 3,
        copy_fract = 0.1,
        plot=True)
    
    
    