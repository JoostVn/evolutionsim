import numpy as np
from matplotlib import pyplot as plt
import math

class GeneticAlgorithm:
    
    
    def __init__(self, selection, crossover, mutations, elitism, copy_fract=0, 
                 social_disaster_sim=1):
        """
        Parameters
        ----------
        selection : Selection instance
            Initialized instance of one of the selection classes. Selects 
            individuals from the population based on fitness. Used for parent
            selection and copy selection.
        crossover : Crossover instance
            Initialized instance of one of the crossover classes. Generates
            offspring from selected parent individuals.
        mutations : List of mutation instances
            Mutation instances are passed as a list an applied in order. 
        elitism : Integer
            The number of elite individuals to be passed to a new generation
            without evolving.
        copy_fract : Float between 0 and 1
            The fraction of individuals to be copied over to a new generation
            without evolving. The default is 0.
        social_disaster_sim : Float betwen 0 and 1, optional
            The similiraty threshold that triggers a social disaster. 
        """
        self.selection = selection
        self.crossover = crossover
        self.mutations = mutations
        self.elitism = elitism
        self.copy_fract = copy_fract
        self.social_disaster_sim = social_disaster_sim

    def evolve_population(self, genomes, fitness, genome_domain):
        """

        Parameters
        ----------
        genomes : numpy array
            Genomes arrays with shape (population_size, genome length). Each
            genome is represented by a float.
        fitness : numpy array
            1d Array with shape (population_size) in the same order als genomes.
        genome_domain : tuple
            The (min, max) values of genes, used for determining normalized 
            population similarity. .

        Returns
        -------
        new_population_genomes : numpy array
            The evolved generation genomes.
        """
        # Fetch the best indivuals as elites
        elite_indices = np.argsort(fitness)[-self.elitism:]
        elite_genomes = genomes[elite_indices]
        
        # Social disaster
        similarity = self.get_similarity(genomes, genome_domain)
        if similarity > self.social_disaster_sim: 
            new_population_genomes = self.social_disaster(genomes, elite_indices)
            return new_population_genomes

        # Copy randomly selected inviduals without mutation or offspring
        self.selection.set_population(genomes, fitness)
        nr_copy = int(self.copy_fract * len(genomes))
        copy_indices = self.selection.get_n_unique(nr_copy, exclude=elite_indices)
        copy_genomes = genomes[copy_indices]
        
        # Selection, Crossover, Mutation
        nr_offspring = len(genomes) - nr_copy - self.elitism
        offspring_genomes = []
        while len(offspring_genomes) < nr_offspring:
            parent_genomes = genomes[self.selection.get_n_unique(2)]
            offspring = self.crossover.get_offspring(parent_genomes)
            
            # Apply mutation for each given mutation instance.
            for m in self.mutations:
                offspring = m.mutate_genome(offspring)
           
            offspring_genomes.append(offspring)
        offspring_genomes = np.array(offspring_genomes)
        
        # Combining elites, copies and offspring into new population
        new_population_genomes = np.vstack((
            elite_genomes, copy_genomes, offspring_genomes))
        return new_population_genomes
            
    def get_similarity(self, genomes, genome_domain):
        """
        Calculates the mean standard deviation, min/max normalized to be between 0 
        and 1 based on the given (min, max) domain, and inverted such that 1 is 
        the most similarity and 0 is the least similarity.
        """
        mean_std = genomes.std(axis=0).mean()
        scaled_mean_std = mean_std / np.std(genome_domain)
        similarity = 1 - scaled_mean_std
        return similarity
    
    def social_disaster(self, genomes, elite_indices):
        """
        If the standard deviation is under a given limit, performs extreme
        mutation over the full population with the exception of elites. The
        goals of this is escaping local minima.
        """
        genome_indices = np.arange(len(genomes))
        non_elites = ~np.in1d(genome_indices, elite_indices)
        disaster_genomes = np.zeros(genomes[non_elites].shape)
        for i, genome in enumerate(genomes[non_elites]):
            for j in range(5):
                for m in self.mutations:
                    genome = m.mutate_genome(genome)
            disaster_genomes[i] = genome
        genomes[non_elites] = disaster_genomes
        return genomes
        




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
    def __init__(self, p, adjustment_range):
        self.p = p
        self.range = adjustment_range

    def mutate_genome(self, genome):
        new_genome = genome.copy()
        for i, gene in enumerate(genome):
            if np.random.uniform(0,1) < self.p:
                adjusted_genome = gene + np.random.uniform(*self.range)
                new_genome[i] = max(-1, min(1, adjusted_genome))
        return new_genome
                
# _____________________________________________________ Test and tuning 

    

def fitness_test_sorting(genomes):
    """
    Each element should be larger than the previous element but smaller than
    the next element. The steps between echt element should be as close 
    to the step mean as possible.
    """
    fitness = np.zeros(len(genomes))
    for i, genome in enumerate(genomes):
        element_diff = np.ediff1d(genome)
        non_ascending_penalty = (element_diff[element_diff<0]).sum()**2 * 4
        ascending_reward = math.sqrt((element_diff[element_diff>0]).sum()) * 2
        non_mean_penalty = abs(element_diff - element_diff.mean()).sum()
        ends_reward = genome[-1] - genome[0]
        fitness[i] = ascending_reward + ends_reward - non_ascending_penalty - non_mean_penalty
    return fitness


def fitness_test_maxposneg(genomes):
    fitness = np.zeros(len(genomes))
    for i, genome in enumerate(genomes):
        sum_pos = genome[genome >= 0].sum()
        sum_neg = -genome[genome < 0].sum()
        fitness[i] = max(sum_pos, sum_neg) / len(genome)
    return fitness


def genome_plot(genomes, fitness, i):
        plt.figure(figsize=(8,8))
        ordered_indices = np.array(list(reversed(np.argsort(fitness))))
        plt.imshow(genomes[ordered_indices][:80], vmin=-1, vmax=1)
        plt.title(f'Sorted genomes generation {i}')
        plt.show()


def quantile_plot(fitness_quantiles):
        plt.figure(figsize=(8,5))
        for val in fitness_quantiles.T:
            plt.plot(val, color='black', linewidth=1, alpha=0.3)
        plt.title('Fitness quantiles per generation')
        plt.show()



def test_evolution(genomes, genalg, fitness_func, nr_gens, quantiles, plot=False):
    """
    Takes a set of genomes, a fitness function and parameters for the 
    genetic algorithm as input, and returns the fitness quantiles for each
    generation in nr_gens evolution runs. can be used to test different 
    parameters and compare results in a plot.
    """
    # Run evolution
    fitness_quantiles = np.zeros((nr_gens, len(quantiles)))
    for i in range(nr_gens):
        fitness = fitness_func(genomes)
        if plot:
            genome_plot(genomes, fitness, i)
        genomes = genalg.evolve_population(genomes, fitness, (-1,1))
        fitness_quantiles[i] = np.quantile(fitness, quantiles)
        
    # Plot quantile evolution
    if plot:
        quantile_plot(fitness_quantiles)
    return fitness_quantiles, genomes, fitness_func(genomes)


def compare_genalgs(initial_genomes, genalg_list, fitness_func, nr_gens, nr_runs):
    """
    Compare mean best results of multiple different GA's over multiple runs
    """
    quantiles = np.arange(0.2, 1.2, 0.2)
    best_results = np.zeros((len(genalg_list),nr_runs, nr_gens))
    
    for i, ga in enumerate(genalg_list):
        for j in range(nr_runs):
            print(i,j)
            genomes = initial_genomes.copy()
            quan_fit, genomes, fitness = test_evolution(
                genomes,ga, fitness_func, nr_gens, quantiles)
            best_genomes = np.flipud(quan_fit.T)[0]
            best_results[i,j] = best_genomes
        
    # Plotting results
    mean_best = best_results.mean(axis=1)
    plt.figure(figsize=(8,5))
    for i, best_evolution in enumerate(mean_best):
        plt.plot(best_evolution, label=f'GA_{i}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    
    # Create random population
    pop_size = 200
    genome_size = 40
    genomes = np.random.uniform(-1,1,(pop_size, genome_size))
    
    # Compare mean best results of different GA's over multiple runs
    ga1 = GeneticAlgorithm(
        selection = SelectionTournament(k=3),
        crossover = CrossoverMultipoint(2), 
        mutations = [
            MutationUniformReplacement(p=0.05), 
            MutationAdjustment(p=0.15, adjustment_range=(-0.1, 0.1))], 
        elitism = 3, 
        copy_fract = 0.1, 
        social_disaster_sim=0.6)
        
    ga2 = GeneticAlgorithm(
        selection = SelectionTournament(k=3),
        crossover = CrossoverMultipoint(2), 
        mutations = [
            MutationUniformReplacement(p=0.05), 
            MutationAdjustment(p=0.15, adjustment_range=(-0.1, 0.1))], 
        elitism = 3, 
        copy_fract = 0.1, 
        social_disaster_sim=0.18)
    
    ga3 = GeneticAlgorithm(
        selection = SelectionTournament(k=3),
        crossover = CrossoverMultipoint(2), 
        mutations = [
            MutationUniformReplacement(p=0.05), 
            MutationAdjustment(p=0.15, adjustment_range=(-0.1, 0.1))], 
        elitism = 3, 
        copy_fract = 0.1, 
        social_disaster_sim=0.2)
    
    
    
    #compare_genalgs(genomes, [ga1, ga2, ga3], fitness_test_sorting, 60, 40)
    
    quan_fit, genomes, fitness = test_evolution(
        genomes,ga1, fitness_test_maxposneg, 60, np.arange(0.2, 1.2, 0.2), plot=True)
    
    
    
