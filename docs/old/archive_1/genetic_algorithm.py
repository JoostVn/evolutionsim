import numpy as np
from matplotlib import pyplot as plt



class GeneticAlgorithm:
    
    def __init__(self, genomes, fitness, selection, crossover, mutation,
                 elitism, copy_fraction):
        
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
        self.copy_fraction = copy_fraction

    def evolve_population(self):
        
        self.selection.set_population(self.genomes, self.fitness)
        
        # Fetch the best indivuals as elites
        elite_indices = np.argsort(self.fitness)[-self.elitism:]
        elite_genomes = self.genomes[elite_indices]

        # Copy randomly selected inviduals without mutation or offspring
        nr_copy = int(self.copy_fraction * self.popsize)
        copy_indices = self.selection.get_n_unique(nr_copy, exclude=elite_indices)
        copy_genomes = self.genomes[copy_indices]
        
        # Selection, Crossover, Mutation
        nr_offspring = self.popsize - nr_copy - self.elitism
        offspring_genomes = []
        while len(offspring_genomes) < nr_offspring:
            parent_genomes = self.genomes[self.selection.get_n_unique(2)]
            offspring = self.crossover.get_offspring(parent_genomes)
            mutated_offspring = self.mutation.mutate_genome(offspring)
            offspring_genomes.append(mutated_offspring)
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
    Produces offspring from two parents by combining their genomes and 
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
        
    
class CrossoverUniform:
    
    pass
    


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





def selection_distribution(selectors, P, n):
    
    
    plt.figure(figsize=(7,5))
    
    for current_selector in selectors:
        
        # Create new population and selection for each n, and log selected rank
        selected_rank = np.zeros(n, dtype=int)
        for i in range(n):
            
            # Create population with exponential distributed fitness
            fitness = np.random.exponential(10, P).round(2)
            genomes = np.random.uniform(-1,1,(P,8)).round(2)
            current_selector.set_population(genomes, fitness)

            # Select parent and get/log its rank
            parent_index = current_selector.get_single()
            parent_rank_inv = np.where(np.argsort(fitness)==parent_index)[0][0]
            parent_rank = P - parent_rank_inv - 1
            selected_rank[i] = parent_rank
        
        # Plot the selection probability for an individual based on their rank
        name = current_selector.__class__.__name__
        plt.hist(selected_rank, bins=np.arange(P), label=name,
                 alpha=0.5, density=True)
                     
                     
    plt.title('Selection PDF based on fitness rank')
    plt.xlabel('Fitness rank of selected parent')
    plt.ylabel('Probability density of selection')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    
    # Create random population
    pop_size = 100
    genome_size = 200
    
    ga = GeneticAlgorithm(
        genomes = np.random.uniform(-1,1,(pop_size, genome_size)), 
        fitness = np.random.exponential(10, pop_size).round(2), 
        selection = SelectionTournament(k=3), 
        crossover = CrossoverMultipoint(n=3), 
        mutation = MutationUniformReplacement(p=0.05),
        elitism = 3,
        copy_fraction = 0.1)
    
    new_genomes = ga.evolve_population()
    
    
    
    
    
    ga.get_genome_index(ga.genomes[15])
    
    
    
    # Testing functions
    selectors = [
        SelectionRanked(), 
        SelectionRoulette(), 
        SelectionTournament(k=3)]
    selection_distribution(selectors, P=50, n=50)


