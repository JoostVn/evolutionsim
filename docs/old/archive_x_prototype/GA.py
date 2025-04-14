import numpy as np
from itertools import combinations  
from matplotlib import pyplot as plt



"""
TODO:
    - Elitism
    - n-point crossover
    - Mating probability
    - More flexibility in pop and offspring sizes
"""

class Bot:
    
    def __init__(self, fitness, genome):
        self.fitness = fitness
        self.genome = genome
    
    def get_genome(self):
        return self.genome



class Selection:
    
    @staticmethod
    def roulette(fitnesses, genomes, num_parents):
        """
        Select individuals with probability proportional to their fitness.
        Fitnesses: np.arrray of all individual fitnesses (best to worst)
        genomes: np.arrray of genomes for each individual (best to worst)
        """
        inv_indices = np.arange(genomes.shape[0])
        p = fitnesses / fitnesses.sum()
        selection = np.random.choice(inv_indices, num_parents, replace=False, p=p)
        parents = genomes[selection]
        return parents
    
    @staticmethod
    def ranked(genomes, num_parents):
        """
        Select individuals with probability proportional to their rank.
        genomes: np.arrray of genomes for each individual (best to worst)
        """
        inv_indices = np.arange(genomes.shape[0])
        ranks = inv_indices + 1
        p = 1 / (ranks + 1)
        p = p / p.sum()
        selection = np.random.choice(inv_indices, num_parents, replace=False, p=p)
        parents = genomes[selection]
        return parents
        
    @staticmethod
    def tournament(individuals, num_parents, k):
        """
        Select winner from tournaments of k individuals parents == num_parents. 
        individuals: ordered numpy array of (fitness, genes) for each individual.
        """
        pass
    
    @staticmethod
    def select_pairs(parents, num_pairs):
        """
        Creates a list of all possible combinations of parents and 
        select num_pairs random pairs of parents.
        Returns array of pairs of individuals.
        parents: ordered numpy array of (fitness, genes) for each parent.
        """
        parent_indices = np.arange(parents.shape[0])
        pairs = np.array(list(combinations(parent_indices, r=2)))
        select_indices = np.random.choice(pairs.shape[0], num_pairs, replace=False)
        selected_parent_pairs = parents[pairs[select_indices]]
        return selected_parent_pairs
        
    
    
class Crossover:
    
    @staticmethod
    def n_point(parent_pairs):
        """
        Performs n point crossover for an array of parent pairs.
        parent_pairs: Array of set of two genomes.
        """
        offspring = np.array(
            [Crossover.n_point_ONE(pair) for pair in parent_pairs])
        nr_offspring = parent_pairs.shape[0] * 2
        genome_lenght = parent_pairs[0][0].shape[0]
        return offspring.reshape((nr_offspring, genome_lenght))
    
    @staticmethod
    def n_point_ONE(parents):
        """
        Perform one n-points crossover for a parent pair. n is determined
        based on the genome size. 
        returns: np.array of two new genomes
        parents: np.arrray of genomes for each individual
        """
        genome_len = parents.shape[1]
        n = int(genome_len/4)
        points = np.random.choice(np.arange(1,genome_len), n, replace=False)
        offspring = np.zeros((2, genome_len))
        switch = 0
        for i, gene in enumerate(parents.T):
            if i in points:
                switch = 1 - switch
            offspring[0][i] = gene[switch]
            offspring[1][i] = gene[1-switch]
        return offspring
       
    @staticmethod
    def single_point(parent_pairs):
        """
        Performs single point crossover for an array of parent pairs.
        parent_pairs: Array of set of two genomes.
        """
        offspring = np.array(
            [Crossover.single_point_ONE(pair) for pair in parent_pairs])
        nr_offspring = parent_pairs.shape[0] * 2
        genome_lenght = parent_pairs[0][0].shape[0]
        return offspring.reshape((nr_offspring, genome_lenght))
     
    @staticmethod
    def single_point_ONE(parents):
        """
        single 1-point crossover. Generates a random integer, which slices the 
        genome of both parents. Two offspring are produced by combining
        the slices of the parent pairs.
        returns: np.array of two new genomes
        parents: np.arrray of genomes for each individual
        """
        gene1 = parents[0]
        gene2 = parents[1]
        point = np.random.randint(1, gene1.shape[0]-2)
        child1 = np.concatenate((gene1[:point], gene2[point:]))
        child2 = np.concatenate((gene2[:point], gene1[point:]))
        return child1, child2
    
    
class Mutation:
    
    @staticmethod
    def uniform_replacement(individuals, pm):
        """
        Applies uniform replacements to all individuals.
        """
        new_individuals = np.array(
            [Mutation.uniform_replacement_ONE(i, pm) for i in individuals])
        return new_individuals
    
    @staticmethod
    def uniform_replacement_ONE(genome, pm):
        """
        Replaces random genes with new uniform(-1,1) genes with mutation
        chance probability pm. Pm should be small: 0.01 = 0.05
        """
        new_genome = genome.copy()
        for i, gene in enumerate(genome):
            r = np.random.uniform(0,1)
            if r < pm:
                new_genome[i] = np.random.uniform(-1,1)
        return new_genome
        
    
    
if __name__ == '__main__':
    
    P = 20
    
    # Creating test population
    pop_fitness = np.random.randint(0,10,P)
    pop_genomes = np.array([np.random.uniform(-1,1,8).round(2) for bot in range(P)])
    
    # Testing functions
    parents = Selection.ranked(pop_genomes, 4)   
    pairs = Selection.select_pairs(parents, 2)
    offspring = Crossover.n_point(pairs)
    offspring = Crossover.single_point(pairs)
    mutated_offspring = Mutation.uniform_replacement(offspring, 0.05)
    
    
    # Selection probability histograms
    
    
    def roulette(fitnesses, genomes, num_parents):
        inv_indices = np.arange(genomes.shape[0])
        p = fitnesses / fitnesses.sum()
        selection = np.random.choice(inv_indices, num_parents, replace=False, p=p)
        return selection
    
    def ranked(genomes, num_parents):
        inv_indices = np.arange(genomes.shape[0])
        ranks = inv_indices + 1
        p = 1 / (ranks + 1)
        p = p / p.sum()
        selection = np.random.choice(inv_indices, num_parents, replace=False, p=p)
        return selection
    
    
    
    
    n = 1000
    rank = []
    roul = []
    
    for i in range(1000):
        rank += list(ranked(pop_genomes, 4))
        roul += list(roulette(pop_fitness, pop_genomes, 4))


    plt.figure(figsize=(8,5))
    plt.hist(rank, density=True, bins=20, label='ranked', alpha=0.5)
    plt.hist(roul, density=True, bins=20, label='roulette', alpha=0.5)
    plt.legend()
    plt.show()
 
    
