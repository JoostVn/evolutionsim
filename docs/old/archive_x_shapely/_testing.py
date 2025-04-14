import numpy as np



def std_similarity(genomes, domain):
    """
    Calculates the mean standard deviation, min/max normalized to be between 0 
    and 1 based on the given (min, max) domain, and inverted such that 1 is 
    the most similarity and 0 is the least similarity.
    """
    mean_std = genomes.std(axis=0).mean()
    scaled_mean_std = mean_std / np.std(domain)
    similarity = 1 - scaled_mean_std
    return similarity


# _______________________________ TEST 1: minimum similarity
v1 = np.full(100, -1)
v2 = np.full(100, 1)

genomes = np.vstack((v1,v2))
sim1 = std_similarity(genomes, (-1,1))


# _______________________________ TEST 2: maximum similarity


def fitness_func(genomes):
    pos = genomes >= 0
    sum_pos = genomes[pos].sum(axis=1)
    sum_neg = -genomes[~pos].sum(axis=1)
    return max(sum_pos, sum_neg)
    
    


genomes = np.random.uniform(-1,1,(100,8))


def ga_similarity_test(nr_runs):
    

    """
    Simple genetic algorithm optimization that maximizes  the absolute value
    of a genome. Can have
    """
    




