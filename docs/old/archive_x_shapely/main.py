from application import Application, StandaloneSimulation
import numpy as np
from bots import Herbivore, HerbivorePopulation
from objects import FoodSupply, Barriers
import warnings
import analysis
import genetic_algorithm as ga
import tools


"""
TODO:
    - https://geoffboeing.com/2016/10/r-tree-spatial-index-python/
    - Starvation with bot coloring based on its level
    - Pass genetic alg as parameter to population with set_fitness and set_genomes
    as method.
    - Create seperate simulation mode for testing custom genomes
    - Save and load populations with pickle
    - Fix image plot overriding plot borders/
    - Genome diversity plot
    - Batch statistics
    - Only draw and update plots when sidebar is extended
    - Clustering genomes to identify different species
    - Restructure simulation class for more logical placement of next_batch
    and evolve
"""


if __name__ == '__main__':
    
    # General parameters
    warnings.simplefilter(action='ignore', category=FutureWarning)
    STANDALONE, VISUAL, DEBUG_CUSTOM, DEBUG_INACTIVE = 0,1,2,3
    environment_size = np.array((600,500))
    window_size = np.array((1000,700))
    
    # Creating genetic algorithm instance
    selection = ga.SelectionTournament(k=5)
    crossover = ga.CrossoverMultipoint(n=2)
    mutations = [
        ga.MutationUniformReplacement(p=0.03), 
        ga.MutationAdjustment(p=0.03, adjustment_range=(-0.1,0.1))]
    genalg = ga.GeneticAlgorithm(
        selection, crossover, mutations, elitism=2, copy_fract=0.1)

    # Creating object sets and populations
    barr = Barriers(environment_size, quantity=15, size=15, wall_width=2)
    food = FoodSupply(environment_size, quantity=30)
    herbivore_pop = HerbivorePopulation(
        Herbivore, 150, genalg, use_debug_bot=True)
    
    # Creating simulation instance
    sim = StandaloneSimulation(
        object_sets = {'food':food, 'barriers':barr},
        populations = {'herbivore': herbivore_pop},
        generation_len = 300,
        nr_generations = 50,
        nr_batches = 10,
        custom_genome = None,
        verbose = 1)
    
    # Simulation runtype selection
    runtype = STANDALONE

    if runtype == STANDALONE:
        an = analysis.Analysis(herbivore_pop, sim)
        an.add_matplotlib_plot(analysis.PLT_FitnessEvolution())
        sim.state = sim.INITIALIZE
        while sim.state != sim.EXIT:
            sim.update()
            an.update()
        
    elif runtype == VISUAL:
        an = analysis.Analysis(herbivore_pop, sim)
        an.add_pygame_plot(analysis.PGP_FitnessAll(0))
        an.add_pygame_plot(analysis.PGP_Network(1))
        an.add_pygame_plot(analysis.PGP_FitnessEvolution(2))
        an.add_pygame_plot(analysis.PGP_GenomeImage(3))
        app = Application(window_size, sim, an)
        app.main_loop()
        
    elif runtype == DEBUG_INACTIVE:
        herbivore_pop = HerbivorePopulation(Herbivore, 1, True)
        sim = StandaloneSimulation(
            object_sets = {'food':food, 'barriers':barr},
            populations = {'herbivore':herbivore_pop},
            generation_len = 100000,
            nr_generations = 1,
            nr_batches = 1)
        an = analysis.Analysis(herbivore_pop, sim)
        an.add_pygame_plot(analysis.PGP_FitnessStats(0))
        an.add_pygame_plot(analysis.PGP_Network(1))
        app = Application(window_size, sim, an)
        app.main_loop()
    
    elif runtype == DEBUG_CUSTOM:
        custom_genome = tools.genomefile_open()
        herbivore_pop = HerbivorePopulation(Herbivore, 1, True)
        sim = StandaloneSimulation(
            object_sets = {'food':food, 'barriers':barr},
            populations = {'herbivore':herbivore_pop},
            generation_len = 100000,
            nr_generations = 1,
            nr_batches = 1,
            custom_genome = custom_genome)
        an = analysis.Analysis(herbivore_pop, sim)
        an.add_pygame_plot(analysis.PGP_FitnessStats(0))
        an.add_pygame_plot(analysis.PGP_Network(1))
        app = Application(window_size, sim, an)
        app.main_loop()
        
    