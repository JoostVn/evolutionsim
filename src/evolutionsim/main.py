from application import SimApplication
import numpy as np
from bots import Herbivore, FlockBot, HerbivorePopulation, FlockPopulation
from objects import FoodSupply, Barriers
import analysis
from algorithms.genalg import GeneticAlgorithm, selection, crossover, mutation, disaster
from simulation import StandaloneSimulation


def create_herbivore_environment(environment_size, nr_bots, generation_len,
                                  empty_genome=False):

    # Genetic algorithm
    genalg = GeneticAlgorithm(
        selection = selection.Tournament(k=3),
        crossover = crossover.Multipoint(1),
        mutations = [
            mutation.UniformReplacement(p=0.02),
            mutation.Adjustment(p=0.3, adjustment_domain=(-0.1,0.1))],
        disaster = disaster.SuperMutation(
            similarity_threshold = 0.85,
            mutations = [mutation.Adjustment(p=1, adjustment_domain=(-0.3,0.3))]),
        num_elites = 3)

    # Object sets and populations
    barr = Barriers(environment_size, quantity=8, size=180, wall_width=2)
    food = FoodSupply(environment_size, quantity=140)
    herbivore_pop = HerbivorePopulation(
        Herbivore, nr_bots, genalg, use_debug_bot=True)

    # Simulation instance
    sim = StandaloneSimulation(
        object_sets = {'food':food, 'barriers':barr},
        populations = {'herbivore': herbivore_pop},
        generation_len = generation_len,
        nr_generations = 500,
        nr_batches = 1,
        custom_genome = None)

    # Clear genome fOr debug bot (used for debugging)
    if empty_genome:
        sim.custom_genome = herbivore_pop.bot((0,0),0).get_genome()

    return herbivore_pop, sim



def run_standalone(population, simulation):

    # Analysis instance
    an = analysis.Analysis(population, simulation)
    an.add_matplotlib_analysis(analysis.PLT_GeneSimilarity())
    an.add_matplotlib_analysis(analysis.PLT_FitnessEvolution())

    # Run simualtion
    simulation.state = simulation.INITIALIZE
    while simulation.state != simulation.EXIT:
        simulation.update()
        an.update()



def run_visual(population, simulation, window_size):

    # Analysis instance
    an = analysis.Analysis(population, simulation)
    an.add_pygame_analysis(analysis.PGP_FitnessAll(0))
    an.add_pygame_analysis(analysis.PGP_Network(1))
    an.add_pygame_analysis(analysis.PGP_FitnessEvolution(2))
    an.add_pygame_analysis(analysis.PGP_GenomeImage(3))

    # Application instance and main loop
    app = SimApplication(window_size, simulation, an)
    app.run()



if __name__ == '__main__':

    environment_size = np.array((1500,1500))
    window_size = np.array((1000,700))

    if False:
        pop, sim = create_herbivore_environment(
            environment_size, nr_bots=1, generation_len=30000, empty_genome=True)
    else:
        pop, sim = create_herbivore_environment(
            environment_size, nr_bots=70, generation_len=500, empty_genome=False)

    run_visual(pop, sim, window_size)
    #run_standalone(pop, sim)


