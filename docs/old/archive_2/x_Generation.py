import numpy as np
import GA
from matplotlib import pyplot as plt
from math import pi, floor
import Tools
from Color import Color
from pygame_plot.plot import Plot



class Generation:
    
    def __init__(self, pop_size, selection_size, mutation_func, midpoint, bot_type):
        """
        Connects simulation to bots + their autopilots/networks and the GA
        algorithm. Compatible with various bot types. A generation is started
        or stopped from the simulation class. Parameters:
            - pop_size (int): population size, should be multiple of 10.
            - mutation_func(function): function for calculating mutation
            probability based on average population fitness.
            - bot_type (bot class): class of bot to use
            - **kwargs: optional bot parameters
        """
        
        # Parameters
        self.pop_size = pop_size
        self.mutation_func = mutation_func
        self.midpoint = np.array(midpoint)
        self.bot_type = bot_type
        self.selection_size = selection_size
        self.num_elites = 2
        
        # Variables
        self.individuals = []
        self.current_gen = 0
        self.pop_fitness = None
        self.pop_genomes = None
        
        # Generation stats
        self.stats_fitness_avg = []
        self.stats_fitness_max = []
        self.stats_fitness_avg_mov = []
        self.stats_fitness_max_mov = []
        self.stats_mutation_prob = []
        
        # Realtime stats
        self.rt_fitness = []
        
        # Best genome so far
        self.best_genome = None
        self.best_fitness = 0
        
        
    def create(self):
        """
        Creates a new generation with circular positions around the center,
        and initializes bot genomes with either a given set of genomes or at random.
        """
        self.individuals = []
        for i in range(self.pop_size):
            direction = Tools.convert_to_vector(np.random.uniform(0, 2*pi))
            position = np.random.normal(self.midpoint, 180)
            bot = self.bot_type(position, direction)
            self.individuals.append(bot)
        self.individuals[0].color = Color.MCYAN
        
        # Resetting real-time stats
        self.rt_fitness = []
        
    def init_genomes(self, genomes=None):
        """
        Sets the genomes for the current population either randomly or according
        to a given set of genomes.
        """
        if genomes is None:
            for bot in self.individuals:
                bot.random_init()
        else:
            for bot, genome in zip(self.individuals, genomes):
                bot.set_genome(genome)
        
    def update(self, t, food):
        """
        Updates the position and sensor reading for all bots in the generation.
        """
        for bot in self.individuals:
            bot.update(t, food)
            
        # Real-time stats
        fit = np.array([bot.fitness for bot in self.individuals])
        self.rt_fitness.append((fit.min(), fit.mean(), fit.max()))
                      
    def end_generation(self):
        """
        Sorts individuals by fitness, and then extracts genomes and fitnesses.
        """
        self.individuals = sorted(
            self.individuals, key=lambda bot: bot.fitness, reverse=True)
        genome_len = len(self.individuals[0].autopilot.network)
        self.pop_fitness = np.zeros(self.pop_size)
        self.pop_genomes = np.zeros((self.pop_size, genome_len))
        for i, bot in enumerate(self.individuals):
            self.pop_fitness[i] = bot.fitness
            self.pop_genomes[i] = bot.get_genome()
        self.current_gen += 1
        
        # Saving best genome
        if self.pop_fitness.max() > self.best_fitness:
            self.best_genome = self.pop_genomes[0]
            Tools.save_genome(self.best_genome, file='genomes/gen1.txt')
        
    def evolution(self):
        """
        Calls GA functions to create new genomes from the current population.
        """
        elites = self.pop_genomes[:self.num_elites]
        parents = GA.Selection.ranked(self.pop_genomes, self.selection_size)   
        number_parent_pairs = int(len(self.individuals)/2 - self.num_elites/2)
        pairs = GA.Selection.select_pairs(parents, number_parent_pairs)
        offspring = GA.Crossover.n_point(pairs)
        mutation_prob = self.mutation_func(self.pop_genomes.mean())
        mutated_offspring = GA.Mutation.uniform_replacement(offspring, mutation_prob)
        new_genomes = np.concatenate((elites, mutated_offspring))
        return new_genomes
    
    def analyze_generation(self):
        """
        TODO:
            - Generalize function to 'static plot'
            - Make function calleable from the first generation such that
            the plot window is already visible
        """
        
        # Recording stats
        mean_fitness = round(self.pop_fitness.mean(),2)
        max_fitness = round(self.pop_fitness.max(),2)
        self.stats_fitness_avg.append(mean_fitness)
        self.stats_fitness_max.append(max_fitness)
        self.stats_mutation_prob.append(round(self.mutation_func(mean_fitness),3))
        
        # Moving averages
        n = 8
        avg_periods = np.array(self.stats_fitness_avg[-n:]).mean()
        max_periods = np.array(self.stats_fitness_max[-n:]).mean()
        self.stats_fitness_avg_mov.append(avg_periods)
        self.stats_fitness_max_mov.append(max_periods)
        
        # Global plot settings
        plot_pos = (40,25)
        plot_dim = (245,120)
        
        # Generation fitness plot
        x = np.arange(self.current_gen)
        xdomain = (0, x.max()+1)
        ydomain = (min(self.stats_fitness_avg) -0.5 , max(self.stats_fitness_max) + 5)
    
        plot = Plot(xdomain, ydomain, plot_pos, plot_dim)
        plot.border = False
        plot.title = 'Generation progression'
        plot.xaxis.lock_position = True
        plot.xaxis.nr_ticks = 7
        plot.yaxis.nr_ticks = 8
        
        # Adding data to plot
        plot.add_line(x, self.stats_fitness_avg, Color.LBLUE, 1, 'Mean')
        plot.add_line(x, self.stats_fitness_avg_mov, Color.MBLUE, 2, 'Mean/MA')
        plot.add_line(x, self.stats_fitness_max, Color.LGREEN, 1, 'Best')
        plot.add_line(x, self.stats_fitness_max_mov, Color.MGREEN, 2, 'Best/MA')
        
        plot.add_legend('upper left', width=80, border=False)
        
        if x.max() < 6:
            plot.xaxis.set_labels(x, x.astype(str))
        else:
           ticks = np.arange(0, x.max()+1, floor(x.max() / 6))
           plot.xaxis.set_labels(ticks, ticks.astype(str))
            
        
        return plot
    
    
    def real_time_analysis(self, t):
        """
        Return plots for real time analysis/
        
        TODO: Generalize function (also in bot class) to only calling other functions.
        """
        
        
        self.rt_plot_fitness_stats(t)
        bot_plot = self.individuals[0].real_time_analysis(t)
        
        return self.rt_plot, bot_plot
        
        
        
    def rt_plot_fitness_stats(self, t):
        
        # Fitness statistics plots
        x = np.arange(t)
        fitmin, fitmean, fitmax = np.asarray(self.rt_fitness).T
        xdomain = (0, max(x)+1)
        ydomain = (min(fitmin)-1, max(fitmax)+1)
        
        if len(fitmin) < 2:
            self.rt_plot = Plot(xdomain, ydomain, (40,25), (245,120))
            self.rt_plot.border = False
            self.rt_plot.title = 'Real-time fitness'
            self.rt_plot.yaxis.nr_ticks = 8
            self.rt_plot.xaxis.lock_position = True
            self.rt_plot.add_legend('upper left', width=50, border=False)
        
        elif len(fitmin) == 2:
            self.rt_plot.add_line(x, fitmax, Color.MGREEN, 1, 'max')
            self.rt_plot.add_line(x, fitmean, Color.MBLUE, 1, 'mean')
            self.rt_plot.add_line(x, fitmin, Color.MRED, 1, 'min')
        
        elif len(fitmin) > 2: 
            self.rt_plot.elements['max'].add_data([x[-1]], [fitmax[-1]])
            self.rt_plot.elements['mean'].add_data([x[-1]], [fitmean[-1]])
            self.rt_plot.elements['min'].add_data([x[-1]], [fitmin[-1]])
            
        if t < 6:
            self.rt_plot.xaxis.set_labels(x, x.astype(str))
        else:
            ticks = np.arange(0, t+1, floor(t / 6))
            self.rt_plot.xaxis.set_labels(ticks, ticks.astype(str))
           
        self.rt_plot.update_dimensions(xdomain, ydomain)
        
        return self.rt_plot
    
    
    def analyze_generation2(self):
        """
        Creates and updates a graph of generation statistics.
        """
        fig, axs = plt.subplots(2, 2, figsize=(24,16))
        
        # Recording stats
        mean_fitness = round(self.pop_fitness.mean(),2)
        max_fitness = round(self.pop_fitness.max(),2)
        self.stats_fitness_avg.append(mean_fitness)
        self.stats_fitness_max.append(max_fitness)
        self.stats_mutation_prob.append(round(self.mutation_func(mean_fitness),3))
        
        # Moving averages
        n = 8
        avg_periods = np.array(self.stats_fitness_avg[-n:]).mean()
        max_periods = np.array(self.stats_fitness_max[-n:]).mean()
        self.stats_fitness_avg_mov.append(avg_periods)
        self.stats_fitness_max_mov.append(max_periods)
        
        # fitness plot
        generations = np.arange(self.current_gen)
        axs[0, 0].plot(generations, self.stats_fitness_avg, label='Average fitness', 
                 color='steelblue', alpha=0.4)
        axs[0, 0].plot(generations, self.stats_fitness_max, label='Max fitness', 
                 color='maroon', alpha=0.4)
        axs[0, 0].plot(generations, self.stats_fitness_avg_mov, label='Moving average (avg fitness)', 
                 color='steelblue')
        axs[0, 0].plot(generations, self.stats_fitness_max_mov, label='Moving average (max fitness)', 
                 color='maroon')
        axs[0, 0].legend(loc='upper left', fontsize=14).get_frame().set_linewidth(0)
        axs[0, 0].set_xlabel('Generation', fontsize=14)
        axs[0, 0].set_title('Generation progression', fontsize=25)
   
        # Mutation prob plot
        axs[0, 1].plot(generations, self.stats_mutation_prob, color='forestgreen')
        axs[0, 1].set_xlabel('Generation', fontsize=14)
        axs[0, 1].set_title('Mutation probability', fontsize=25)
        
       # Best individual network
        self.individuals[0].autopilot.plotter.pytplot_structure(
            axs[1, 1], node_size=1000, font_size=12)
        axs[1, 1].set_title('Best neural network', fontsize=25)
    
        plt.show()
        
        # Print generation stats
        print(f'gen {self.current_gen}, fit {mean_fitness}, max {max_fitness}')
    
    def status_information(self, t):
        """
        Returns a list of textual current generation information to display 
        on the screen.
        """
        fitness = np.array([bot.fitness for bot in self.individuals])
        info_list = [
            '','SIMULATION STATISTICS',
            f'Generation        {self.current_gen}',
            f'Average fitness   {round(fitness.mean(),2)}',
            f'Max fitness       {fitness.max()}',
            f'Population size   {len(self.individuals)}',
            f'current t:        {t}'
            ]
        return info_list

        







