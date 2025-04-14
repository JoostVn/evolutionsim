import numpy as np
from pygametools.color.color import Color
from pygametools.plotting import Canvas
from pygametools.plotting.plots import Line, Network, ArrayImage
from algorithms.neural.plotting import NetworkPlotter
from matplotlib import pyplot as plt



class Analysis:

    """
    Analysis object that holds all recorded population data and pygame/pyplot
    plots. An Analysis instance is linked to a population and a simulation.
    It gets its functionality from plots that are added in with the add_plots
    methods. The Analysis instance intself does not hold any data, it just
    triggers the plot updates and draw methods. Each plot has a scope that
    determines the point at which the plot is updated.
    """

    # Data scopes
    SCOPE_TICK = 1          # Batch statistics, updated each tick
    SCOPE_BATCH = 2         # Generation statistics, updated each batch end
    SCOPE_GEN = 3           # Population statistics, updated gen end

    def __init__(self, population, simulation):
        self.pop = population
        self.sim = simulation
        self.pygame_ans = []
        self.matplotlib_ans = []
        self.best_individuals = []

    def add_pygame_analysis(self, analysis):
        self.pygame_ans.append(analysis)

    def add_matplotlib_analysis(self, analysis):
        self.matplotlib_ans.append(analysis)

    def set_pygame_plot_dimensions(self, sidebar):
        """
        Sets pygame plot pos and dim constants based on sidebar attributes
        """
        self.plot_pos = (40,25)
        self.plot_dim = np.subtract(sidebar.figure_dim, (55, 45))

    def update(self, sidebar=None):
        #TODO: state handling should be done in plots
        """
        Updates all statistics and realtime plots. The scope if each pygame
        plot is used to determine wether or not the plot gets reset, updated
        or drawn. Resetting initializes to PygamePlot object and clears the
        plot data. Updating extends the plot data with the lastest population
        statistics. Draw inserts the plot into the on-screen sidebar. Once a
        plot is inserted, it gets drawn without having to re-insert it. so,
        static plots only need to be inserted once. Dynamic plots need to be
        inserted again after every data update.
        """

        # Initial plot reset for first batch of first generation
        state_prestart = self.sim.t == self.sim.cur_batch == self.sim.cur_gen == 0
        state_init = state_prestart and self.sim.state == self.sim.RUN
        if state_init and self.sim.state == self.sim.RUN:
            self.reset_plots(self.SCOPE_TICK)
            self.reset_plots(self.SCOPE_BATCH)
            self.reset_plots(self.SCOPE_GEN)

        # Reset plots
        if self.sim.state == self.sim.BATCH_END:
            self.reset_plots(self.SCOPE_TICK)
        if self.sim.state == self.sim.GEN_END:
            self.reset_plots(self.SCOPE_BATCH)

        # Update plots
        if self.sim.state == self.sim.RUN:
            self.update_plots(self.SCOPE_TICK)
        if state_init or self.sim.state == self.sim.BATCH_END:
            self.update_plots(self.SCOPE_BATCH)
        if state_init or self.sim.state == self.sim.GEN_END:
            self.update_plots(self.SCOPE_GEN)

        # Draw plots
        if self.sim.state == self.sim.RUN:
            self.draw_plots(self.SCOPE_TICK, sidebar)
        if state_init or self.sim.state == self.sim.BATCH_END:
            self.draw_plots(self.SCOPE_BATCH, sidebar)
        if state_init or self.sim.state == self.sim.GEN_END:
            self.draw_plots(self.SCOPE_GEN, sidebar)


        # Save best genome
        if self.sim.state == self.sim.GEN_END:
           self. save_best_genome()

    def reset_plots(self, scope):
        """Reset all plots in scope. Only required for pygame plots"""
        for an in self.pygame_ans:
            if an.scope == scope:
                an.reset(self.pop, self.sim, self.plot_pos, self.plot_dim)

    def update_plots(self, scope):
        """Update all plots in scope"""
        for an in self.pygame_ans + self.matplotlib_ans:
            if an.scope == scope:
                an.update(self.pop, self.sim)

    def draw_plots(self, scope, sidebar):
        """Draw all plots pygame plots."""
        for an in self.pygame_ans:
            if an.scope == scope:
                sidebar.insert_canvas(an.canvas, an.position)

    def save_best_genome(self):
        #TODO: move to plots
        """Save the best pop individual and the recent best genome"""

        # Saving best individual of current generation
        fitness = self.pop.get_fitness()
        best_ind = self.pop.individuals[np.argmax(fitness)]
        self.best_individuals.append(best_ind)

        # Saving best genome of the last 10 generations
        recent_best = self.best_individuals[-10:]
        best_index = np.argmax([ind.fitness for ind in recent_best])
        best_genome = recent_best[best_index].get_genome()
        filename = 'genomes/{}_G{}I{}T{}.txt'.format(
            self.pop.__class__.__name__,
            self.sim.nr_generations,
            len(self.pop.individuals),
            self.sim.generation_len)
        with open(filename, 'w') as file:
            file.writelines('\n'.join(best_genome.astype(str)))



class  PGP_FitnessStats:

    """
    Realtime fitness plot that shows the min, average, and max fitness values
    for each population batch.
    """

    def __init__(self, position):
        self.canvas = None
        self.position = position
        self.scope = Analysis.SCOPE_TICK

    def reset(self, population, simulation, pos, dim):
        """Reset the pygameplot figure and data series"""
        self.canvas = Canvas((0,1), (0,1), pos, dim)
        self.canvas.set_title('Real-time fitness')
        self.canvas.set_xaxis_nr_ticks(4)
        self.canvas.set_xaxis_locked(True)
        self.canvas.set_yaxis_nr_ticks(6)
        self.canvas.set_legend('upper left', width=45, border=False)

        # Creating lines
        self.line_max = Line(self.canvas, 'max', Color.GREEN2, 1)
        self.line_avg = Line(self.canvas, 'avg', Color.BLUE2, 1)
        self.line_min = Line(self.canvas, 'min', Color.RED2, 1)

    def update(self, population, simulation):
        """Add new datapoints to each plot element series."""
        t = simulation.t
        fitness = population.get_fitness(batch_only=True)
        self.line_max.add_data(t, np.max(fitness))
        self.line_avg.add_data(t, np.mean(fitness))
        self.line_min.add_data(t, np.min(fitness))
        ymin = min(self.line_min.y) - 1
        ymax = max(self.line_max.y) + 1
        self.canvas.set_dimensions((0, t+1), (ymin, ymax))



class  PGP_FitnessAll:

    """
    Realtime fitness plot that shows ALL current batch fitness values.
    """

    def __init__(self, position):
        self.canvas = None
        self.position = position
        self.scope = Analysis.SCOPE_TICK
        self.color = Color.GREY4
        self.nr_bots = 0

    def reset(self, population, simulation, pos, dim):
        """Reset the pygameplot figure and data series"""
        self.canvas = Canvas((0,simulation.generation_len), (0,1), pos, dim)
        self.canvas.set_title('Real-time fitness')
        self.canvas.set_xaxis_nr_ticks(4)
        self.canvas.set_xaxis_locked(True)
        self.canvas.set_yaxis_nr_ticks(6)
        self.nr_bots = len(population.batch_individuals)
        self.bot_lines = []
        for i in range(self.nr_bots):
            line = Line(self.canvas, f'bot_{i}', Color.random_dull(), 1)
            self.bot_lines.append(line)

    def update(self, population, simulation):
        """Add new datapoints to each plot element series."""
        t = simulation.t
        fitness = population.get_fitness(batch_only=True)
        for fit, line in zip(fitness, self.bot_lines):
            line.add_data(t, fit)
        self.canvas.fit_ydomain(margins=(0,1))



class PGP_Network:

    """
    Visualization of the neural network for the debug bot autopilot that shows
    node activations and edge values.
    """

    def __init__(self, position):
        self.canvas = None
        self.position = position
        self.scope = Analysis.SCOPE_TICK

    def reset(self, population, simulation, pos, dim):
        """Reset the pygameplot figure and data series"""

        # Fetch network and use networkplotter to get coordinates
        plotter = NetworkPlotter(population.debug_bot.autopilot.network)
        node_coords, edge_coords = plotter.node_coords, plotter.edge_coords

        # Create canvas
        self.canvas = Canvas((0,1), (0,1), pos, dim)
        self.canvas.set_xaxis_locked(True)
        self.canvas.set_yaxis_locked(True)
        self.canvas.set_title('Debug bot network')

        # Create network plot
        self.plot = Network(self.canvas, 'network', node_coords, edge_coords)
        self.canvas.fit_xdomain(margins=(-0.2,0.2))
        self.canvas.fit_ydomain(margins=(-1,1))

        # Input layer names and y axis labels
        input_layer_y = node_coords[1][node_coords[0]==0]
        labels_y = []
        for sens in population.debug_bot.sensors:
            short_name = sens.__class__.__name__[:3]
            sens_names = [f'{short_name}{i}' for i in range(sens.nr_sensors)]
            labels_y += sens_names
        self.canvas.set_yaxis_custom_ticks(input_layer_y, labels_y)

        # Layer descriptions and x axis labels
        layers_x = list(set(node_coords[0]))
        labels_x = ['input', 'hidden', 'output']
        self.canvas.set_xaxis_custom_ticks(layers_x, labels_x)

    def update(self, population, simulation):
        """Set the node and edge values for the network plot."""
        bot = population.debug_bot
        sensor_values = [sens.sensor_values for sens in bot.sensors]
        sensor_values = np.concatenate(sensor_values)
        nvals, evals = bot.autopilot.debug_forward_pass(sensor_values)
        self.plot.set_values(nvals, evals)


class PGP_GenomeImage:

    """
    Visualization of the population genomes, sorted by the fitness of
    individuals. Genomes get re-sorted at every batch end.
    """

    def __init__(self, position):
        self.canvas = None
        self.position = position
        self.scope = Analysis.SCOPE_BATCH

    def reset(self, population, simulation, pos, dim):
        self.canvas = Canvas((0,1), (0,1), pos, dim)
        self.canvas.set_title('Population genomes')
        self.canvas.set_xaxis_ticks_disabled()
        self.canvas.set_yaxis_ticks_disabled()
        self.plot = ArrayImage(self.canvas, 'Genomes')

    def update(self, population, simulation):
        genomes = population.get_genomes() / 2 + 1
        fitness = population.get_fitness()
        sorted_genomes = genomes[np.argsort(fitness)]
        self.plot.set_image_grayscale(sorted_genomes, Color.GREY4)



class PGP_FitnessEvolution:

    """
    Pygame fitness evolution plot with max and average fitness for each
    generation and simple movin average (SMA).
    """

    def __init__(self, position):
        self.canvas = None
        self.scope = Analysis.SCOPE_GEN
        self.position = position
        self.pop_name = None
        self.n = 8

    def calculate_sma(self, data):
        """Returns the simple moving average of a data series."""
        arr = np.array(data)
        n = self.n
        padding = np.full(n-1, np.nan)
        sma = [arr[i-n+1:i+1].mean() for i in range(n-1, len(arr))]
        return np.concatenate((padding, sma))

    def reset(self, population, simulation, pos, dim):
        """Reset the pygameplot figure and data series"""
        self.canvas = Canvas((0,1), (0,1), pos, dim)
        self.canvas.set_title('Fitness evolution')
        self.canvas.set_xaxis_nr_ticks(4)
        self.canvas.set_xaxis_locked(True)
        self.canvas.set_yaxis_nr_ticks(6)
        self.canvas.set_legend('upper left', width=65, border=False)
        self.plot_avg = Line(self.canvas, 'avg', Color.CYAN3, 1)
        self.plot_avgma = Line(self.canvas, 'avgma', Color.CYAN1, 1)
        self.plot_max = Line(self.canvas, 'max', Color.BLUE3, 1)
        self.plot_maxma = Line(self.canvas, 'maxma', Color.BLUE1, 1)

    def update(self, population, simulation):
        """Computes and updates the population fitness stats and SMA"""
        fitness = population.get_fitness()
        gen = simulation.cur_gen
        self.plot_avg.add_data(gen, fitness.mean())
        avg_sma = self.calculate_sma(self.plot_avg.y)
        self.plot_avgma.add_data(gen, avg_sma[-1])
        self.plot_max.add_data(gen, fitness.max())
        max_sma = self.calculate_sma(self.plot_max.y)
        self.plot_maxma.add_data(gen, max_sma[-1])
        ymin = min(self.plot_avg.y)
        ymax = max(self.plot_max.y) + 1
        self.canvas.set_dimensions((0, gen+1), (ymin, ymax+1))



class PLT_Quantiles:

    """
    Matplotlib fitness quantiles plot.
    """

    def __init__(self):
        self.scope = Analysis.SCOPE_GEN
        self.pop_name = None
        self.quants = np.arange(0.2,1.2,0.2)
        self.data = []

    def update(self, population, simulation):
        fitness = population.get_fitness()
        quant_vals = np.quantile(fitness, self.quants)
        self.data.append(quant_vals)
        self.pop_name = population.__class__.__name__
        self.draw()

    def draw(self):
        fig, ax = plt.subplots()
        for q in np.transpose(self.data):
            ax.plot(q, color='black', alpha=0.35)
        secax = ax.secondary_yaxis('right')
        secax.set_yticks(self.data.T[-1])
        secax.set_yticklabels([f'{int(round(100*q,0))}%' for q in self.quants])
        ax.set_title(f'Fitness quantiles {self.pop_name}')
        plt.show()



class PLT_FitnessEvolution:

    """
    Matplotlib fitness evolution plot with max and average fitness for each
    generation and simple movin average (SMA).
    """

    def __init__(self):
        self.scope = Analysis.SCOPE_GEN
        self.pop_name = None
        self.n = 8
        self.fit_avg_cur = []
        self.fit_max_cur = []
        self.fit_avg_sma = []
        self.fit_max_sma = []

    def calculate_sma(self, data):
        """Returns the simple moving average of a data series."""
        arr = np.array(data)
        n = self.n
        padding = np.full(n-1, np.nan)
        sma = [arr[i-n+1:i+1].mean() for i in range(n-1, len(arr))]
        return np.concatenate((padding, sma))

    def update(self, population, simulation):
        """Computes and updates the population fitness stats and SMA"""
        fitness = population.get_fitness()
        self.fit_avg_cur.append(fitness.mean())
        self.fit_max_cur.append(fitness.max())
        self.fit_avg_sma = self.calculate_sma(self.fit_avg_cur)
        self.fit_max_sma = self.calculate_sma(self.fit_max_cur)
        self.pop_name = population.__class__.__name__
        self.draw()

    def draw(self):
        """Creates and draws the matplotlib plot"""
        fig, ax = plt.subplots()
        ax.plot(self.fit_avg_cur[1:], color='darkblue', alpha=0.3,
                    linewidth=1, label='fit_avg_cur')
        ax.plot(self.fit_max_cur[1:], color='darkgreen', alpha=0.3,
                    linewidth=1, label='fit_max_cur')
        ax.plot(self.fit_avg_sma[1:], color='darkblue', alpha=0.7,
                    linewidth=2, label='fit_avg_sma')
        ax.plot(self.fit_max_sma[1:], color='darkgreen', alpha=0.7,
                    linewidth=2, label='fit_max_sma')
        ax.legend(frameon=False, loc='upper left')
        ax.set_title(f'Fitness avg/max {self.pop_name}')
        plt.show()


class PLT_GeneSimilarity:

    """
    Matplotlib genetic similarity plot that visualizes
    """

    def __init__(self):
        self.scope = Analysis.SCOPE_GEN
        self.pop_name = None
        self.similarity = []
        self.hline_y = 0

    def update(self, population, simulation):
        """Computes and updates the population fitness stats and SMA"""
        self.pop_name = population.__class__.__name__
        self.hline_y = population.genalg.disaster.similarity_threshold
        genomes = population.get_genomes()
        sim = population.genalg.disaster.get_genome_similarity(genomes, (-1,1))
        self.similarity.append(sim)
        self.draw()

    def draw(self):
        """Creates and draws the matplotlib plot"""
        fig, ax = plt.subplots()
        ax.plot(self.similarity[1:], color='darkblue', alpha=0.7,
                linewidth=2, label='Gene similarity')
        ax.set_title(f'Gene similarity {self.pop_name}')
        ax.set_ylim((0,1))

        # Grid lines and horizontal social disaster line
        sim = np.array(self.similarity)
        for y in np.arange(0.6, 1, 0.05):
            ax.axhline(
                y, 0, len(sim), color='grey', alpha=0.2, label='SD trigger', linewidth=1)
        ax.axhline(
            self.hline_y, 0, len(sim), color='red', alpha=0.5, label='SD trigger')

        # Vertical lines marking social disaster points
        social_disasters = np.arange(len(sim))[sim >= self.hline_y]
        for x in social_disasters:
            ax.axvline(x-1, 0, 1, color='darkred', alpha=0.3, linewidth=1)


        plt.show()
