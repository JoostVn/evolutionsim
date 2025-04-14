import numpy as np
from pygame_plot import PygamePlot, Color
from neural_network import NetworkPlotter
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
        self.pygame_plots = []
        self.matplotlib_plots = []
        self.best_individuals = []
        
    def add_pygame_plot(self, plot):
        self.pygame_plots.append(plot)

    def add_matplotlib_plot(self, plot):
        self.matplotlib_plots.append(plot)

    def set_pygame_plot_dimensions(self, sidebar):
        """
        Sets pygame plot pos and dim constants based on sidebar attributes
        """
        self.plot_pos = (40,25)
        self.plot_dim = np.subtract(sidebar.figure_dim, (55, 45))
        
    def update(self, sidebar=None):
        """
        Updates all statistics and realtime plots. The scope if each pygame 
        plot is used to determine wether or not the plot gets reset, updated
        or drawn. Resetting initializes to PygamePlot object and clears the
        plot data. Updating extends the plot data with the lastest population
        statistics. Draw insertts the plot into the on-screen sidebar. Once a
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
        for p in self.pygame_plots:
            if p.scope == scope:
                p.reset(self.pop, self.sim, self.plot_pos, self.plot_dim)
   
    def update_plots(self, scope):
        """Update all plots in scope"""
        for p in self.pygame_plots + self.matplotlib_plots:
            if p.scope == scope:
                p.update(self.pop, self.sim)
    
    def draw_plots(self, scope, sidebar):
        """Draw all plots pygame plots."""
        for p in self.pygame_plots:
            if p.scope == scope:
                sidebar.insert_plot(p.plot, p.position)
        
    def save_best_genome(self):
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
        self.plot = None
        self.position = position
        self.scope = Analysis.SCOPE_TICK
        self.colors = {'max':Color.GREEN2, 'avg':Color.BLUE2, 'min':Color.RED2}
    
    def reset(self, population, simulation, pos, dim):
        """Reset the pygameplot figure and data series"""
        self.plot = PygamePlot((0,1), (0,1), pos, dim)
        self.plot.set_title('Real-time fitness')
        self.plot.set_xaxis_nr_ticks(4)
        self.plot.set_xaxis_locked(True)
        self.plot.set_yaxis_nr_ticks(6)
        self.plot.set_legend('upper left', width=45, border=False)
        for statkey in self.colors.keys():
            self.plot.add_line([], [], self.colors[statkey], 1, statkey)
  
    def update(self, population, simulation):
        """Add new datapoints to each plot element series."""
        t = simulation.t
        fitness = population.get_fitness(batch_only=True)
        self.plot.elements['max'].add_data(t, np.max(fitness))
        self.plot.elements['avg'].add_data(t, np.mean(fitness))
        self.plot.elements['min'].add_data(t, np.min(fitness))
        ymin = min(self.plot.elements['min'].y)
        ymax = max(self.plot.elements['max'].y) + 1
        self.plot.set_dimensions((0, t+1), (ymin, ymax))



class  PGP_FitnessAll:
    
    """
    Realtime fitness plot that shows ALL current batch fitness values.
    """
    
    def __init__(self, position):
        self.plot = None
        self.position = position
        self.scope = Analysis.SCOPE_TICK
        self.color = Color.GREY4
        self.nr_bots = 0
    
    def reset(self, population, simulation, pos, dim):
        """Reset the pygameplot figure and data series"""
        self.plot = PygamePlot((0,simulation.generation_len), (0,1), pos, dim)
        self.plot.set_title('Real-time fitness')
        self.plot.set_xaxis_nr_ticks(4)
        self.plot.set_xaxis_locked(True)
        self.plot.set_yaxis_nr_ticks(6)
        self.nr_bots = len(population.batch_individuals)
        for i in range(self.nr_bots):
            self.plot.add_line([], [], self.color, 1, f'bot_{i}')
  
    def update(self, population, simulation):
        """Add new datapoints to each plot element series."""
        t = simulation.t
        fitness = population.get_fitness(batch_only=True)
        for i, fit in enumerate(fitness):
            self.plot.elements[f'bot_{i}'].add_data(t, fit)
        self.plot.fit_ydomain(margins=(0,1))



class PGP_Network:
    
    """
    Visualization of the neural network for the debug bot autopilot that shows
    node activations and edge values.
    """
    
    def __init__(self, position):
        self.plot = None
        self.position = position
        self.scope = Analysis.SCOPE_TICK    
        
    def reset(self, population, simulation, pos, dim):
        """Reset the pygameplot figure and data series"""
        
        # Fetch network and use networkplotter to get coordinates
        plotter = NetworkPlotter(population.debug_bot.autopilot.network)
        node_coords, edge_coords = plotter.node_coords, plotter.edge_coords
        
        # Create plot
        self.plot = PygamePlot((0,1), (0,1), pos, dim)
        self.plot.add_network(node_coords, edge_coords, 'network')
        self.plot.set_xaxis_locked(True)
        self.plot.set_yaxis_locked(True)
        self.plot.fit_xdomain(margins=(-0.2,0.2))
        self.plot.fit_ydomain(margins=(-1,1))
        self.plot.set_title('Debug bot network')

        # Input layer names
        input_layer_y = node_coords[1][node_coords[0]==0]
        labels_y = []
        for sens in population.debug_bot.sensors:
            short_name = sens.__class__.__name__[:3]
            sens_names = [f'{short_name}{i}' for i in range(sens.nr_sensors)]
            labels_y += sens_names
        self.plot.set_yaxis_custom_ticks(input_layer_y, labels_y)
        layers_x = list(set(node_coords[0]))
        labels_x = ['input', 'hidden', 'output']
        self.plot.set_xaxis_custom_ticks(layers_x, labels_x)

    def update(self, population, simulation):
        """Set the node and edge values for the network plot."""
        bot = population.debug_bot
        sensor_values = [sens.sensor_values for sens in bot.sensors]
        sensor_values = np.concatenate(sensor_values)
        nvals, evals = bot.autopilot.debug_forward_pass(sensor_values)
        self.plot.elements['network'].set_values(nvals, evals)
    

class PGP_GenomeImage:
    
    """
    Visualization of the population genomes, sorted by the fitness of 
    individuals. Genomes get re-sorted at every batch end.
    """
    
    def __init__(self, position):
        self.plot = None
        self.position = position
        self.scope = Analysis.SCOPE_BATCH    
        
    def reset(self, population, simulation, pos, dim):
        self.plot = PygamePlot((0,10), (0,10), pos, dim)
        self.plot.set_title('Population genomes')
        self.plot.set_xaxis_ticks_disabled()
        self.plot.set_yaxis_ticks_disabled()
        genomes = population.get_genomes() / 2 + 1
        fitness = population.get_fitness()
        sorted_genomes = genomes[np.argsort(fitness)]
        self.plot.add_array_image(Color.WHITE, Color.GREY4, 'arr_img')
        self.plot.elements['arr_img'].set_image(sorted_genomes)

    def update(self, population, simulation):
        genomes = population.get_genomes() / 2 + 1
        fitness = population.get_fitness()
        sorted_genomes = genomes[np.argsort(fitness)]
        self.plot.add_array_image(Color.WHITE, Color.GREY4, 'arr_img')
        self.plot.elements['arr_img'].set_image(sorted_genomes)


class PGP_FitnessEvolution:
    
    """
    Pygame fitness evolution plot with max and average fitness for each
    generation and simple movin average (SMA).
    """

    def __init__(self, position):
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
        self.plot = PygamePlot((0,1), (0,1), pos, dim)
        self.plot.set_title('Fitness evolution')
        self.plot.set_xaxis_nr_ticks(4)
        self.plot.set_xaxis_locked(True)
        self.plot.set_yaxis_nr_ticks(6)
        self.plot.set_legend('upper left', width=65, border=False)
        self.plot.add_line([], [], Color.CYAN3, 1, 'avg')
        self.plot.add_line([], [], Color.CYAN1, 1, 'avgma')
        self.plot.add_line([], [], Color.BLUE3, 1, 'max')
        self.plot.add_line([], [], Color.BLUE1, 1, 'maxma')
    
    def update(self, population, simulation):
        """Computes and updates the population fitness stats and SMA"""
        fitness = population.get_fitness()
        gen = simulation.cur_gen
        self.plot.elements['avg'].add_data(gen, fitness.mean())
        avg_sma = self.calculate_sma(self.plot.elements['avg'].y)
        self.plot.elements['avgma'].add_data(gen, avg_sma[-1])
        self.plot.elements['max'].add_data(gen, fitness.max())
        max_sma = self.calculate_sma(self.plot.elements['max'].y)
        self.plot.elements['maxma'].add_data(gen, max_sma[-1])
        ymin = min(self.plot.elements['avg'].y)
        ymax = max(self.plot.elements['max'].y) + 1
        self.plot.set_dimensions((0, gen+1), (ymin, ymax+1))
        
        
        


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
        ax.plot(self.fit_avg_cur, color='darkblue', alpha=0.3, 
                    linewidth=1, label='fit_avg_cur')
        ax.plot(self.fit_max_cur, color='darkgreen', alpha=0.3, 
                    linewidth=1, label='fit_max_cur')
        ax.plot(self.fit_avg_sma, color='darkblue', alpha=0.7, 
                    linewidth=2, label='fit_avg_sma')
        ax.plot(self.fit_max_sma, color='darkgreen', alpha=0.7, 
                    linewidth=2, label='fit_max_sma')
        ax.legend(frameon=False, loc='upper left')
        ax.set_title(f'Fitness avg/max {self.pop_name}')
        plt.show()
    


