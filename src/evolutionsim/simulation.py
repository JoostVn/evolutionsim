import time
import numpy as np


class StandaloneSimulation:
    
    #TODO: dict with states and methods instead of long if case
    #TODO: rename to simulation
    #TODO: docstring, i/o documentation
    #TODO: simulation.draw (only called when used in visual app)
    
    HOLD = 0
    INITIALIZE = 1
    START = 2
    RUN = 3
    BATCH_END = 4
    GEN_END = 5
    EXIT = 6
    
    def __init__(self, object_sets, populations, generation_len, 
                 nr_generations, nr_batches, custom_genome=None):
        
        # Parameters
        self.object_sets = object_sets
        self.populations = populations
        self.generation_len = generation_len
        self.nr_generations = nr_generations
        self.nr_batches = nr_batches
        self.custom_genome = custom_genome

        # State and timing variables
        self.state = self.HOLD
        
        # Time variables
        self.t = 0
        self.cur_gen = 0
        self.cur_batch = 0
        
        # Statistics
        self.prev_state = self.state
        self.stats_gentime = 0
        self.stats_gentime_log = []
   
    def update(self):
        
        self.prev_state = self.state
        
        if self.state == self.HOLD:
            self.state_hold()
        elif self.state == self.INITIALIZE:
            self.state_initialize()
        elif self.state == self.START:
            self.state_start()
        elif self.state == self.RUN:
            self.state_run()
        elif self.state == self.BATCH_END:
            self.state_batch_end()
        elif self.state == self.GEN_END:
            self.state_gen_end()
        elif self.state == self.EXIT:
            self.state_exit()
    
    def state_hold(self):
        """
        Passive state that waits for the simulation to be started.
        """
        pass
        
    def state_initialize(self):
        """
        Only triggered once per simulation. Any setup actions are applied here.
        """
        self.t = 0
        self.cur_gen = 0
        self.cur_batch = 0
        self.stats_gentime = 0
        self.stats_gentime_log = []
        self.state = self.START
    
    def state_start(self):
        """
        Triggered at the start of each new simulation run. Creates a random
        population for the first generation, and then either calls next 
        batches or evolves the population. Also resets objects.
        """
        self.t = 0
        
        print(f'> G{self.cur_gen}B{self.cur_batch}')
        
        # Initialize objects
        for objset in self.object_sets.values():
            objset.initialize()
        
        # Initialize populations
        for pop in self.populations.values():
            
            # Random population for first generation
            if self.cur_gen == 0 and self.cur_batch == 0:
                pop.initialize(self.object_sets)
                self.stats_gentime = time.time()
                if not self.custom_genome is None:
                    pop.individuals[0].set_genome(self.custom_genome)

            # Evolve population for first batch of each generation
            elif self.cur_gen > 0 and self.cur_batch == 0:
                new_genomes = pop.evolve()
                pop.initialize(self.object_sets)
                pop.set_genomes(new_genomes)
                self.stats_gentime = time.time()
            pop.next_batch(self.nr_batches)
        
        self.state = self.RUN
        
    def state_run(self):
        """
        Updates all populations and objects in the simulation.
        """
        self.t += 1
        for pop in self.populations.values():
            pop.update(self.object_sets, self.populations)
        for objset in self.object_sets.values():
            objset.update(self.object_sets, self.populations)
        if self.t == self.generation_len:
            self.state = self.BATCH_END
        
    def state_batch_end(self):
        """
        Triggered at the end of each simulation run. Determines whether a new
        batch or a new generation should be initialized.
        """
        self.cur_batch += 1
        if self.cur_batch < self.nr_batches:
            self.state = self.START
        elif self.cur_batch == self.nr_batches:
            if self.cur_gen == self.nr_generations - 1:
                self.state = self.EXIT
            else:
                self.cur_gen += 1
                self.cur_batch = 0
                self.state = self.GEN_END
        
    def state_gen_end(self):
        """
        End the generation and print generation statistics.
        """
        generation_time =  time.time() - self.stats_gentime
        self.stats_gentime_log.append(generation_time)
        sec_gen = round(np.mean(self.stats_gentime_log),2)
        min_left = round((sec_gen * (self.nr_generations - self.cur_gen))/60, 2)
        print(f'> G{self.cur_gen}/{self.nr_generations} completed ({sec_gen} sec/gen, {min_left} min remaining)')
        self.state = self.START
                    
    def state_exit(self):
        """
        Triggered when the last generation is completed.
        """
        print('simulation comleted.')
        self.state = self.HOLD



