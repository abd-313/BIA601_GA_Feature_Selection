# src/ga_feature_select/ga_core.py

import numpy as np
import random
# Import fitness function from the same package
from .fitness import calculate_fitness

class GAFeatureSelector:
    
    def __init__(self, X_train, X_test, y_train, y_test, total_features,
                 population_size=50, max_generations=50, mutation_rate=0.01,
                 crossover_rate=0.8, alpha=0.99, penalty_weight=0.01):
        
        # Data and feature counts
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.N_FEATURES = total_features
        
        # GA parameters
        self.POP_SIZE = population_size
        self.MAX_GEN = max_generations
        self.MUT_RATE = mutation_rate
        self.CROSS_RATE = crossover_rate
        
        # Fitness weights
        self.ALPHA = alpha
        self.PENALTY_WEIGHT = penalty_weight
        
        # GA state
        self.population = []
        self.best_chromosome = None
        self.best_fitness = -np.inf
        
    def initialize_population(self):
        """Initializes the first generation of binary chromosomes."""
        self.population = [self.create_individual() for _ in range(self.POP_SIZE)]

    def create_individual(self):
        
        chromosome = np.random.randint(0, 2, self.N_FEATURES)
        # Ensure at least one feature is selected
        if np.sum(chromosome) == 0:
            chromosome[random.randint(0, self.N_FEATURES - 1)] = 1
        return chromosome
    
    def selection(self, fitness_values, num_parents):
        
        parents = []        
 
        for _ in range(num_parents):
            # Select 3 random individuals for the tournament
            tournament_indices = np.random.choice(range(self.POP_SIZE), size=3, replace=False)
            # Find the best fitness among them
            best_idx = tournament_indices[np.argmax(fitness_values[tournament_indices])]
            parents.append(self.population[best_idx])
        return parents
