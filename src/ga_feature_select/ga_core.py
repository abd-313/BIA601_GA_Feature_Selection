# src/ga_feature_select/ga_core.py

import numpy as np
import random
from typing import List, Tuple, Union, Any 

# [FIX] Use safe relative import within the ga_feature_select package
from .fitness import calculate_fitness


def create_individual(N_FEATURES: int) -> np.ndarray:
    """Creates a single binary chromosome (individual) for the GA population."""
    
    chromosome = np.random.randint(0, 2, N_FEATURES)
    # Ensure at least one feature is selected
    if np.sum(chromosome) == 0:
        chromosome[random.randint(0, N_FEATURES - 1)] = 1
    return chromosome


def create_initial_population(POP_SIZE: int, N_FEATURES: int) -> List[np.ndarray]:
    """Initializes the first generation of binary chromosomes."""
    
    population = [create_individual(N_FEATURES) for _ in range(POP_SIZE)]
    return population


def selection_tournament(population: List[np.ndarray], fitness_values: np.ndarray, num_parents: int) -> List[np.ndarray]:
    """
    Selects parents using Tournament Selection (k=3).
    
    Parameters
    ----------
    population : List[np.ndarray]
        The current list of chromosomes.
    fitness_values : np.ndarray
        Array of fitness scores corresponding to the population.
    num_parents : int
        Number of parents to select.
        
    Returns
    -------
    List[np.ndarray]
        List of selected parent chromosomes.
    """
    
    POP_SIZE = len(population)
    parents = []        
 
    for _ in range(num_parents):
        # Select 3 random individuals for the tournament
        tournament_indices = np.random.choice(range(POP_SIZE), size=3, replace=False) 
        # Find the best fitness among them
        best_idx = tournament_indices[np.argmax(fitness_values[tournament_indices])]
        parents.append(population[best_idx])
    return parents