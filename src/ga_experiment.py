# src/ga_experiment.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, Any, Tuple

# [FIX] Use absolute imports to the subdirectory 'ga_feature_select'
from src.ga_feature_select.ga_core import create_initial_population, selection_tournament
from src.ga_feature_select.operators import crossover, mutation
from src.ga_feature_select.fitness import calculate_fitness

# --- Configuration ---
# Set Seaborn for better looking plots
sns.set_theme(style="whitegrid")

def run_ga(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    params: Dict[str, Any], 
    save_path: str = "results/plots"
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Runs the full Genetic Algorithm process for feature selection.
    
    Parameters
    ----------
    X_train, y_train, X_test, y_test : pd.DataFrame, pd.Series
        Training and testing data for fitness evaluation.
    params : Dict[str, Any]
        Dictionary containing GA parameters: 'pop_size', 'generations', 
        'crossover_rate', 'mutation_rate', 'alpha', 'penalty_weight'.
    save_path : str
        Directory to save the fitness evolution plot.
        
    Returns
    -------
    Tuple[np.ndarray, float, Dict[str, Any]]
        (best_mask, best_fitness, history_dict)
    """
    
    # Extract parameters
    POP_SIZE = params.get('pop_size', 100)
    GENERATIONS = params.get('generations', 50)
    CROSSOVER_RATE = params.get('crossover_rate', 0.8)
    MUTATION_RATE = params.get('mutation_rate', 0.01)
    ALPHA = params.get('alpha', 0.9)
    PENALTY_WEIGHT = params.get('penalty_weight', 0.1)
    
    N_FEATURES = X_train.shape[1]
    
    # 1. Initialization
    current_population = create_initial_population(POP_SIZE, N_FEATURES)
    best_chromosome = current_population[0].copy()
    best_fitness = -1.0 
    
    # History tracking for analysis (Section 5 requirement)
    fitness_history = {'best_fitness': [], 'avg_fitness': []}

    # 2. Main GA Loop
    for gen in range(GENERATIONS):
        
        # A. Fitness Evaluation (Using the function from fitness.py)
        # Note: We pass ALPHA and PENALTY_WEIGHT to the fitness function
        # [0] is used to retrieve the fitness score from the (fitness, accuracy) tuple
        fitness_values = np.array([
            calculate_fitness(chrom, X_train, y_train, X_test, y_test, ALPHA, PENALTY_WEIGHT)[0] 
            for chrom in current_population
        ])
        
        # B. Update Best Chromosome
        current_best_idx = np.argmax(fitness_values)
        current_best_fitness = fitness_values[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = current_population[current_best_idx].copy()
            
        # C. Track History
        fitness_history['best_fitness'].append(current_best_fitness)
        fitness_history['avg_fitness'].append(np.mean(fitness_values))
        
        # D. Selection (Using the function from ga_core.py)
        # Note: Using selection_tournament
        selected_parents = selection_tournament(current_population, fitness_values, POP_SIZE * 2)
        
        # E. Crossover and Mutation (Using functions from operators.py)
        next_population = []
        # Elitism: Preserve the best chromosome for the next generation
        next_population.append(best_chromosome.copy()) 
        
        # Crossover/Mutation loop
        for i in range(0, POP_SIZE, 2):
            if i >= len(selected_parents) - 1:
                break
                
            parent1 = selected_parents[i]
            parent2 = selected_parents[i+1]
            
            # Crossover (using function from operators.py)
            child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)
            
            # Mutation (using function from operators.py)
            child1 = mutation(child1, MUTATION_RATE)
            child2 = mutation(child2, MUTATION_RATE)
            
            next_population.extend([child1, child2])

        # Ensure population size is maintained (handling odd pop_size and elitism)
        current_population = np.array(next_population[:POP_SIZE])

    # 3. Analysis and Plotting (Section 5 requirement)
    plot_fitness_evolution(fitness_history, GENERATIONS, save_path)
    
    return best_chromosome, best_fitness, fitness_history

def plot_fitness_evolution(history: Dict[str, list], generations: int, save_path: str):
    """
    Generates and saves the fitness evolution plot using Matplotlib/Seaborn.
    
    Parameters
    ----------
    history : Dict[str, list]
        Dictionary containing 'best_fitness' and 'avg_fitness' lists over generations.
    generations : int
        Total number of generations to set the plot limits.
    save_path : str
        Directory to save the plot.
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plotting the data
    plt.plot(history['best_fitness'], label='Best Fitness', color=sns.color_palette("rocket")[0], linewidth=2)
    plt.plot(history['avg_fitness'], label='Average Fitness', color=sns.color_palette("rocket")[2], linestyle='--')
    
    # Customizing the plot for professional look
    plt.title('Fitness Evolution Over Generations (Genetic Algorithm)', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Fitness Value (Model Accuracy)', fontsize=14)
    plt.xlim(0, generations)
    
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot
    plot_filename = os.path.join(save_path, "fitness_evolution.png")
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"\nAnalysis complete. Plot saved to: {plot_filename}")