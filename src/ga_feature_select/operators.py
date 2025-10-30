"""
Operators for Genetic Algorithm-based Feature Selection.

This module implements:
- Uniform crossover (recommended for binary feature masks)
- Bit-flip mutation with configurable rate

All operators assume chromosomes are 1D NumPy arrays of dtype bool or int (0/1),
with length equal to the total number of features (e.g., 561 in the HAR dataset).
"""

import numpy as np
import random

def crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform uniform crossover between two parent chromosomes with a given probability.

    If crossover does not occur (based on crossover_rate), parents are returned unchanged.

    Parameters
    ----------
    parent1, parent2 : np.ndarray
        Binary or boolean feature selection masks of shape (n_features,).
    crossover_rate : float
        Probability of performing crossover (default: 0.8).

    Returns
    -------
    child1, child2 : np.ndarray
        Two offspring (may be identical to parents if crossover was skipped).
    """
    if np.random.rand() > crossover_rate:
        return parent1.copy(), parent2.copy()

    if parent1.shape != parent2.shape:
        raise ValueError("Parents must have the same shape.")

    mask = np.random.rand(parent1.size) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)

    # Preserve input dtype (bool or int)
    return child1.astype(parent1.dtype), child2.astype(parent2.dtype)


def mutation(chromosome: np.ndarray, mutation_rate: float = 0.01) -> np.ndarray:
    """
    Apply bit-flip mutation to a chromosome.

    Each gene is flipped (0 ↔ 1) independently with probability `mutation_rate`.

    Parameters
    ----------
    chromosome : np.ndarray
        Binary or boolean feature selection mask.
    mutation_rate : float
        Probability of flipping each bit (default: 0.02).

    Returns
    -------
    mutated_chromosome : np.ndarray
        A new mutated chromosome of the same shape and dtype.
    """
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0, 1].")

    mutated = chromosome.copy()
    flip_mask = np.random.rand(chromosome.size) < mutation_rate

    if mutated.dtype == bool:
        mutated[flip_mask] = ~mutated[flip_mask]
    else:
        mutated[flip_mask] = 1 - mutated[flip_mask]

    return mutated