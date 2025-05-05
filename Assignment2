# -*- coding: utf-8 -*-
"""
Evolution Strategies (μ + λ)-ES for Multimodal Optimization
-----------------------------------------------------------
Objective:
Minimize the multimodal Rastrigin function defined as:
    f(x) = A * n + Σ [xi² - A * cos(2πxi)], for i = 1 to n

Where:
- A = 10 (constant)
- n = dimensionality of the input vector x
- x ∈ [-5.12, 5.12]^n

The Rastrigin function is known for its large number of regularly distributed
local minima and a single global minimum at f(x) = 0 when x = [0, ..., 0].

Optimization Goal:
Design and implement an Evolution Strategy (μ + λ)-ES to minimize f(x),
effectively escaping local minima and converging to the global minimum in
a high-dimensional search space.

Author: Iko Tan
Date: 2025-05-04
"""


# IMPORT STATEMENTS
import numpy as np
import matplotlib.pyplot as plt


# GLOBAL CONSTANTS FOR MUTATION ADAPTATION
ADAPT_CONST = 0.86          # Factor used for adapting mutation step size (sigma)
SUCCESS_THRESHOLD = 0.2     # 1/5 success rule threshold for mutation success
MUTATION_PROBABILITY = 0.1  # Mutation probability per gene (10%)


# EVOLUTION STRATEGY CLASS DEFINITION
class EvolutionStrategy:
    def __init__(self, num_parameters=10, population_size=20, sigma=0.1, objective=0):
        """
        Initialize the Evolution Strategy parameters and create the initial population.

        Args:
            num_parameters (int): Dimensionality of the problem (number of genes)
            population_size (int): Number of individuals in the population
            sigma (float): Initial mutation step size (std. deviation for Gaussian noise)
            objective (float): Target function value (usually 0 for minimization)
        """
        self.num_parameters = num_parameters
        self.population_size = population_size
        self.sigma = sigma
        self.objective = objective
        self.max_error = 0.001                  # Convergence threshold for fitness error
        self.bounds = (-5.12, 5.12)             # Bounds of the Rastrigin search space
        self.max_generations = 1000             # Maximum number of generations to run

        # Initialize population: each individual is a randomly generated vector in the search space
        self.population = [
            np.random.uniform(self.bounds[0], self.bounds[1], self.num_parameters)
            for _ in range(self.population_size)
        ]

        # Evaluate initial fitness values for the population
        self.fitness = [self.target(ind) for ind in self.population]
        self.best_solution = self.population[np.argmin(self.fitness)]
        self.best_fitness = min(self.fitness)
        self.error = abs(self.objective - self.best_fitness)
        self.fitness_history = [self.best_fitness]

    # Objective Function
    def target(self, x):
        A = 10
        return A * len(x) + sum([xi**2 - A * np.cos(2 * np.pi * xi) for xi in x])

    # Mutation Operator
    def mutate(self, individual):
        """
        Generate a mutated offspring by applying Gaussian noise to an individual.
        Each gene has a chance given by MUTATION_PROBABILITY to mutate.
        """
        new_individual = individual.copy()
        for i in range(len(new_individual)):
            if np.random.rand() < MUTATION_PROBABILITY:
                new_individual[i] += np.random.normal(0, self.sigma)
                # Ensure that the mutated gene stays within defined bounds
                new_individual[i] = np.clip(new_individual[i], self.bounds[0], self.bounds[1])
        return new_individual

    # Main Evolution Loop
    def evolve(self):
        """
        The main evolutionary loop implementing the (μ + λ) Evolution Strategy.

        Process:
            1. Generate one offspring per parent via mutation.
            2. Evaluate the fitness of the offspring.
            3. Combine parents and offspring and select the best individuals.
            4. Adapt the mutation step-size (sigma) periodically using a 1/5 success rule.
            5. Repeat until convergence criteria or maximum generations are reached.
        """
        generation = 0
        gen_sigma = 10  # Interval (in generations) to adapt the mutation step size
        successful_mutations = 0

        # Evolutionary loop: continues until termination criteria are met
        while generation < self.max_generations and self.error > self.max_error:
            offspring = []

            # Generate offspring: Each parent produces one offspring by mutation
            for parent in self.population:
                child = self.mutate(parent)
                offspring.append(child)

            # Evaluate fitness of all offspring
            offspring_fitness = [self.target(child) for child in offspring]

            # Combine parents and offspring for selection (μ + λ strategy)
            combined = self.population + offspring
            combined_fitness = self.fitness + offspring_fitness

            # Select the top 'population_size' individuals based on fitness
            indices = np.argsort(combined_fitness)[:self.population_size]
            self.population = [combined[i] for i in indices]
            self.fitness = [combined_fitness[i] for i in indices]

            # Track the best solution found so far and count successful mutations
            current_best = min(self.fitness)
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = self.population[np.argmin(self.fitness)]
                successful_mutations += 1

            # Update fitness history and convergence error
            self.fitness_history.append(self.best_fitness)
            self.error = abs(self.objective - self.best_fitness)
            generation += 1

            # Adapt mutation step size every 'gen_sigma' generations
            if generation % gen_sigma == 0:
                success_rate = successful_mutations / gen_sigma
                if success_rate > SUCCESS_THRESHOLD:
                    self.sigma /= ADAPT_CONST  # Increase exploration
                else:
                    self.sigma *= ADAPT_CONST  # Increase exploitation
                successful_mutations = 0  # Reset the count for the next interval

        self.actual_generations = generation  # Record total generations run

    # Result Display and Visualization
    def display_result(self):
        """
        Display the final optimization results and plot the convergence history.
        """
        print("===== Optimization Summary =====")
        print(f"Final Best Solution Vector:\n{self.best_solution}")
        print(f"Final Fitness Value: {self.best_fitness:.6f}")
        print(f"Error from Target: {self.error:.6f}")
        print(f"Total Generations Run: {self.actual_generations}")
        print(f"Final Mutation Step Size (Sigma): {self.sigma:.5f}")

        # Plot fitness convergence over generations
        plt.plot(self.fitness_history)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Fitness Convergence Over Generations")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


#MAIN EXECUTION

if __name__ == '__main__':
    es = EvolutionStrategy(num_parameters=10, population_size=20)
    es.evolve()
    es.display_result()
