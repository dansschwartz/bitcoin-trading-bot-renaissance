"""
Step 14: Genetic Weight Optimizer
Evolves bot signal weights using a genetic algorithm based on historical performance.
"""

import sqlite3
import json
import logging
import random
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Tuple, Optional

class GeneticWeightOptimizer:
    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.population_size = 20
        self.generations = 5
        self.mutation_rate = 0.1
        
    async def run_optimization_cycle(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """Runs one generation of evolution and returns the best weights."""
        self.logger.info("Starting Genetic Weight Optimization (Step 14)...")
        
        # 1. Fetch historical performance data
        performance_data = self._fetch_performance_data()
        if len(performance_data) < 50:
            self.logger.info("Insufficient data for genetic optimization. Need at least 50 decisions.")
            return current_weights
            
        # 2. Create initial population
        population = self._initialize_population(current_weights)
        
        # 3. Evolve for N generations
        best_weights = current_weights
        for gen in range(self.generations):
            # Calculate fitness for each individual
            fitness_scores = [self._calculate_fitness(ind, performance_data) for ind in population]
            
            # Select best individuals
            best_idx = np.argmax(fitness_scores)
            best_weights = population[best_idx]
            
            self.logger.info(f"Generation {gen}: Best Fitness = {fitness_scores[best_idx]:.4f}")
            
            # Create next generation
            population = self._evolve_population(population, fitness_scores)
            
        self.logger.info("Genetic optimization complete.")
        return best_weights

    def _fetch_performance_data(self) -> List[Dict[str, Any]]:
        """Fetch decisions and trades to evaluate performance."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Fetch last 500 decisions
            cursor.execute("SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 500")
            rows = cursor.fetchall()
            conn.close()
            
            data = []
            for row in rows:
                item = dict(row)
                item['reasoning'] = json.loads(item['reasoning'])
                data.append(item)
            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch data for optimization: {e}")
            return []

    def _initialize_population(self, seed_weights: Dict[str, float]) -> List[Dict[str, float]]:
        population = [seed_weights]
        for _ in range(self.population_size - 1):
            individual = {}
            total = 0
            for k in seed_weights.keys():
                val = max(0.01, seed_weights[k] + random.uniform(-0.1, 0.1))
                individual[k] = val
                total += val
            # Normalize
            for k in individual:
                individual[k] /= total
            population.append(individual)
        return population

    def _calculate_fitness(self, weights: Dict[str, float], data: List[Dict[str, Any]]) -> float:
        """
        Calculates fitness based on correlation with REALIZED PnL from labels table.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Fetch labels for the decisions we are analyzing
            decision_ids = [str(item['id']) for item in data]
            if not decision_ids:
                return 0.0
                
            query = f"SELECT * FROM labels WHERE decision_id IN ({','.join(decision_ids)})"
            cursor.execute(query)
            labels = {row['decision_id']: dict(row) for row in cursor.fetchall()}
            conn.close()
            
            if not labels:
                # Fallback to internal simulation if no labels exist yet
                return self._simulate_internal_fitness(weights, data)
                
            total_score = 0.0
            processed_count = 0
            
            for item in data:
                label = labels.get(item['id'])
                if not label:
                    continue
                    
                reasoning = item.get('reasoning', {})
                contributions = reasoning.get('signal_contributions', {})
                if not contributions:
                    continue
                
                # Recalculate weighted signal with NEW weights
                new_weighted_signal = 0.0
                stored_weights = reasoning.get('weights', self.bot_signal_weights if hasattr(self, 'bot_signal_weights') else {})
                
                for k, weight in weights.items():
                    # Approximate original signal value: contribution / weight
                    # (In production, store 'raw_signals' in decisions table)
                    orig_w = stored_weights.get(k, 0.1)
                    orig_signal = contributions.get(k, 0.0) / (orig_w + 1e-6)
                    new_weighted_signal += orig_signal * weight
                
                # Fitness is the alignment with actual realized return direction
                actual_ret = label['ret_pct']
                # alignment = signal * actual_return (Positive if both same sign)
                # Scale signal to match magnitude of return or use sign
                alignment = np.sign(new_weighted_signal) * np.sign(actual_ret) if abs(actual_ret) > 0.0005 else 0.0
                
                # Bonus for magnitude match
                magnitude_match = 1.0 - abs(np.clip(new_weighted_signal, -1, 1) - np.clip(actual_ret * 100, -1, 1))
                
                total_score += (alignment + magnitude_match)
                processed_count += 1
                
            return total_score / processed_count if processed_count > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Fitness calculation failed: {e}")
            return 0.0

    def _simulate_internal_fitness(self, weights: Dict[str, float], data: List[Dict[str, Any]]) -> float:
        """Original proxy fitness when real labels are unavailable."""
        simulated_pnl = 0.0
        for item in data:
            reasoning = item.get('reasoning', {})
            contributions = reasoning.get('signal_contributions', {})
            if not contributions: continue
            
            new_weighted_signal = 0.0
            for k, weight in weights.items():
                orig_weight = 0.1 # Simplified
                orig_signal = contributions.get(k, 0.0) / (orig_weight + 1e-6)
                new_weighted_signal += orig_signal * weight
            
            target = 1.0 if item['action'] == 'BUY' else -1.0 if item['action'] == 'SELL' else 0.0
            simulated_pnl += (1.0 - abs(new_weighted_signal - target))
            
        return simulated_pnl / len(data) if data else 0.0

    def _evolve_population(self, population: List[Dict[str, float]], fitness: List[float]) -> List[Dict[str, float]]:
        next_gen = []
        
        # Elitism: keep top 2
        sorted_indices = np.argsort(fitness)[::-1]
        next_gen.append(population[sorted_indices[0]])
        next_gen.append(population[sorted_indices[1]])
        
        while len(next_gen) < self.population_size:
            # Selection
            parent1 = self._select_parent(population, fitness)
            parent2 = self._select_parent(population, fitness)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child)
            
            next_gen.append(child)
            
        return next_gen

    def _select_parent(self, population, fitness) -> Dict[str, float]:
        # Tournament selection
        tournament = random.sample(list(zip(population, fitness)), 3)
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(self, p1, p2) -> Dict[str, float]:
        child = {}
        for k in p1.keys():
            child[k] = p1[k] if random.random() > 0.5 else p2[k]
        # Re-normalize
        total = sum(child.values())
        for k in child:
            child[k] /= total
        return child

    def _mutate(self, ind) -> Dict[str, float]:
        if random.random() < self.mutation_rate:
            k = random.choice(list(ind.keys()))
            ind[k] = max(0.01, ind[k] + random.uniform(-0.05, 0.05))
            # Re-normalize
            total = sum(ind.values())
            for k in ind:
                ind[k] /= total
        return ind
