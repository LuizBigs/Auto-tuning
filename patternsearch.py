import numpy as np
from typing import Callable, List, Tuple, Dict, Any
import math

class PatternSearch:
    """
    Implementação do algoritmo Pattern Search (Busca Padrão)
    para otimização sem derivadas
    """
    
    def __init__(self, 
                 objective_function: Callable[[List[float]], float],
                 initial_point: List[float],
                 initial_step: float = 1.0,
                 step_reduction: float = 0.5,
                 expansion_factor: float = 2.0,
                 tolerance: float = 1e-6,
                 max_iterations: int = 1000):
        """
        Inicializa o algoritmo Pattern Search
        
        Args:
            objective_function: Função objetivo a ser minimizada
            initial_point: Ponto inicial da busca
            initial_step: Tamanho inicial do passo
            step_reduction: Fator de redução do passo
            expansion_factor: Fator de expansão do passo
            tolerance: Tolerância para critério de parada
            max_iterations: Número máximo de iterações
        """
        self.objective_function = objective_function
        self.current_point = np.array(initial_point, dtype=float)
        self.best_point = self.current_point.copy()
        self.step_size = initial_step
        self.step_reduction = step_reduction
        self.expansion_factor = expansion_factor
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        self.dimension = len(initial_point)
        self.history = []
        self.best_value = objective_function(initial_point)
        
    def generate_pattern(self, center: np.ndarray) -> List[np.ndarray]:
        """
        Gera pontos do padrão ao redor do ponto central
        """
        pattern_points = []
        
        # Gera pontos nas direções positivas e negativas de cada dimensão
        for i in range(self.dimension):
            # Direção positiva
            point_plus = center.copy()
            point_plus[i] += self.step_size
            pattern_points.append(point_plus)
            
            # Direção negativa
            point_minus = center.copy()
            point_minus[i] -= self.step_size
            pattern_points.append(point_minus)
        
        return pattern_points
    
    def search(self) -> Dict[str, Any]:
        """
        Executa o algoritmo Pattern Search
        """
        iteration = 0
        improvement = True
        
        while (iteration < self.max_iterations and 
               self.step_size > self.tolerance and 
               improvement):
            
            improvement = False
            
            # Gera pontos do padrão ao redor do ponto atual
            pattern_points = self.generate_pattern(self.current_point)
            
            # Avalia todos os pontos do padrão
            best_pattern_point = self.current_point
            best_pattern_value = self.best_value
            
            for point in pattern_points:
                try:
                    value = self.objective_function(point.tolist())
                    
                    if value < best_pattern_value:
                        best_pattern_value = value
                        best_pattern_point = point
                        improvement = True
                except Exception as e:
                    continue
            
            if improvement:
                # Move para o melhor ponto e expande o passo
                self.current_point = best_pattern_point
                self.best_point = self.current_point.copy()
                self.best_value = best_pattern_value
                self.step_size *= self.expansion_factor
            else:
                # Reduz o passo se não houve melhoria
                self.step_size *= self.step_reduction
            
            # Registra o histórico
            self.history.append({
                'iteration': iteration,
                'point': self.current_point.copy(),
                'value': self.best_value,
                'step_size': self.step_size
            })
            
            iteration += 1
        
        return {
            'optimal_point': self.best_point.tolist(),
            'optimal_value': self.best_value,
            'iterations': iteration,
            'converged': self.step_size <= self.tolerance,
            'history': self.history
        }

    def print_progress(self, iteration: int, point: List[float], value: float):
        """
        Imprime o progresso da otimização
        """
        print(f"Iteração {iteration}:")
        print(f"  Ponto: {point}")
        print(f"  Valor: {value:.6f}")
        print(f"  Step: {self.step_size:.6f}")
        print("-" * 40)