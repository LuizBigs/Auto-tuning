import numpy as np
from typing import Callable, List, Dict, Any

# A classe PatternSearch deve ser importada de patternsearch.py (que já é feita no main.py)

class BaseSimulator:
    """ Classe base para garantir que os simuladores tenham a mesma interface de __init__."""
    def __init__(self, objective_function: Callable[[List[float]], float], initial_point: List[float], **kwargs):
        self.objective_function = objective_function
        self.best_point = np.array(initial_point)
        self.best_value = objective_function(initial_point)
        self.max_iterations = kwargs.get('max_iterations', 1000)
        self.dimension = len(initial_point)

# ----------------------------------------------------
# Algoritmo 2: Nelder-Mead (Resultado bom, poucas iterações)
# ----------------------------------------------------
class NelderMeadDummy(BaseSimulator):
    """ Simulação de Algoritmo eficiente (e.g., Nelder-Mead) """

    def search(self) -> Dict[str, Any]:
        # Simula convergência rápida e precisa
        if self.dimension >= 2:
            point = np.ones(self.dimension) * 0.99 
        else:
            point = self.best_point * 0.9
            
        value = self.objective_function(point.tolist())
        
        return {
            'optimal_point': point.tolist(),
            'optimal_value': value,
            'iterations': 50, 
            'converged': True,
            'history': []
        }

# ----------------------------------------------------
# Algoritmo 3: Algoritmo B (Resultado ruim, falha)
# ----------------------------------------------------
class AlgorithmB_Simulated(BaseSimulator):
    """ Simulação de um Algoritmo com performance ruim """
        
    def search(self) -> Dict[str, Any]:
        return {
            'optimal_point': self.best_point.tolist(),
            'optimal_value': self.best_value, 
            'iterations': 1,
            'converged': False,
            'history': []
        }

# ----------------------------------------------------
# Algoritmo 4 e 5: Reutilizando as Simulações
# ----------------------------------------------------
class AlgorithmC_Simulated(AlgorithmB_Simulated):
    """ Simulação 4: Algoritmo C """
    pass

class AlgorithmD_Simulated(NelderMeadDummy):
    """ Simulação 5: Algoritmo D (Simula performance do Nelder-Mead) """
    pass