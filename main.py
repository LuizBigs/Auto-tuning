# main.py (Versão Final Multi-Algoritmo e Multi-Modelo)

import sys
import numpy as np
from typing import List, Dict, Any, Callable

# Importe suas classes
from parameter_tunneling import ParameterTunneling, parse_initial_point
from patternsearch import PatternSearch 
from algorithms import NelderMeadDummy, AlgorithmB_Simulated, AlgorithmC_Simulated, AlgorithmD_Simulated

# ----------------------------------------------------
## 1. Mapeamento de Algoritmos
# ----------------------------------------------------
ALGORITHM_MAP = {
    'patternsearch': PatternSearch,
    'nelder-mead': NelderMeadDummy,
    'algoritmo-b': AlgorithmB_Simulated,
    'algoritmo-c': AlgorithmC_Simulated, 
    'algoritmo-d': AlgorithmD_Simulated,       
}

# ----------------------------------------------------
## 2. Definição das Funções Objetivo (Modelos)
# ----------------------------------------------------

def sphere_function(x: List[float]) -> float:
    return sum(xi**2 for xi in x)

def rosenbrock_function(x: List[float]) -> float:
    n = len(x)
    value = 0
    for i in range(n - 1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value

OBJECTIVE_FUNCTIONS = {
    'sphere': sphere_function,
    'rosenbrock': rosenbrock_function,
}

# ----------------------------------------------------
## 3. Coleta Interativa de Parâmetros
# ----------------------------------------------------

def get_user_input_params(default_params: Dict) -> Dict[str, Any]:
    """Pede todos os parâmetros ao usuário via terminal, incluindo algoritmos e modelos."""
    
    print("\n--- Configuração Interativa da Otimização ---")
    user_params = {}
    
    # 1. Algoritmos
    available_algs = ", ".join(ALGORITHM_MAP.keys())
    while 'algorithms_list' not in user_params:
        alg_str = input(f"Algoritmos (Separe por vírgula. Ex: patternsearch,nelder-mead): ").lower()
        algs = [a.strip() for a in alg_str.split(',') if a.strip()]
        
        valid_algs = [a for a in algs if a in ALGORITHM_MAP]
        
        if valid_algs:
            user_params['algorithms_list'] = valid_algs
        else:
            print(f"Erro: Nenhum algoritmo válido foi inserido. Algoritmos disponíveis: {available_algs}")
    
    # 2. Funções Objetivo (Modelos)
    available_funcs = ", ".join(OBJECTIVE_FUNCTIONS.keys())
    while 'functions_list' not in user_params:
        func_str = input(f"Funções Objetivo (Separe por vírgula. Ex: sphere,rosenbrock): ").lower()
        funcs = [f.strip() for f in func_str.split(',') if f.strip()]
        valid_funcs = [f for f in funcs if f in OBJECTIVE_FUNCTIONS]
        
        if valid_funcs:
            user_params['functions_list'] = valid_funcs
        else:
            print(f"Erro: Nenhuma função válida foi inserida. Funções disponíveis: {available_funcs}")
            
    # 3. Dimensão
    while 'dimension' not in user_params:
        try:
            dim = int(input(f"Dimensão (Padrão: {default_params['dimension']}): ") or default_params['dimension'])
            if dim > 0:
                user_params['dimension'] = dim
            else:
                print("A dimensão deve ser um número positivo.")
        except ValueError:
            print("Entrada inválida. Digite um número inteiro.")
            
    # 4. Ponto Inicial
    print("\n--- Parâmetros de Execução ---")
    point_str = input(f"Ponto Inicial (e.g., 0,0,0 - Padrão: {default_params['initial_point']}): ")
    if point_str:
        user_params['initial_point'] = point_str
        
    # 5. Parâmetros numéricos (Hyperparâmetros)
    param_keys = ['initial_step', 'step_reduction', 'tolerance', 'max_iterations']
    
    for key in param_keys:
        default_val = default_params[key]
        while True:
            try:
                value_str = input(f"{key.replace('_', ' ').title()} (Padrão: {default_val}): ")
                value = type(default_val)(value_str) if value_str else default_val
                
                user_params[key] = value
                break
            except ValueError:
                print("Entrada inválida. Digite um número válido.")
                
    user_params['sources'] = ['user_input']
    
    return user_params


# ----------------------------------------------------
## 4. Execução Principal (Com Loop Duplo)
# ----------------------------------------------------

def main():
    
    # Parâmetros padrão
    default_params = {
        'initial_step': 1.0,
        'step_reduction': 0.5,
        'tolerance': 1e-6,
        'max_iterations': 1000,
        'expansion_factor': 2.0,
        'dimension': 2,
        'initial_point': '1,1'
    }
    
    params = get_user_input_params(default_params)
    
    pt = ParameterTunneling()
    pt.parameters.update(params)
    pt.add_defaults(default_params)
    
    final_params = pt.get_parameters()
    
    print("\n--- Configuração Final ---")
    print("-" * 34)
    print(f"Algoritmos a testar: {', '.join(final_params['algorithms_list'])}")
    print(f"Modelos (Funções) a testar: {', '.join(final_params['functions_list'])}")
    print("-" * 34)
    
    # LOOP DUPLO: Algoritmos X Funções
    for alg_name in final_params['algorithms_list']:
        algorithm_class = ALGORITHM_MAP[alg_name]
        
        for function_name in final_params['functions_list']:
            
            print(f"\n=============================================")
            print(f"🚀 {alg_name.upper()} em {function_name.upper()} 🚀")
            print(f"=============================================")
            
            objective_func = OBJECTIVE_FUNCTIONS[function_name]
            
            dimension = final_params['dimension']
            initial_point_str = final_params.get('initial_point')
            initial_point = parse_initial_point(initial_point_str, dimension)
            
            # 5. Execução do Algoritmo
            optimizer = algorithm_class(
                objective_function=objective_func,
                initial_point=initial_point,
                initial_step=final_params['initial_step'],
                step_reduction=final_params['step_reduction'],
                tolerance=final_params['tolerance'],
                max_iterations=final_params['max_iterations'],
                expansion_factor=final_params.get('expansion_factor', 2.0)
            )
            
            results = optimizer.search()
            
            # 6. Exibir Resultados
            print("\n--- Resultados da Otimização ---")
            print(f"Algoritmo: {alg_name.title()}")
            print(f"Função: {function_name.title()}")
            print(f"Ponto Ótimo Encontrado: {results['optimal_point']}")
            print(f"Valor Ótimo: {results['optimal_value']:.6f}")
            print(f"Iterações: {results['iterations']}")
            print(f"Status: {'Convergiu' if results['converged'] else 'Máximo de Iterações/Erro'}")
            print("-" * 34)

if __name__ == "__main__":
    main()