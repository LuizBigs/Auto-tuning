# main.py

from parameter_tunneling import ParameterTunneling, parse_initial_point
from patternsearch import PatternSearch
from typing import List

# ----------------------------------------------------
# 1. Definição das Funções Objetivo
# ----------------------------------------------------

def sphere_function(x: List[float]) -> float:
    """Função Sphere: Mínimo global em (0, 0, ...) com valor 0."""
    return sum(xi**2 for xi in x)

def rosenbrock_function(x: List[float]) -> float:
    """Função Rosenbrock: Mínimo global em (1, 1, ...) com valor 0."""
    n = len(x)
    value = 0
    for i in range(n - 1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value

OBJECTIVE_FUNCTIONS = {
    'sphere': sphere_function,
    'rosenbrock': rosenbrock_function,
    # Adicione outras funções aqui conforme necessário
}

# ----------------------------------------------------
# 2. Execução Principal
# ----------------------------------------------------

def main():
    # Carregar e Priorizar Parâmetros (Atividade 1)
    pt = ParameterTunneling()
    pt.add_command_line() \
      .add_config_file() \
      .add_environment_variables()
    
    # Parâmetros padrão finais (para garantir que tudo tem um valor)
    default_params = {
        'initial_step': 1.0,
        'step_reduction': 0.5,
        'tolerance': 1e-6,
        'max_iterations': 1000,
        'expansion_factor': 2.0 # Usado no Pattern Search mas não no CLI
    }
    pt.add_defaults(default_params)
    
    params = pt.get_parameters()
    
    print("--- Configuração da Otimização ---")
    pt.print_sources()
    print("-" * 34)
    
    # 3. Preparação para o Pattern Search
    
    # 3.1. Validar função objetivo
    function_name = params.get('function').lower()
    if function_name not in OBJECTIVE_FUNCTIONS:
        print(f"Erro: Função objetivo '{function_name}' não encontrada.")
        print(f"Funções disponíveis: {list(OBJECTIVE_FUNCTIONS.keys())}")
        return

    objective_func = OBJECTIVE_FUNCTIONS[function_name]
    
    # 3.2. Parse do ponto inicial
    dimension = params['dimension']
    initial_point_str = params.get('initial_point')
    
    # O parse_initial_point já trata o caso de None ou string inválida
    initial_point = parse_initial_point(initial_point_str, dimension)
    
    # 3.3. Configurar e Executar Pattern Search (Atividade 2)
    print(f"Função: {function_name}, Dimensão: {dimension}")
    print(f"Ponto Inicial: {initial_point}")
    print(f"Passo Inicial: {params['initial_step']}")
    print("-" * 34)
    
    optimizer = PatternSearch(
        objective_function=objective_func,
        initial_point=initial_point,
        initial_step=params['initial_step'],
        step_reduction=params['step_reduction'],
        tolerance=params['tolerance'],
        max_iterations=params['max_iterations'],
        expansion_factor=params.get('expansion_factor', 2.0)
    )
    
    results = optimizer.search()
    
    # 4. Exibir Resultados
    print("\n--- Resultados da Otimização ---")
    print(f"Ponto Ótimo Encontrado: {results['optimal_point']}")
    print(f"Valor Ótimo: {results['optimal_value']:.6f}")
    print(f"Iterações: {results['iterations']}")
    print(f"Status: {'Convergiu' if results['converged'] else 'Máximo de Iterações/Erro'}")
    print("-" * 34)

if __name__ == "__main__":
    main()