#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ferramenta de Otimização Automática (Autotuner) para o executável modelo10.exe.
Este script integra diversas heurísticas de otimização e gerencia a avaliação
robusta do modelo externo com retries, timeouts e registro detalhado de dados.
"""

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import List, Tuple, Optional, Dict, Any, Callable

# -----------------------------------------------------------------------------------
# Dependências opcionais (Scipy e Optuna)
# Inicializamos as variáveis para que estejam sempre no escopo global.
optuna = None
minimize = None 

try:
    # Tenta importar Optuna
    import optuna
except ImportError:
    pass # Optuna não está instalado

try:
    # Tenta importar a função minimize do SciPy (Necessária para Simplex/Nelder-Mead)
    # Se falhar, 'minimize' permanece como None.
    from scipy.optimize import minimize
except ImportError:
    pass

# ---------------- Configurações Padrão (Sobrescritas via CLI) ----------------
DEFAULT_EXECUTABLE_PATH = "simulado.exe"
DEFAULT_REPLICATES = 1
DEFAULT_RETRIES = 2
DEFAULT_TIMEOUT_S = 12.0
OPTUNA_DEFAULT_TRIALS = 40
OPTUNA_GLOBAL_TIMEOUT = 60 * 10
GA_POPULATION_SIZE = 16
GA_GENERATIONS = 20
PS_MULTI_STARTS = 2
SIMPLEX_MAX_ITERATIONS = 120
INITIAL_RANDOM_SEED = 42
# -----------------------------------------------------------------------------------

random.seed(INITIAL_RANDOM_SEED)

# Objetivo da otimização: 'max' ou 'min'
GOAL_MAX = 'max'
GOAL_MIN = 'min'

def score_for_goal(value: float, goal: str) -> float:
    """Retorna o score que será usado para comparação interna.
    - Para maximizar, score = value
    - Para minimizar, score = -value (para que comparações 'maior é melhor' funcionem)
    """
    if value is None or value == -math.inf:
        return -math.inf
    if goal == GOAL_MIN:
        return -value
    return value

# Registro global de todas as avaliações do modelo externo
_global_evaluations_record: List[Dict[str, Any]] = []

# -----------------------------------------------------------
# Utilitários e Comunicação Externa
# -----------------------------------------------------------
def simple_logger(message: str):
    """ Imprime uma mensagem com timestamp para acompanhamento. """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def extract_float_from_output(text: str) -> float:
    """ Tenta extrair o primeiro valor numérico (float ou int) de uma string de texto. """
    if text is None:
        raise ValueError("A saída fornecida está vazia.")

    # 1. Normaliza o separador decimal (usa ponto para floats)
    processed_text = re.sub(r'(\d+),(\d+)', lambda m: f"{m.group(1)}.{m.group(2)}", text)

    # 2. Busca pelo primeiro número encontrado (positivo ou negativo)
    match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", processed_text)
    if not match:
        raise ValueError(f"Nenhum valor numérico válido foi encontrado na saída: {text!r}")
    return float(match.group(0))

def validate_executable(path: str):
    """ Verifica se o caminho do executável é válido. """
    from shutil import which
    if which(path) is None and not Path(path).exists():
        raise FileNotFoundError(f"O programa executável '{path}' não foi localizado no sistema (PATH ou diretório atual).")

def execute_external_process(exec_path: str, opt_type: str, params: List[int], timeout: float, retries: int) -> Tuple[float, str, float]:
    """
    Chama o executável (simulaod.exe) com múltiplas tentativas e timeout. 
    Retorna (valor_otimizado, saída_completa, tempo_gasto).
    """
    # Formato esperado: [exec_path, P1, P2, P3, P4, P5]
    # Nota: opt_type não é usado pelo executável simulado.exe
    arguments = [exec_path] + [str(int(x)) for x in params]
    last_exception = None
    backoff_delay = 0.1

    for attempt in range(1, retries + 1):
        time_start = time.time()
        try:
            process = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)
            stdout_output = (process.stdout or "").strip()
            stderr_output = (process.stderr or "").strip()
            elapsed_time = time.time() - time_start

            if process.returncode != 0:
                last_exception = RuntimeError(f"O processo terminou com código de erro {process.returncode}. Saída de erro (stderr): {stderr_output}")
                time.sleep(backoff_delay)
                backoff_delay = min(2.0, backoff_delay * 2)
                continue

            # Tenta extrair o valor da otimização
            complete_output = stdout_output if stdout_output else stderr_output
            try:
                if complete_output:
                    optimized_value = extract_float_from_output(complete_output)
                    return optimized_value, complete_output, elapsed_time
                else:
                    last_exception = RuntimeError("O processo não produziu nenhuma saída (stdout/stderr).")
                    time.sleep(backoff_delay)
                    backoff_delay = min(2.0, backoff_delay * 2)
                    continue
            except ValueError as ve:
                last_exception = ve
                time.sleep(backoff_delay)
                backoff_delay = min(2.0, backoff_delay * 2)
                continue

        except KeyboardInterrupt:
            simple_logger("❌ Execução interrompida pelo usuário (Ctrl+C)")
            raise KeyboardInterrupt("Execução interrompida pelo usuário")
        except subprocess.TimeoutExpired:
            last_exception = RuntimeError(f"O tempo limite ({timeout}s) foi excedido durante a execução de: {arguments}")
            time.sleep(backoff_delay)
            backoff_delay = min(2.0, backoff_delay * 2)
            continue
        except Exception as e:
            last_exception = e
            time.sleep(backoff_delay)
            backoff_delay = min(2.0, backoff_delay * 2)
            continue

    raise RuntimeError(f"A execução falhou após {retries} tentativas. Último erro: {last_exception}")

def evaluate_average_performance(exec_path: str, opt_type: str, params: List[int], replicates: int, timeout: float, retries: int, method_label: Optional[str]=None, executor: Optional[ThreadPoolExecutor]=None):
    """
    Executa o modelo 'replicates' vezes e calcula a média do desempenho (valor de otimização).
    """
    def eval_single(r):
        try:
            value, output, elapsed = execute_external_process(exec_path, opt_type, params, timeout=timeout, retries=retries)
            return {
                "valor": value, "tempo": elapsed, "rep": r,
                "stdout": (output[:1000] + "...") if output and len(output) > 1000 else output,
                "erro": None
            }
        except KeyboardInterrupt:
            raise KeyboardInterrupt("Execução interrompida pelo usuário")
        except Exception as e:
            return {
                "valor": None, "tempo": None, "rep": r,
                "stdout": None, "erro": str(e)
            }

    try:
        if executor and replicates > 1:
            futures = [executor.submit(eval_single, r) for r in range(replicates)]
            results = []
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except KeyboardInterrupt:
                    # Cancela futures pendentes
                    for future in futures:
                        future.cancel()
                    raise KeyboardInterrupt("Execução interrompida pelo usuário")
                except Exception as e:
                    # Trata exceções em threads, mas continua
                    results.append({"valor": None, "tempo": None, "rep": -1, "stdout": None, "erro": str(e)})

        else:
            results = [eval_single(r) for r in range(replicates)]

    except KeyboardInterrupt:
        simple_logger("❌ Avaliação interrompida pelo usuário")
        raise KeyboardInterrupt("Execução interrompida pelo usuário")

    optimized_values = [res["valor"] for res in results if res["valor"] is not None]
    execution_times = [res["tempo"] for res in results if res["tempo"] is not None]

    for res in results:
        _global_evaluations_record.append({
            "metodo": method_label, "tipo": opt_type, "params": params.copy(),
            "valor": res["valor"], "tempo": res["tempo"], "rep": res["rep"],
            "stdout": res["stdout"], "erro": res["erro"], "timestamp": time.time()
        })

    if not optimized_values:
        raise RuntimeError(f"Todas as réplicas falharam para o tipo={opt_type}, parâmetros={params}.")
    return mean(optimized_values), optimized_values, execution_times

# -----------------------------------------------------------
# Métodos de Otimização (Heurísticas)
# -----------------------------------------------------------
def optimization_method_ps(exec_path, replicates, retries, timeout, max_iter=100, multistarts=PS_MULTI_STARTS, parallel=False, goal=GOAL_MAX):
    """ Busca Padrão (Pattern Search) com múltiplas inicializações. """
    simple_logger("➡ Busca Padrão (Pattern Search, multi-start)")
    time_start = time.time()
    best_result_global: Dict[str, Any] = {"valor": None, "tipo": None, "params": None}

    executor = ThreadPoolExecutor(max_workers=8) if parallel else None

    try:
        for start in range(multistarts):
            # 5 parâmetros numéricos (valores de 1 a 100)
            opt_type = "default"  # Mantém para compatibilidade, mas não é usado
            
            # Estratégia de inicialização diversificada
            if start == 0:
                # Primeira tentativa: todos no meio
                current_params = [50, 50, 50, 50, 50]
            elif start == 1:
                # Segunda tentativa: todos altos
                current_params = [100, 100, 100, 100, 100]
            elif start == 2:
                # Terceira tentativa: valores altos variados
                current_params = [random.randint(70, 100) for _ in range(5)]
            else:
                # Outras tentativas: totalmente aleatório
                current_params = [random.randint(1, 100) for _ in range(5)]

            try:
                current_value, _, _ = evaluate_average_performance(exec_path, opt_type, current_params, replicates, timeout, retries, method_label="Pattern Search", executor=executor)
                simple_logger(f"  🔍 PS Start {start+1}/{multistarts} - Inicial: {current_params} = {current_value:.2f}")
            except KeyboardInterrupt:
                raise KeyboardInterrupt("Pattern Search interrompido pelo usuário")
            except Exception as e:
                simple_logger(f"⚠ PS inicialização {start} falhou: {e}")
                continue

            # Scores para comparação (sempre: maior é melhor)
            current_score = score_for_goal(current_value, goal)

            step_size = 25  # Começar com passos maiores para explorar mais rápido
            iteration_count = 0
            last_progress_report = 0
            while step_size >= 1 and iteration_count < max_iter:
                candidates = []
                for i in range(5):
                    for delta in (-step_size, step_size):
                        candidate_params = current_params.copy()
                        candidate_params[i] = int(min(100, max(1, candidate_params[i] + delta)))
                        candidates.append((candidate_params, i, delta))

                improved = False
                candidate_values = {}

                # Avaliação dos vizinhos (paralela ou sequencial)
                try:
                    if executor:
                        futures = {executor.submit(evaluate_average_performance, exec_path, opt_type, cand[0], replicates, timeout, retries, "Pattern Search"): cand for cand in candidates}
                        for future in as_completed(futures):
                            cand = futures[future]
                            try:
                                value_cand, _, _ = future.result()
                                candidate_values[tuple(cand[0])] = value_cand
                                iteration_count += 1
                            except KeyboardInterrupt:
                                # Cancela todos os futures pendentes
                                for f in futures:
                                    f.cancel()
                                raise KeyboardInterrupt("Pattern Search interrompido pelo usuário")
                            except Exception:
                                continue
                    else:
                        for cand in candidates:
                            try:
                                value_cand, _, _ = evaluate_average_performance(exec_path, opt_type, cand[0], replicates, timeout, retries, "Pattern Search")
                                candidate_values[tuple(cand[0])] = value_cand
                                iteration_count += 1
                            except KeyboardInterrupt:
                                raise KeyboardInterrupt("Pattern Search interrompido pelo usuário")
                            except Exception:
                                continue
                                
                except KeyboardInterrupt:
                    simple_logger(f"⚠ Pattern Search interrompido na iteração {iteration_count}")
                    break

                # Escolhe o melhor vizinho (usando score)
                for cand_params in candidate_values:
                    value_cand = candidate_values[cand_params]
                    cand_score = score_for_goal(value_cand, goal)
                    if cand_score > current_score:
                        current_value = value_cand
                        current_score = cand_score
                        current_params = list(cand_params)
                        improved = True

                if not improved:
                    step_size = step_size // 2 # Reduz o passo
                
                # Relatório de progresso a cada 50 iterações
                if iteration_count - last_progress_report >= 50:
                    elapsed = time.time() - time_start
                    simple_logger(f"  📈 Progresso: {iteration_count} avaliações, {elapsed:.1f}s decorridos, step={step_size}, atual={current_value:.2f}")
                    last_progress_report = iteration_count

                # Atualiza melhor global
                if best_result_global["valor"] is None:
                    best_result_global.update({"valor": current_value, "tipo": opt_type, "params": current_params.copy()})
                    simple_logger(f"  ✨ NOVO MELHOR: {current_params} = {current_value:.2f}")
                else:
                    best_score = score_for_goal(best_result_global["valor"], goal)
                    if current_score > best_score:
                        best_result_global.update({"valor": current_value, "tipo": opt_type, "params": current_params.copy()})
                        simple_logger(f"  ✨ NOVO MELHOR: {current_params} = {current_value:.2f}")

    except KeyboardInterrupt:
        simple_logger("⚠ Pattern Search interrompido pelo usuário")
        # Retorna o melhor resultado encontrado até agora
    finally:
        if executor:
            executor.shutdown()

    elapsed_time = time.time() - time_start
    
    # Se não encontrou nenhum resultado válido, retorna -inf
    if best_result_global["valor"] is None:
        simple_logger("⚠ Pattern Search não conseguiu obter nenhum resultado válido")
        return {"metodo": "Pattern Search", "melhor_valor": -math.inf, "parametros": (None, None), "tempo": elapsed_time}
    
    simple_logger(f"🏁 Pattern Search finalizado: Melhor={best_result_global['valor']:.2f} em {best_result_global['params']}")
    return {"metodo": "Pattern Search", "melhor_valor": best_result_global["valor"], "parametros": (best_result_global["tipo"], best_result_global["params"]), "tempo": elapsed_time}

def optimization_method_ga(exec_path, replicates, retries, timeout, pop_size=GA_POPULATION_SIZE, generations=GA_GENERATIONS, parallel=False, goal=GOAL_MAX, seed_individual: Optional[Tuple[str, List[int]]] = None):
    """ Algoritmo Genético (Genetic Algorithm).
    Agora aceita falhas nas avaliações e usa `score_for_goal` para selecionar/exibir o melhor
    indivíduo independentemente de max/min.
    Opcionalmente, `seed_individual` pode ser fornecido como (tipo, params) para inicializar
    a população com um indivíduo promissor (usado pelo método combinado).
    """
    simple_logger("➡ Algoritmo Genético (GA)")
    time_start = time.time()
    opt_type_default = "default"  # Tipo fixo, pois o executável não usa

    executor = ThreadPoolExecutor(max_workers=8) if parallel else None

    def evaluate_individual(individual):
        try:
            v, _, _ = evaluate_average_performance(exec_path, individual["tipo"], individual["params"], replicates, timeout, retries, method_label="Algoritmo Genético", executor=executor if replicates > 1 else None)
            individual["valor"] = v
        except Exception:
            individual["valor"] = None # Falha

    def tournament_selection(pop, k=3, goal=GOAL_MAX):
        candidates = random.sample(pop, min(k, len(pop)))
        # Atenção: usa -math.inf como valor de falha, que tem o pior score.
        return max(candidates, key=lambda x: score_for_goal(x.get("valor", -math.inf if x.get("valor") is not None else -math.inf), goal))

    # Inicialização da População (pode ser semear com seed_individual)
    population = []
    for i in range(pop_size):
        if i == 0 and seed_individual is not None:
            population.append({"tipo": seed_individual[0], "params": seed_individual[1].copy(), "valor": None})
        else:
            population.append({"tipo": opt_type_default, "params": [random.randint(1, 100) for _ in range(5)], "valor": None})

    # Avaliação da População Inicial
    try:
        if executor and parallel:
            futures = [executor.submit(evaluate_individual, ind) for ind in population]
            for f in as_completed(futures): f.result()
        else:
            for ind in population: evaluate_individual(ind)
    except KeyboardInterrupt:
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        raise

    # Loop de Gerações
    for gen in range(generations):
        try:
            # Filtra indivíduos com valor para evitar erros de comparação
            valid_population = [ind for ind in population if ind.get("valor") is not None]
            
            if not valid_population:
                simple_logger(f"⚠ Geração {gen+1}: População totalmente falha. Parando GA.")
                break

            # Ordena pelo score (maior é melhor)
            population.sort(key=lambda x: score_for_goal(x.get("valor", -math.inf), goal), reverse=True)
            n_elite = max(1, pop_size // 5)
            new_population = population[:n_elite]
            offspring = []
            
            # Garante que a seleção do torneio use apenas a população válida para evitar erros
            pop_for_selection = [ind for ind in population if ind.get("valor") is not None]
            
            while len(new_population) + len(offspring) < pop_size:
                # Seleção e Cruzamento (Cross-over)
                if len(pop_for_selection) < 2:
                    # Se não houver indivíduos válidos suficientes, usa aleatórios
                    parent1 = random.choice(population)
                    parent2 = random.choice(population)
                else:
                    parent1 = tournament_selection(pop_for_selection, k=3, goal=goal)
                    parent2 = tournament_selection(pop_for_selection, k=3, goal=goal)

                child = {"tipo": opt_type_default, "params": [], "valor": None}
                
                # O cruzamento mistura os parâmetros de P1 e P2
                for a, b in zip(parent1["params"], parent2["params"]):
                    child["params"].append(random.choice([a, b]))

                # Mutação
                if random.random() < 0.12:
                    idx = random.randrange(5)
                    child["params"][idx] = random.randint(1, 100)

                offspring.append(child)

            # Avaliação dos Filhos
            if executor and parallel:
                futures = [executor.submit(evaluate_individual, child) for child in offspring]
                for f in as_completed(futures): f.result()
            else:
                for child in offspring: evaluate_individual(child)

            new_population.extend(offspring)
            population = new_population
            
            # Melhor da geração (pode ser None se todas falharem)
            best_of_gen = max(population, key=lambda x: score_for_goal(x.get("valor", -math.inf), goal))
            if best_of_gen.get('valor') is not None:
                simple_logger(f"  🧬 Geração {gen+1}/{generations} - Melhor: {best_of_gen['params']} = {best_of_gen.get('valor'):.2f}")

        except KeyboardInterrupt:
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)
            raise

    if executor and parallel:
        executor.shutdown()

    best_individual = max(population, key=lambda x: score_for_goal(x.get("valor", -math.inf), goal))
    elapsed_time = time.time() - time_start
    
    # Verifica se o melhor indivíduo tem valor válido
    best_value = best_individual.get("valor")
    if best_value is None or best_value == -math.inf:
        simple_logger("⚠ Algoritmo Genético não conseguiu obter nenhum resultado válido")
        return {"metodo": "Algoritmo Genético", "melhor_valor": -math.inf, "parametros": (None, None), "tempo": elapsed_time}
    
    simple_logger(f"🏁 GA finalizado: Melhor={best_value:.2f} em {best_individual['params']}")
    return {"metodo": "Algoritmo Genético", "melhor_valor": best_value, "parametros": (best_individual["tipo"], best_individual["params"]), "tempo": elapsed_time}


def optimization_method_combined(exec_path, replicates, retries, timeout, goal=GOAL_MAX, ps_max_iter=50, ga_pop_size=GA_POPULATION_SIZE, ga_generations=GA_GENERATIONS, parallel=False):
    """Método combinado: executa Pattern Search rápido para obter um seed, então
    roda GA usando o indivíduo resultante como semente. Retorna o melhor entre os dois.
    """
    simple_logger("➡ Método combinado: PatternSearch -> GA")
    time_start = time.time()

    # 1) Executa um Pattern Search rápido (multistarts=1)
    ps_res = optimization_method_ps(exec_path, replicates, retries, timeout, max_iter=ps_max_iter, multistarts=1, parallel=parallel, goal=goal)
    ps_val = ps_res.get("melhor_valor")
    ps_params = ps_res.get("parametros")    # (tipo, params)

    seed = None
    if ps_params and ps_params[0] is not None and ps_params[1] is not None:
        seed = (ps_params[0], ps_params[1])

    # 2) Roda GA usando o seed do PS
    ga_res = optimization_method_ga(exec_path, replicates, retries, timeout, pop_size=ga_pop_size, generations=ga_generations, parallel=parallel, goal=goal, seed_individual=seed)
    ga_val = ga_res.get("melhor_valor")
    ga_params = ga_res.get("parametros")

    # Escolhe o melhor considerando goal
    ps_score = score_for_goal(ps_val, goal) if ps_val is not None else -math.inf
    ga_score = score_for_goal(ga_val, goal) if ga_val is not None else -math.inf

    if ga_score >= ps_score:
        best = {"metodo": "Combined (GA after PS)", "melhor_valor": ga_val, "parametros": ga_params, "tempo": (time.time() - time_start)}
    else:
        best = {"metodo": "Combined (PS)", "melhor_valor": ps_val, "parametros": ps_params, "tempo": (time.time() - time_start)}

    return best

def optimization_method_simplex(exec_path, replicates, retries, timeout, maxiter=SIMPLEX_MAX_ITERATIONS, goal=GOAL_MAX):
    """ Simplex / Nelder-Mead usando SciPy para otimização contínua.
    Respeita o `goal`: se goal==GOAL_MAX, transforma a função em -value para minimizar.
    """
    simple_logger("➡ Simplex / Nelder-Mead (opcional)")
    
    # Verifica se a importação do SciPy foi bem-sucedida (corrigido para usar a variável global 'minimize')
    if minimize is None:
        simple_logger("⚠ SciPy não está instalado. Pulando Simplex.")
        return {"metodo": "Simplex", "melhor_valor": -math.inf, "parametros": None, "tempo": 0}

    time_start = time.time()
    opt_type = "default"  # Tipo fixo, não usado pelo executável
    
    # Novo: Contador para rastrear as iterações (avaliações do modelo)
    iteration_counter = 0

    def objective_function_scipy(x_float_array):
        # Novo: Acessa e incrementa o contador da função externa
        nonlocal iteration_counter
        iteration_counter += 1
        
        # Converte o array de floats do SciPy para inteiros [1-100] (entrada do .exe)
        params = [int(min(100, max(1, round(xx)))) for xx in x_float_array]
        try:
            value, _, _ = evaluate_average_performance(exec_path, opt_type, params, replicates, timeout, retries, method_label="Simplex")
            
            # Novo: Log detalhado da avaliação
            simple_logger(f"  Simplex {iteration_counter}/{maxiter} - Parâmetros: {params} | Valor: {value:.6g}")
            
            # Se queremos maximizar o valor original, retornamos -value para que o minimize
            # do SciPy encontre o máximo. Se queremos minimizar, retornamos value.
            return -value if goal == GOAL_MAX else value
        except Exception:
            simple_logger(f"  Simplex {iteration_counter}/{maxiter} - Falha na avaliação para parâmetros: {params}")
            return 1e9 # Penalidade alta

    # Ponto inicial x0 para os 5 parâmetros (float/int)
    initial_point_x0 = [random.randint(1, 100) for _ in range(5)]

    # Executa a otimização
    result = minimize(objective_function_scipy, initial_point_x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    
    # Processa o resultado final
    best_params = [int(min(100, max(1, round(x)))) for x in result.x]
    
    # SciPy minimize retorna o valor da função objetivo (que pode ser -valor_real se goal=max)
    # Por isso, invertemos o sinal se o objetivo era maximizar.
    best_value = -result.fun if goal == GOAL_MAX else result.fun
    if result.fun in (None, float("inf"), float("-inf")) or best_value in (None, float("inf"), float("-inf")):
        best_value = -math.inf
        simple_logger("⚠ Simplex não conseguiu obter resultado válido")

    elapsed_time = time.time() - time_start
    return {"metodo": "Simplex", "melhor_valor": best_value, "parametros": (opt_type, best_params), "tempo": elapsed_time}

def optimization_method_optuna(exec_path, replicates, retries, timeout, n_trials=OPTUNA_DEFAULT_TRIALS, overall_timeout=OPTUNA_GLOBAL_TIMEOUT, storage_path="sqlite:///optuna_study.db", goal=GOAL_MAX):
    """ Otimização Bayesiana (Optuna). """
    if optuna is None:
        raise RuntimeError("Optuna não está instalado. Por favor, instale: pip install optuna")

    simple_logger("➡ Otimização Bayesiana (Optuna)")
    time_start = time.time()
    sampler = optuna.samplers.TPESampler(seed=INITIAL_RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    try:
        # Cria ou retoma o estudo em SQLite
        direction = "maximize" if goal == GOAL_MAX else "minimize"
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner,
                                     storage=storage_path, study_name="optimization_study_model", load_if_exists=True)
    except Exception as e:
        simple_logger(f"⚠ Erro ao iniciar Optuna com SQLite: {e}. Optuna desabilitado.")
        return {"metodo": "Optuna (bayesiana)", "melhor_valor": -math.inf, "parametros": None, "tempo": 0}


    def objective_optuna(trial):
        # Define o espaço de busca (hiperparâmetros)
        opt_type = "default"  # Tipo fixo, não usado pelo executável
        # 5 parâmetros (inteiros 1-100)
        params = [trial.suggest_int(f"p{i+1}", 1, 100) for i in range(5)]

        try:
            avg_value, _, _ = evaluate_average_performance(exec_path, opt_type, params, replicates, timeout, retries, method_label="Optuna (bayesiana)")
            
            # Log detalhado da avaliação
            simple_logger(f"  Optuna Trial {trial.number}/{n_trials} - Parâmetros: {params} | Valor: {avg_value:.6g}")
            
            return avg_value # Retorna o valor para maximização (Optuna gerencia o sinal)
        except Exception as e:
            simple_logger(f"⚠ Optuna trial {trial.number} falhou: {e}")
            return -1e12 # Penalidade

    study.optimize(objective_optuna, n_trials=n_trials, timeout=overall_timeout, show_progress_bar=True)
    best_trial = study.best_trial
    
    # Extrai o melhor conjunto de parâmetros
    best_type = best_trial.params.get("tipo")
    best_params_list = [best_trial.params.get(f"p{i+1}") for i in range(5)]

    elapsed_time = time.time() - time_start
    return {"metodo": "Optuna (bayesiana)", "melhor_valor": best_trial.value, "parametros": (best_type, best_params_list), "tempo": elapsed_time, "study": study}

# -----------------------------------------------------------
# I/O: salvar avaliações / resumo
# -----------------------------------------------------------
def save_evaluations_to_csv(filepath: str, method_filter: Optional[str] = None):
    """ Salva o registro completo de todas as avaliações no formato CSV. 
    Se method_filter for especificado, salva apenas as avaliações daquele método.
    """
    fields = ["metodo", "tipo", "params", "valor", "tempo", "rep", "stdout", "erro", "timestamp"]
    
    # Filtra as avaliações se necessário
    if method_filter:
        records = [r for r in _global_evaluations_record if r.get("metodo") == method_filter]
    else:
        records = _global_evaluations_record
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "metodo": row.get("metodo"),
                "tipo": row.get("tipo"),
                "params": ",".join(map(str, row.get("params", []))) if row.get("params") else None,
                "valor": row.get("valor"),
                "tempo": row.get("tempo"),
                "rep": row.get("rep"),
                "stdout": row.get("stdout"),
                "erro": row.get("erro"),
                "timestamp": row.get("timestamp")
            })

def save_summary_to_json(filepath: str, summary: dict):
    """ Salva o resumo dos melhores resultados em formato JSON. """
    with open(filepath, "w", encoding="utf-8") as f:
        # Remove a referência ao Optuna Study antes de salvar (não é serializável)
        for result in summary.get("resultados", []):
            if "study" in result:
                del result["study"]
        json.dump(summary, f, ensure_ascii=False, indent=2)

def save_individual_method_files(result, method_name, global_time_start, args):
    """Salva arquivos individuais (CSV, JSON, TXT) para um método específico."""
    # Normaliza o nome do método para usar como sufixo
    method_suffix = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    
    # 1. Salvar CSV de avaliações do método
    csv_filename = f"avaliacoes_{method_suffix}.csv"
    save_evaluations_to_csv(csv_filename, method_filter=method_name)
    
    # 2. Salvar JSON de resumo do método
    json_filename = f"resumo_{method_suffix}.json"
    summary = {
        "timestamp": time.time(),
        "metodo": method_name,
        "resultado": result,
        "total_avaliacoes": len([r for r in _global_evaluations_record if r.get("metodo") == method_name]),
        "tempo_total_s": result.get("tempo", 0),
        "objetivo": args.goal
    }
    save_summary_to_json(json_filename, summary)
    
    # 3. Salvar relatório TXT do método
    txt_filename = f"relatorio_{method_suffix}.txt"
    report_text = generate_detailed_report([result], global_time_start, args, method_filter=method_name)
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    return csv_filename, json_filename, txt_filename

# -----------------------------------------------------------
# Função de Interação
# -----------------------------------------------------------
def ask_yes_no(question: str) -> bool:
    """ Faz uma pergunta ao usuário no console (sim/nao). """
    while True:
        try:
            response = input(f"{question} (sim/nao): ").strip().lower()
            if response in ['sim', 's', 'yes', 'y']:
                return True
            elif response in ['nao', 'n', 'no']:
                return False
            else:
                print("Resposta inválida. Por favor, responda com 'sim' ou 'nao'.")
        except EOFError:
            return False
        except KeyboardInterrupt:
            print("\nExecução interrompida pelo usuário.")
            return False

# -----------------------------------------------------------
# Função de Geração de Relatório Detalhado
# -----------------------------------------------------------
def generate_detailed_report(all_results, global_time_start, args, method_filter: Optional[str] = None):
    """Gera um relatório detalhado em texto com os resultados da otimização.
    Se method_filter for especificado, gera relatório apenas para aquele método.
    """
    # Filtra resultados se necessário
    if method_filter:
        results_to_report = [r for r in all_results if method_filter.lower() in r.get("metodo", "").lower()]
        title = f"RELATÓRIO DETALHADO - {method_filter.upper()}"
    else:
        results_to_report = all_results
        title = "RELATÓRIO DETALHADO DE OTIMIZAÇÃO"
    
    # Conta avaliações do método específico
    if method_filter:
        eval_count = len([r for r in _global_evaluations_record if r.get("metodo") == method_filter])
    else:
        eval_count = len(_global_evaluations_record)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(title)
    report_lines.append("=" * 80)
    report_lines.append(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Tempo Total de Execução: {time.time() - global_time_start:.2f} segundos ({(time.time() - global_time_start)/60:.2f} minutos)")
    report_lines.append(f"Objetivo: {'MAXIMIZAR' if args.goal == GOAL_MAX else 'MINIMIZAR'}")
    report_lines.append(f"Total de Avaliações do Modelo: {eval_count}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Resultados por método
    report_lines.append("RESULTADOS POR MÉTODO:")
    report_lines.append("-" * 80)
    
    for i, result in enumerate(results_to_report, 1):
        method_name = result.get("metodo", "Desconhecido")
        best_value = result.get("melhor_valor")
        exec_time = result.get("tempo", 0)
        params = result.get("parametros")
        
        report_lines.append(f"\n{i}. {method_name}")
        report_lines.append(f"   Melhor Valor: {best_value:.6g}" if best_value not in (None, -math.inf) else "   Melhor Valor: N/A")
        
        if exec_time is not None:
            report_lines.append(f"   Tempo de Execução: {exec_time:.2f} segundos ({exec_time/60:.2f} minutos)")
        else:
            report_lines.append(f"   Tempo de Execução: N/A")
        
        if params:
            tipo, param_list = params
            report_lines.append(f"   Tipo: {tipo}")
            report_lines.append(f"   Parâmetros: {param_list}")
        else:
            report_lines.append(f"   Parâmetros: N/A")
        report_lines.append("")
    
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Vencedor
    valid_results = [r for r in results_to_report if r.get("melhor_valor") not in (None, -math.inf)]
    if valid_results:
        winner = max(valid_results, key=lambda r: score_for_goal(r["melhor_valor"], args.goal))
        report_lines.append("🏆 MELHOR RESULTADO GERAL:")
        report_lines.append("-" * 80)
        report_lines.append(f"Método Vencedor: {winner.get('metodo', 'Desconhecido')}")
        
        winner_value = winner.get('melhor_valor')
        if winner_value is not None and winner_value != -math.inf:
            report_lines.append(f"Melhor Valor: {winner_value:.6g}")
        else:
            report_lines.append(f"Melhor Valor: N/A")
        
        winner_tempo = winner.get('tempo')
        if winner_tempo is not None:
            report_lines.append(f"Tempo de Execução: {winner_tempo:.2f} segundos ({winner_tempo/60:.2f} minutos)")
        else:
            report_lines.append(f"Tempo de Execução: N/A")
        
        if winner.get('parametros'):
            tipo, param_list = winner['parametros']
            report_lines.append(f"Tipo: {tipo}")
            report_lines.append(f"Parâmetros Ótimos: {param_list}")
    else:
        report_lines.append("❌ Nenhum resultado válido foi obtido.")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("FIM DO RELATÓRIO")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

# -----------------------------------------------------------
# Função principal de execução
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ferramenta de Otimização Automática para simulaod.exe")
    parser.add_argument("--exec", default=DEFAULT_EXECUTABLE_PATH, help="Caminho para o executável")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--optuna-trials", type=int, default=OPTUNA_DEFAULT_TRIALS)
    parser.add_argument("--optuna-timeout", type=int, default=OPTUNA_GLOBAL_TIMEOUT)
    parser.add_argument("--parallel", action="store_true", default=True, help="Ativa a avaliação paralela (ThreadPoolExecutor) para GA e PS")
    parser.add_argument("--methods", nargs="+", default=["ps", "ga", "optuna", "simplex", "combined"], help="Métodos a executar: ps (Pattern Search), ga (Genético), optuna (Bayesiana), simplex (Nelder-Mead), combined (PS->GA)")
    parser.add_argument("--goal", choices=[GOAL_MAX, GOAL_MIN], default=GOAL_MAX, help="Objetivo da otimização: max (maximizar) ou min (minimizar)")
    parser.add_argument("--out-csv", default="avaliacoes.csv", help="Caminho do arquivo de registro de avaliações")
    parser.add_argument("--out-json", default="resumo_resultados.json", help="Caminho do arquivo de resumo")
    parser.add_argument("--out-report", default="relatorio_otimizacao.txt", help="Caminho do arquivo de relatório detalhado")
    parser.add_argument("--seed", type=int, default=INITIAL_RANDOM_SEED)
    parser.add_argument("--execution-time", type=int, default=20, help="Tempo de execução em minutos (padrão: 20)")
    args = parser.parse_args()

    random.seed(args.seed)
    try:
        validate_executable(args.exec)
    except FileNotFoundError as e:
        simple_logger(str(e))
        sys.exit(1)

    simple_logger("=" * 80)
    simple_logger("=== OTIMIZADOR AUTOMÁTICO - DOIS MELHORES ALGORITMOS ===")
    simple_logger("=" * 80)
    simple_logger("💡 Pressione Ctrl+C a qualquer momento para interromper graciosamente")
    simple_logger("")
    
    global_time_start = time.time()
    all_results = []
    
    # Tempo máximo de execução (em segundos)
    MAX_EXECUTION_TIME = args.execution_time * 60  # Converte minutos para segundos
    
    # Interface de escolha do usuário
    selected_mode = None
    while selected_mode is None:
        try:
            print("\n" + "=" * 80)
            print("ESCOLHA O MODO DE OTIMIZAÇÃO:")
            print("=" * 80)
            print("1. Pattern Search (PS) - Busca exploratória sistemática")
            print("2. Algoritmo Genético (GA) - Evolução populacional")
            print("3. COMBINADO (PS + GA) - Híbrido com melhor dos dois mundos")
            print("=" * 80)
            
            user_choice = input("Digite sua escolha [1/2/3]: ").strip()
            
            if user_choice == "1":
                selected_mode = "ps"
                simple_logger("✓ Modo selecionado: Pattern Search (PS)")
            elif user_choice == "2":
                selected_mode = "ga"
                simple_logger("✓ Modo selecionado: Algoritmo Genético (GA)")
            elif user_choice == "3":
                selected_mode = "combined"
                simple_logger("✓ Modo selecionado: COMBINADO (PS + GA)")
            else:
                print("❌ Escolha inválida. Por favor, digite 1, 2 ou 3.")
        except KeyboardInterrupt:
            simple_logger("\n❌ Execução cancelada pelo usuário.")
            return
    
    simple_logger(f"\n⏱ Tempo máximo de execução: {args.execution_time} minutos ({MAX_EXECUTION_TIME} segundos)")
    simple_logger(f"🎯 Objetivo: {'MAXIMIZAR' if args.goal == GOAL_MAX else 'MINIMIZAR'}")
    simple_logger(f"📊 Estratégia: Explorar valores altos primeiro, depois refinar")
    simple_logger("")

    try:
        if selected_mode == "ps":
            # Executa apenas Pattern Search
            simple_logger("\n" + "=" * 80)
            simple_logger("🚀 EXECUTANDO: Pattern Search (PS)")
            simple_logger("=" * 80)
            
            # Ajusta parâmetros para usar o tempo disponível (20 minutos)
            ps_max_iter = 700  # Mais iterações para explorar melhor
            ps_multistarts = 2  # Mais pontos iniciais para diversidade
            
            start_time = time.time()
            res_ps = optimization_method_ps(
                args.exec, args.replicates, args.retries, args.timeout,
                max_iter=ps_max_iter, multistarts=ps_multistarts,
                parallel=args.parallel, goal=args.goal
            )
            res_ps['tempo'] = time.time() - start_time
            all_results.append(res_ps)
            
            melhor_valor = res_ps.get("melhor_valor")
            tempo_exec = res_ps.get("tempo", 0)
            if melhor_valor is not None and melhor_valor != -math.inf:
                simple_logger(f"✔ Pattern Search concluído: {melhor_valor:.6g} (tempo {tempo_exec:.1f}s)")
            else:
                simple_logger(f"⚠ Pattern Search concluído sem resultado válido (tempo {tempo_exec:.1f}s)")
        
        elif selected_mode == "ga":
            # Executa apenas Algoritmo Genético
            simple_logger("\n" + "=" * 80)
            simple_logger("🚀 EXECUTANDO: Algoritmo Genético (GA)")
            simple_logger("=" * 80)
            
            # Ajusta parâmetros para usar o tempo disponível (20 minutos)
            ga_pop_size = 40 # População maior
            ga_generations = 70  # Mais gerações para evolução
            
            start_time = time.time()
            res_ga = optimization_method_ga(
                args.exec, args.replicates, args.retries, args.timeout,
                pop_size=ga_pop_size, generations=ga_generations,
                parallel=args.parallel, goal=args.goal
            )
            res_ga['tempo'] = time.time() - start_time
            all_results.append(res_ga)
            
            melhor_valor_ga = res_ga.get("melhor_valor")
            tempo_exec_ga = res_ga.get("tempo", 0)
            if melhor_valor_ga is not None and melhor_valor_ga != -math.inf:
                simple_logger(f"✔ Algoritmo Genético concluído: {melhor_valor_ga:.6g} (tempo {tempo_exec_ga:.1f}s)")
            else:
                simple_logger(f"⚠ Algoritmo Genético concluído sem resultado válido (tempo {tempo_exec_ga:.1f}s)")
        
        elif selected_mode == "combined":
            # Executa PS e GA separadamente, depois combinado
            simple_logger("\n" + "=" * 80)
            simple_logger("🚀 MODO COMBINADO: Executando PS, GA e Híbrido")
            simple_logger("=" * 80)
            
            # Divide o tempo entre os três métodos (aproximadamente)
            time_per_method = MAX_EXECUTION_TIME / 3
            
            # 1. Pattern Search
            simple_logger("\n[1/3] Executando Pattern Search...")
            start_time = time.time()
            ps_max_iter = 700
            ps_multistarts = 3
            
            res_ps = optimization_method_ps(
                args.exec, args.replicates, args.retries, args.timeout,
                max_iter=ps_max_iter, multistarts=ps_multistarts,
                parallel=args.parallel, goal=args.goal
            )
            res_ps['tempo'] = time.time() - start_time
            all_results.append(res_ps)
            
            melhor_ps = res_ps.get("melhor_valor")
            tempo_ps = res_ps.get("tempo", 0)
            if melhor_ps is not None and melhor_ps != -math.inf:
                simple_logger(f"✔ Pattern Search concluído: {melhor_ps:.6g} (tempo {tempo_ps:.1f}s)")
            else:
                simple_logger(f"⚠ Pattern Search concluído sem resultado válido (tempo {tempo_ps:.1f}s)")
            
            # Verifica se ainda há tempo
            if time.time() - global_time_start >= MAX_EXECUTION_TIME:
                simple_logger("⏱ Tempo limite atingido. Finalizando...")
                raise KeyboardInterrupt("Tempo limite atingido")
            
            # 2. Algoritmo Genético
            simple_logger("\n[2/3] Executando Algoritmo Genético...")
            start_time = time.time()
            ga_pop_size = 25
            ga_generations = 30
            
            res_ga = optimization_method_ga(
                args.exec, args.replicates, args.retries, args.timeout,
                pop_size=ga_pop_size, generations=ga_generations,
                parallel=args.parallel, goal=args.goal
            )
            res_ga['tempo'] = time.time() - start_time
            all_results.append(res_ga)
            
            melhor_ga_comb = res_ga.get("melhor_valor")
            tempo_ga_comb = res_ga.get("tempo", 0)
            if melhor_ga_comb is not None and melhor_ga_comb != -math.inf:
                simple_logger(f"✔ Algoritmo Genético concluído: {melhor_ga_comb:.6g} (tempo {tempo_ga_comb:.1f}s)")
            else:
                simple_logger(f"⚠ Algoritmo Genético concluído sem resultado válido (tempo {tempo_ga_comb:.1f}s)")
            
            # Verifica se ainda há tempo
            if time.time() - global_time_start >= MAX_EXECUTION_TIME:
                simple_logger("⏱ Tempo limite atingido. Finalizando...")
                raise KeyboardInterrupt("Tempo limite atingido")
            
            # 3. Método Combinado (PS -> GA)
            simple_logger("\n[3/3] Executando Método Combinado (PS -> GA Híbrido)...")
            start_time = time.time()
            
            res_combined = optimization_method_combined(
                args.exec, args.replicates, args.retries, args.timeout,
                ps_max_iter=80, ga_pop_size=20, ga_generations=25,
                parallel=args.parallel, goal=args.goal
            )
            res_combined['tempo'] = time.time() - start_time
            all_results.append(res_combined)
            
            melhor_comb = res_combined.get("melhor_valor")
            tempo_comb = res_combined.get("tempo", 0)
            if melhor_comb is not None and melhor_comb != -math.inf:
                simple_logger(f"✔ Método Combinado concluído: {melhor_comb:.6g} (tempo {tempo_comb:.1f}s)")
            else:
                simple_logger(f"⚠ Método Combinado concluído sem resultado válido (tempo {tempo_comb:.1f}s)")

    except KeyboardInterrupt:
        simple_logger("\n🛑 Execução interrompida (tempo limite ou usuário)")
        simple_logger("💾 Salvando resultados parciais...")
    except Exception as e:
        simple_logger(f"❌ Erro durante execução: {e}")
        import traceback
        simple_logger(f"Detalhes do erro:\n{traceback.format_exc()}")

    # Salvar resultados
    try:
        # Salvar arquivos gerais (todos os métodos)
        save_evaluations_to_csv(args.out_csv)
        
        valid_results = [r for r in all_results if r.get("melhor_valor") not in (None, -math.inf)]
        winner = max(valid_results, key=lambda r: score_for_goal(r["melhor_valor"], args.goal)) if valid_results else None
        
        summary = {
            "timestamp": time.time(),
            "modo_selecionado": selected_mode if selected_mode else "Nenhum",
            "tempo_execucao_minutos": args.execution_time,
            "resultados": all_results,
            "vencedor": winner,
            "total_avaliacoes": len(_global_evaluations_record),
            "tempo_total_s": time.time() - global_time_start,
            "objetivo": args.goal
        }
        save_summary_to_json(args.out_json, summary)
        
        # Gerar e salvar relatório detalhado geral
        if all_results:
            report_text = generate_detailed_report(all_results, global_time_start, args)
            with open(args.out_report, "w", encoding="utf-8") as f:
                f.write(report_text)
            
            # Imprimir relatório no console
            print("\n")
            print(report_text)
        else:
            simple_logger("⚠ Nenhum resultado para gerar relatório.")
        
        # Salvar arquivos individuais para cada método executado
        simple_logger("\n� Gerando arquivos individuais por método...")
        individual_files = []
        for result in all_results:
            method_name = result.get("metodo")
            if method_name:
                try:
                    csv_file, json_file, txt_file = save_individual_method_files(
                        result, method_name, global_time_start, args
                    )
                    individual_files.append((method_name, csv_file, json_file, txt_file))
                    simple_logger(f"✓ Arquivos gerados para {method_name}")
                except Exception as e:
                    simple_logger(f"⚠ Erro ao gerar arquivos para {method_name}: {e}")
        
        # Resumo de arquivos gerados
        simple_logger(f"\n📊 ARQUIVOS GERADOS:")
        simple_logger(f"\n🔹 Arquivos Gerais (todos os métodos):")
        simple_logger(f"   - Avaliações: {args.out_csv}")
        simple_logger(f"   - Resumo JSON: {args.out_json}")
        if all_results:
            simple_logger(f"   - Relatório: {args.out_report}")
        
        if individual_files:
            simple_logger(f"\n🔹 Arquivos Individuais por Método:")
            for method_name, csv_f, json_f, txt_f in individual_files:
                simple_logger(f"\n   {method_name}:")
                simple_logger(f"      • CSV: {csv_f}")
                simple_logger(f"      • JSON: {json_f}")
                simple_logger(f"      • Relatório: {txt_f}")
        
    except Exception as e:
        simple_logger(f"❌ Erro ao salvar resultados: {e}")
        import traceback
        simple_logger(f"Detalhes do erro:\n{traceback.format_exc()}")
    
    simple_logger("\n" + "=" * 80)
    simple_logger("=== EXECUÇÃO FINALIZADA ===")
    simple_logger("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Programa encerrado pelo usuário.")
        sys.exit(130)  # Código de saída padrão para KeyboardInterrupt
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        sys.exit(1)