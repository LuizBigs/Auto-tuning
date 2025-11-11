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

# Dependências opcionais
try:
    import optuna
except Exception:
    optuna = None

try:
    from scipy.optimize import minimize
except Exception:
    pass # SciPy é opcional, checado na função optimization_method_simplex

# ---------------- Configurações Padrão (Sobrescritas via CLI) ----------------
DEFAULT_EXECUTABLE_PATH = "modelo10.exe"
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
    Chama o executável (modelo10.exe) com múltiplas tentativas e timeout. 
    Retorna (valor_otimizado, saída_completa, tempo_gasto).
    """
    # Formato esperado: [exec_path, TIPO, P1, P2, P3, P4, P5, P6, P7, P8, P9]
    arguments = [exec_path, str(opt_type)] + [str(int(x)) for x in params]
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
        except Exception as e:
            return {
                "valor": None, "tempo": None, "rep": r,
                "stdout": None, "erro": str(e)
            }

    if executor and replicates > 1:
        futures = [executor.submit(eval_single, r) for r in range(replicates)]
        results = [f.result() for f in as_completed(futures)]
    else:
        results = [eval_single(r) for r in range(replicates)]

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
def optimization_method_ps(exec_path, replicates, retries, timeout, max_iter=100, multistarts=PS_MULTI_STARTS, parallel=False):
    """ Busca Padrão (Pattern Search) com múltiplas inicializações. """
    simple_logger("➡ Busca Padrão (Pattern Search, multi-start)")
    time_start = time.time()
    best_result_global: Dict[str, Any] = {"valor": -math.inf, "tipo": None, "params": None}

    executor = ThreadPoolExecutor(max_workers=8) if parallel else None

    for start in range(multistarts):
        # 9 parâmetros numéricos x2..x10 e 1 parâmetro categórico (tipo)
        opt_type = random.choice(["baixo", "medio", "alto"])
        current_params = [random.randint(1, 100) for _ in range(9)]

        try:
            current_value, _, _ = evaluate_average_performance(exec_path, opt_type, current_params, replicates, timeout, retries, method_label="Pattern Search", executor=executor)
        except Exception as e:
            simple_logger(f"⚠ PS inicialização {start} falhou: {e}")
            continue

        step_size = 16
        iteration_count = 0
        while step_size >= 1 and iteration_count < max_iter:
            candidates = []
            for i in range(9):
                for delta in (-step_size, step_size):
                    candidate_params = current_params.copy()
                    candidate_params[i] = int(min(100, max(1, candidate_params[i] + delta)))
                    candidates.append((candidate_params, i, delta))

            improved = False
            candidate_values = {}

            # Avaliação dos vizinhos (paralela ou sequencial)
            if executor:
                futures = {executor.submit(evaluate_average_performance, exec_path, opt_type, cand[0], replicates, timeout, retries, "Pattern Search"): cand for cand in candidates}
                for future in as_completed(futures):
                    cand = futures[future]
                    try:
                        value_cand, _, _ = future.result()
                        candidate_values[tuple(cand[0])] = value_cand
                        iteration_count += 1
                    except Exception:
                        continue
            else:
                for cand in candidates:
                    try:
                        value_cand, _, _ = evaluate_average_performance(exec_path, opt_type, cand[0], replicates, timeout, retries, "Pattern Search")
                        candidate_values[tuple(cand[0])] = value_cand
                        iteration_count += 1
                    except Exception:
                        continue

            # Escolhe o melhor vizinho
            for cand_params in candidate_values:
                value_cand = candidate_values[cand_params]
                if value_cand > current_value: # Maximização
                    current_value = value_cand
                    current_params = list(cand_params)
                    improved = True

            if not improved:
                step_size = step_size // 2 # Reduz o passo

        if current_value > best_result_global["valor"]:
            best_result_global.update({"valor": current_value, "tipo": opt_type, "params": current_params.copy()})

    if executor:
        executor.shutdown()

    elapsed_time = time.time() - time_start
    return {"metodo": "Pattern Search", "melhor_valor": best_result_global["valor"], "parametros": (best_result_global["tipo"], best_result_global["params"]), "tempo": elapsed_time}

def optimization_method_ga(exec_path, replicates, retries, timeout, pop_size=GA_POPULATION_SIZE, generations=GA_GENERATIONS, parallel=False):
    """ Algoritmo Genético (Genetic Algorithm). """
    simple_logger("➡ Algoritmo Genético (GA)")
    time_start = time.time()
    available_types = ["baixo", "medio", "alto"]

    executor = ThreadPoolExecutor(max_workers=8) if parallel else None

    def evaluate_individual(individual):
        try:
            v, _, _ = evaluate_average_performance(exec_path, individual["tipo"], individual["params"], replicates, timeout, retries, method_label="Algoritmo Genético", executor=executor if replicates > 1 else None)
            individual["valor"] = v
        except Exception:
            individual["valor"] = -math.inf # Penalidade

    # Inicialização da População
    population = [{"tipo": random.choice(available_types), "params": [random.randint(1, 100) for _ in range(9)], "valor": -math.inf} for _ in range(pop_size)]

    # Avaliação da População Inicial
    if executor and parallel:
        futures = [executor.submit(evaluate_individual, ind) for ind in population]
        for f in as_completed(futures): f.result()
    else:
        for ind in population: evaluate_individual(ind)

    def tournament_selection(pop, k=3):
        candidates = random.sample(pop, min(k, len(pop)))
        return max(candidates, key=lambda x: x["valor"])

    # Loop de Gerações
    for gen in range(generations):
        population.sort(key=lambda x: x["valor"], reverse=True)
        n_elite = max(1, pop_size // 5)
        new_population = population[:n_elite]
        offspring = []
        while len(new_population) + len(offspring) < pop_size:
            # Seleção e Cruzamento
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = {"tipo": random.choice([parent1["tipo"], parent2["tipo"]]), "params": [], "valor": -math.inf}
            for a, b in zip(parent1["params"], parent2["params"]):
                child["params"].append(random.choice([a, b]))

            # Mutação
            if random.random() < 0.12:
                idx = random.randrange(9)
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
        best_of_gen = max(population, key=lambda x: x["valor"])
        simple_logger(f"  Geração {gen+1}/{generations} - melhor: {best_of_gen['valor']:.6g}")

    if executor and parallel:
        executor.shutdown()

    best_individual = max(population, key=lambda x: x["valor"])
    elapsed_time = time.time() - time_start
    return {"metodo": "Algoritmo Genético", "melhor_valor": best_individual["valor"], "parametros": (best_individual["tipo"], best_individual["params"]), "tempo": elapsed_time}

def optimization_method_simplex(exec_path, replicates, retries, timeout, maxiter=SIMPLEX_MAX_ITERATIONS):
    """ Simplex / Nelder-Mead usando SciPy para otimização contínua. """
    simple_logger("➡ Simplex / Nelder-Mead (opcional)")
    if 'minimize' not in locals():
        simple_logger("⚠ SciPy não está instalado. Pulando Simplex.")
        return {"metodo": "Simplex", "melhor_valor": -math.inf, "parametros": None, "tempo": 0}

    time_start = time.time()
    opt_type = random.choice(["baixo", "medio", "alto"])

    def objective_function_scipy(x_float_array):
        # Converte o array de floats do SciPy para inteiros [1-100] (entrada do .exe)
        params = [int(min(100, max(1, round(xx)))) for xx in x_float_array]
        try:
            value, _, _ = evaluate_average_performance(exec_path, opt_type, params, replicates, timeout, retries, method_label="Simplex")
            return -value # Retorna o negativo para MINIMIZAR (já que SciPy minimiza)
        except Exception:
            return 1e9 # Penalidade alta

    # Ponto inicial x0 para os 9 parâmetros (float/int)
    initial_point_x0 = [random.randint(1, 100) for _ in range(9)]

    # Executa a otimização
    result = minimize(objective_function_scipy, initial_point_x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    
    # Processa o resultado final
    best_params = [int(min(100, max(1, round(x)))) for x in result.x]
    best_value = -result.fun if result.fun not in (None, float("inf")) else -math.inf
    
    elapsed_time = time.time() - time_start
    return {"metodo": "Simplex", "melhor_valor": best_value, "parametros": (opt_type, best_params), "tempo": elapsed_time}

def optimization_method_optuna(exec_path, replicates, retries, timeout, n_trials=OPTUNA_DEFAULT_TRIALS, overall_timeout=OPTUNA_GLOBAL_TIMEOUT, storage_path="sqlite:///optuna_study.db"):
    """ Otimização Bayesiana (Optuna). """
    if optuna is None:
        raise RuntimeError("Optuna não está instalado. Por favor, instale: pip install optuna")

    simple_logger("➡ Otimização Bayesiana (Optuna)")
    time_start = time.time()
    sampler = optuna.samplers.TPESampler(seed=INITIAL_RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    try:
        # Cria ou retoma o estudo em SQLite
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                     storage=storage_path, study_name="optimization_study_model", load_if_exists=True)
    except Exception as e:
        simple_logger(f"⚠ Erro ao iniciar Optuna com SQLite: {e}. Optuna desabilitado.")
        return {"metodo": "Optuna (bayesiana)", "melhor_valor": -math.inf, "parametros": None, "tempo": 0}


    def objective_optuna(trial):
        # Define o espaço de busca (hiperparâmetros)
        opt_type = trial.suggest_categorical("tipo", ["baixo", "medio", "alto"])
        # x2..x10: 9 parâmetros (inteiros 1-100)
        params = [trial.suggest_int(f"p{i+1}", 1, 100) for i in range(9)]

        try:
            avg_value, _, _ = evaluate_average_performance(exec_path, opt_type, params, replicates, timeout, retries, method_label="Optuna (bayesiana)")
            return avg_value # Retorna o valor para maximização
        except Exception as e:
            simple_logger(f"⚠ Optuna trial {trial.number} falhou: {e}")
            return -1e12 # Penalidade

    study.optimize(objective_optuna, n_trials=n_trials, timeout=overall_timeout, show_progress_bar=True)
    best_trial = study.best_trial
    
    # Extrai o melhor conjunto de parâmetros
    best_type = best_trial.params.get("tipo")
    best_params_list = [best_trial.params.get(f"p{i+1}") for i in range(9)]

    elapsed_time = time.time() - time_start
    return {"metodo": "Optuna (bayesiana)", "melhor_valor": best_trial.value, "parametros": (best_type, best_params_list), "tempo": elapsed_time, "study": study}

# -----------------------------------------------------------
# I/O: salvar avaliações / resumo
# -----------------------------------------------------------
def save_evaluations_to_csv(filepath: str):
    """ Salva o registro completo de todas as avaliações no formato CSV. """
    fields = ["metodo", "tipo", "params", "valor", "tempo", "rep", "stdout", "erro", "timestamp"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in _global_evaluations_record:
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
        json.dump(summary, f, ensure_ascii=False, indent=2)

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
# Função principal de execução
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ferramenta de Otimização Automática para modelo10.exe")
    parser.add_argument("--exec", default=DEFAULT_EXECUTABLE_PATH, help="Caminho para o executável")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--optuna-trials", type=int, default=OPTUNA_DEFAULT_TRIALS)
    parser.add_argument("--optuna-timeout", type=int, default=OPTUNA_GLOBAL_TIMEOUT)
    parser.add_argument("--parallel", action="store_true", help="Ativa a avaliação paralela (ThreadPoolExecutor) para GA e PS")
    parser.add_argument("--methods", nargs="+", default=["ps", "ga", "optuna", "simplex"], help="Métodos a executar: ps (Pattern Search), ga (Genético), optuna (Bayesiana), simplex (Nelder-Mead)")
    parser.add_argument("--out-csv", default="avaliacoes.csv", help="Caminho do arquivo de registro de avaliações")
    parser.add_argument("--out-json", default="resumo_resultados.json", help="Caminho do arquivo de resumo")
    parser.add_argument("--seed", type=int, default=INITIAL_RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    try:
        validate_executable(args.exec)
    except FileNotFoundError as e:
        simple_logger(str(e))
        sys.exit(1)

    simple_logger("=== Comparador de Otimização - Início ===")
    global_time_start = time.time()
    all_results = []

    # Execução e perguntas para cada método
    if "ps" in args.methods:
        if ask_yes_no("Executar o método Busca Padrão (Pattern Search)?"):
            try:
                res_ps = optimization_method_ps(args.exec, args.replicates, args.retries, args.timeout, parallel=args.parallel)
                all_results.append(res_ps)
                simple_logger(f"✔ PS concluído: {res_ps['melhor_valor']:.6g} (tempo {res_ps['tempo']:.1f}s)")
            except Exception as e:
                simple_logger(f"❌ Erro no Pattern Search: {e}")

    if "ga" in args.methods:
        if ask_yes_no("Executar o método Algoritmo Genético (GA)?"):
            try:
                res_ga = optimization_method_ga(args.exec, args.replicates, args.retries, args.timeout, parallel=args.parallel)
                all_results.append(res_ga)
                simple_logger(f"✔ GA concluído: {res_ga['melhor_valor']:.6g} (tempo {res_ga['tempo']:.1f}s)")
            except Exception as e:
                simple_logger(f"❌ Erro no Algoritmo Genético: {e}")

    if "optuna" in args.methods:
        if ask_yes_no("Executar o método Otimização Bayesiana (Optuna)?"):
            try:
                storage_uri = "sqlite:///optuna_study.db"
                res_opt = optimization_method_optuna(args.exec, args.replicates, args.retries, args.timeout, n_trials=args.optuna_trials, overall_timeout=args.optuna_timeout, storage_path=storage_uri)
                all_results.append(res_opt)
                if res_opt.get("melhor_valor") != -math.inf:
                    simple_logger(f"✔ Optuna concluído: {res_opt['melhor_valor']:.6g} (tempo {res_opt['tempo']:.1f}s)")
            except Exception as e:
                simple_logger(f"❌ Erro na Otimização Bayesiana: {e}")

    if "simplex" in args.methods:
        if ask_yes_no("Executar o método Simplex / Nelder-Mead?"):
            try:
                res_sx = optimization_method_simplex(args.exec, args.replicates, args.retries, args.timeout)
                all_results.append(res_sx)
                if res_sx.get("melhor_valor") != -math.inf:
                    simple_logger(f"✔ Simplex concluído: {res_sx['melhor_valor']:.6g} (tempo {res_sx['tempo']:.1f}s)")
            except Exception as e:
                simple_logger(f"❌ Erro no Simplex: {e}")

    # escolher vencedor
    valid_results = [r for r in all_results if r.get("melhor_valor") not in (None, -math.inf)]
    winner = max(valid_results, key=lambda r: r["melhor_valor"]) if valid_results else None

    # salvar
    save_evaluations_to_csv(args.out_csv)
    summary = {
        "timestamp": time.time(),
        "resultados": all_results,
        "vencedor": winner,
        "total_avaliacoes": len(_global_evaluations_record),
        "tempo_total_s": time.time() - global_time_start
    }
    save_summary_to_json(args.out_json, summary)

    # imprimir resumo
    simple_logger("\n=== RESUMO FINAL ===")
    for r in all_results:
        params_str = f"({r['parametros'][0]}, {r['parametros'][1]})" if r['parametros'] else "N/A"
        simple_logger(f"{r['metodo']:<25} | melhor: {r['melhor_valor']:.6g} | tempo: {r['tempo']:.1f}s | params: {params_str}")
    if winner:
        simple_logger(f"\n>> VENCEDOR: {winner['metodo']} -> {winner['melhor_valor']} {winner['parametros']}")
    simple_logger(f"Avaliações registradas em: {args.out_csv}")
    simple_logger(f"Resumo salvo em: {args.out_json}")
    simple_logger(f"Tempo total: {summary['tempo_total_s']:.1f}s")
    simple_logger("=== FIM ===")

if __name__ == "__main__":
    main()