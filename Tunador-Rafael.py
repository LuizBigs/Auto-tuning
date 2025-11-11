#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autotuner para o executável modelo10.exe
Versão refatorada: nomes em português, argparse, Optuna+SQLite, paralelismo e logging.
Modificado para incluir prompts de sim/não antes de executar cada método de otimização.
Otimizado para velocidade: implementado paralelismo no GA e PS (quando --parallel), reduzidos hiperparâmetros default (pop_size=16, gen=20, trials=40, multistarts=2, max_iter=100), replicates=1 por default.
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
from typing import List, Tuple, Optional

# Dependências opcionais
try:
    import optuna
except Exception:
    optuna = None

# ---------------- Configuração padrão (pode ser sobrescrita via CLI) ----------------
DEFAULT_EXECUTAVEL = "modelo10.exe"
DEFAULT_REPLICATES = 1  # Reduzido para 1 para velocidade, sem comprometer muito (média de 1 = single run)
DEFAULT_RETRIES = 2
DEFAULT_TIMEOUT = 12.0
DEFAULT_OPTUNA_TRIALS = 40  # Reduzido de 60
DEFAULT_OPTUNA_TIMEOUT = 60 * 10
GA_POP_SIZE = 16  # Reduzido de 24
GA_GEN = 20  # Reduzido de 30
PS_MULTISTARTS = 2  # Reduzido de 3
SIMPLEX_MAXITER = 120
RANDOM_SEED = 42
# -----------------------------------------------------------------------------------

random.seed(RANDOM_SEED)

# armazenamento global das avaliações
_avaliacoes = []  # lista de dicts: {metodo, tipo, params, valor, tempo, rep, stdout, erro, timestamp}

# -----------------------------------------------------------
# Utilitários
# -----------------------------------------------------------
def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def extrair_numero_da_saida(texto: str) -> float:
    """
    Extrai a primeira ocorrência numérica (float ou int) em 'texto'.
    Aceita tanto '.' quanto ',' como separador decimal.
    """
    if texto is None:
        raise ValueError("Saída vazia")
    # normaliza vírgula decimal para ponto (mas só quando for parte de número)
    txt = texto.replace('.', '.')
    # detectar números com vírgula decimal (ex: 1.234,56 ou 123,45) -> padronizar para ponto
    txt = re.sub(r'(\d+),(\d+)', lambda m: f"{m.group(1)}.{m.group(2)}", txt)
    m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", txt)
    if not m:
        raise ValueError(f"Nenhum número encontrado na saída: {texto!r}")
    return float(m.group(0))

def checar_executavel(path: str):
    from shutil import which
    if which(path) is None and not Path(path).exists():
        raise FileNotFoundError(f"Executável '{path}' não encontrado no PATH nem no caminho informado.")

# -----------------------------------------------------------
# Chamadas externas e avaliação
# -----------------------------------------------------------
def chamar_externo(exec_path: str, tipo: str, params: List[int], timeout: float, retries: int) -> Tuple[float, str, float]:
    """
    Chama o executável com retries e timeout. Retorna (valor_float, stdout_or_stderr, tempo_em_segundos).
    Lança RuntimeError em falha definitiva.
    """
    args = [exec_path, str(tipo)] + [str(int(x)) for x in params]
    last_exc = None
    backoff = 0.1
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            elapsed = time.time() - t0

            if proc.returncode != 0:
                last_exc = RuntimeError(f"Processo retornou código {proc.returncode}. stderr: {stderr}")
                time.sleep(backoff)
                backoff = min(2.0, backoff * 2)
                continue

            # preferir stdout, mas aceitar stderr como fallback
            try:
                if stdout:
                    valor = extrair_numero_da_saida(stdout)
                    return valor, stdout, elapsed
                elif stderr:
                    valor = extrair_numero_da_saida(stderr)
                    return valor, stderr, elapsed
                else:
                    last_exc = RuntimeError("Saída vazia (stdout e stderr).")
                    time.sleep(backoff)
                    backoff = min(2.0, backoff * 2)
                    continue
            except ValueError as ve:
                last_exc = ve
                time.sleep(backoff)
                backoff = min(2.0, backoff * 2)
                continue

        except subprocess.TimeoutExpired:
            last_exc = RuntimeError(f"Timeout ({timeout}s) ao executar: {args}")
            time.sleep(backoff)
            backoff = min(2.0, backoff * 2)
            continue
        except Exception as e:
            last_exc = e
            time.sleep(backoff)
            backoff = min(2.0, backoff * 2)
            continue

    raise RuntimeError(f"Falha ao executar após {retries} tentativas. Último erro: {last_exc}")

def avaliar_media(exec_path: str, tipo: str, params: List[int], replicates: int, timeout: float, retries: int, metodo_label: Optional[str]=None, executor: Optional[ThreadPoolExecutor]=None):
    """
    Executa 'replicates' vezes e retorna (media, lista_valores, lista_tempos).
    Registra cada execução em _avaliacoes.
    Suporta paralelismo via executor se fornecido.
    """
    def eval_single(r):
        try:
            v, out, elapsed = chamar_externo(exec_path, tipo, params, timeout=timeout, retries=retries)
            return {
                "valor": v,
                "tempo": elapsed,
                "rep": r,
                "stdout": (out[:1000] + "...") if out and len(out) > 1000 else out,
                "erro": None
            }
        except Exception as e:
            return {
                "valor": None,
                "tempo": None,
                "rep": r,
                "stdout": None,
                "erro": str(e)
            }

    if executor and replicates > 1:
        futures = [executor.submit(eval_single, r) for r in range(replicates)]
        results = [f.result() for f in as_completed(futures)]
    else:
        results = [eval_single(r) for r in range(replicates)]

    valores = [res["valor"] for res in results if res["valor"] is not None]
    tempos = [res["tempo"] for res in results if res["tempo"] is not None]

    for res in results:
        _avaliacoes.append({
            "metodo": metodo_label,
            "tipo": tipo,
            "params": params.copy(),
            "valor": res["valor"],
            "tempo": res["tempo"],
            "rep": res["rep"],
            "stdout": res["stdout"],
            "erro": res["erro"],
            "timestamp": time.time()
        })

    if not valores:
        raise RuntimeError(f"Todas as réplicas falharam para tipo={tipo}, params={params}.")
    return mean(valores), valores, tempos

# -----------------------------------------------------------
# Métodos de otimização
# -----------------------------------------------------------
def pattern_search(exec_path, replicates, retries, timeout, max_iter=100, multistarts=PS_MULTISTARTS, parallel=False):  # max_iter reduzido para 100
    log("➡ Pattern Search (multi-start)")
    inicio = time.time()
    melhor_global = {"valor": -math.inf, "tipo": None, "params": None}

    executor = ThreadPoolExecutor(max_workers=8) if parallel else None  # Ajuste max_workers conforme necessário

    for start in range(multistarts):
        tipo = random.choice(["baixo", "medio", "alto"])
        atual = [random.randint(1, 100) for _ in range(9)]
        try:
            val_atual, _, _ = avaliar_media(exec_path, tipo, atual, replicates, timeout, retries, metodo_label="Pattern Search", executor=executor)
        except Exception as e:
            log(f"⚠ PS start {start} falhou: {e}")
            continue

        step = 16
        iter_count = 0
        while step >= 1 and iter_count < max_iter:
            candidatos = []
            for i in range(9):
                for delta in (-step, step):
                    candidato = atual.copy()
                    candidato[i] = int(min(100, max(1, candidato[i] + delta)))
                    candidatos.append((candidato, i, delta))

            melhorou = False
            val_cands = {}
            if executor:
                futures = {executor.submit(avaliar_media, exec_path, tipo, cand[0], replicates, timeout, retries, "Pattern Search"): cand for cand in candidatos}
                for future in as_completed(futures):
                    cand = futures[future]
                    try:
                        val_cand, _, _ = future.result()
                        val_cands[tuple(cand[0])] = val_cand
                        iter_count += 1
                    except Exception:
                        continue
            else:
                for cand in candidatos:
                    try:
                        val_cand, _, _ = avaliar_media(exec_path, tipo, cand[0], replicates, timeout, retries, "Pattern Search")
                        val_cands[tuple(cand[0])] = val_cand
                        iter_count += 1
                    except Exception:
                        continue

            for cand_params in val_cands:
                val_cand = val_cands[cand_params]
                if val_cand > val_atual:
                    val_atual = val_cand
                    atual = list(cand_params)
                    melhorou = True

            if not melhorou:
                step = step // 2

        if val_atual > melhor_global["valor"]:
            melhor_global.update({"valor": val_atual, "tipo": tipo, "params": atual.copy()})

    if executor:
        executor.shutdown()
    tempo = time.time() - inicio
    return {"metodo": "Pattern Search", "melhor_valor": melhor_global["valor"], "parametros": (melhor_global["tipo"], melhor_global["params"]), "tempo": tempo}

def algoritmo_genetico(exec_path, replicates, retries, timeout, pop_size=GA_POP_SIZE, generations=GA_GEN, parallel=False):
    log("➡ Algoritmo Genético (GA)")
    inicio = time.time()
    tipos = ["baixo", "medio", "alto"]

    executor = ThreadPoolExecutor(max_workers=8) if parallel else None

    def eval_ind(ind):
        try:
            v, _, _ = avaliar_media(exec_path, ind["tipo"], ind["params"], replicates, timeout, retries, metodo_label="Algoritmo Genético", executor=executor)
            ind["valor"] = v
        except Exception:
            ind["valor"] = -math.inf

    populacao = [{"tipo": random.choice(tipos), "params": [random.randint(1, 100) for _ in range(9)], "valor": -math.inf} for _ in range(pop_size)]

    if executor:
        futures = [executor.submit(eval_ind, ind) for ind in populacao]
        for f in as_completed(futures):
            f.result()
    else:
        for ind in populacao:
            eval_ind(ind)

    def torneio_select(pop, k=3):
        aspirantes = random.sample(pop, min(k, len(pop)))
        return max(aspirantes, key=lambda x: x["valor"])

    for gen in range(generations):
        populacao.sort(key=lambda x: x["valor"], reverse=True)
        n_elite = max(1, pop_size // 5)
        nova = populacao[:n_elite]
        filhos = []
        while len(nova) + len(filhos) < pop_size:
            p1 = torneio_select(populacao)
            p2 = torneio_select(populacao)
            filho = {"tipo": random.choice([p1["tipo"], p2["tipo"]]), "params": [], "valor": -math.inf}
            for a, b in zip(p1["params"], p2["params"]):
                filho["params"].append(random.choice([a, b]))
            if random.random() < 0.12:
                idx = random.randrange(9)
                filho["params"][idx] = random.randint(1, 100)
            filhos.append(filho)

        if executor:
            futures = [executor.submit(eval_ind, filho) for filho in filhos]
            for f in as_completed(futures):
                f.result()
        else:
            for filho in filhos:
                eval_ind(filho)

        nova.extend(filhos)
        populacao = nova
        melhor_gen = max(populacao, key=lambda x: x["valor"])
        log(f"  Geração {gen+1}/{generations} - melhor: {melhor_gen['valor']:.6g}")

    if executor:
        executor.shutdown()

    melhor = max(populacao, key=lambda x: x["valor"])
    tempo = time.time() - inicio
    return {"metodo": "Algoritmo Genético", "melhor_valor": melhor["valor"], "parametros": (melhor["tipo"], melhor["params"]), "tempo": tempo}

def simplex_search(exec_path, replicates, retries, timeout, maxiter=SIMPLEX_MAXITER):
    log("➡ Simplex / Nelder-Mead (opcional)")
    try:
        from scipy.optimize import minimize
    except Exception:
        log("⚠ SciPy não disponível. Pulando Simplex.")
        return {"metodo": "Simplex", "melhor_valor": -math.inf, "parametros": None, "tempo": 0}

    inicio = time.time()
    tipo = random.choice(["baixo", "medio", "alto"])

    def obj(x):
        params = [int(min(100, max(1, round(xx)))) for xx in x]
        try:
            # Para consistência, usar avaliar_media mesmo no Simplex (com replicates)
            v, _, _ = avaliar_media(exec_path, tipo, params, replicates, timeout, retries, metodo_label="Simplex")
            return -v
        except Exception:
            return 1e9

    x0 = [random.randint(1, 100) for _ in range(9)]
    res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-2})
    melhor_params = [int(min(100, max(1, round(x)))) for x in res.x]
    melhor_val = -res.fun if res.fun not in (None, float("inf")) else -math.inf
    tempo = time.time() - inicio
    return {"metodo": "Simplex", "melhor_valor": melhor_val, "parametros": (tipo, melhor_params), "tempo": tempo}

def otimizacao_bayesiana(exec_path, replicates, retries, timeout, n_trials=DEFAULT_OPTUNA_TRIALS, overall_timeout=DEFAULT_OPTUNA_TIMEOUT, storage_path="sqlite:///optuna_study.db"):
    if optuna is None:
        raise RuntimeError("Optuna não está instalado. pip install optuna")

    log("➡ Otimização Bayesiana (Optuna)")
    inicio = time.time()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    # cria/retoma estudo em SQLite para poder retomar execuções
    estudo = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, storage=storage_path, study_name="estudo_modelo10", load_if_exists=True)

    def objetivo(trial):
        tipo = trial.suggest_categorical("tipo", ["baixo", "medio", "alto"])
        params = [trial.suggest_int(f"p{i+1}", 1, 100) for i in range(9)]
        try:
            media, _, _ = avaliar_media(exec_path, tipo, params, replicates, timeout, retries, metodo_label="Optuna (bayesiana)")
            return media
        except Exception as e:
            log(f"⚠ Optuna trial falhou: {e}")
            return -1e12

    estudo.optimize(objetivo, n_trials=n_trials, timeout=overall_timeout, show_progress_bar=True)
    best = estudo.best_trial
    tipo_best = best.params.get("tipo")
    params_best = [best.params.get(f"p{i+1}") for i in range(9)]
    tempo = time.time() - inicio
    return {"metodo": "Optuna (bayesiana)", "melhor_valor": best.value, "parametros": (tipo_best, params_best), "tempo": tempo, "study": estudo}

# -----------------------------------------------------------
# I/O: salvar avaliações / resumo
# -----------------------------------------------------------
def salvar_avaliacoes_csv(caminho: str):
    campos = ["metodo", "tipo", "params", "valor", "tempo", "rep", "stdout", "erro", "timestamp"]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for linha in _avaliacoes:
            writer.writerow({
                "metodo": linha.get("metodo"),
                "tipo": linha.get("tipo"),
                "params": ",".join(map(str, linha.get("params", []))) if linha.get("params") else None,
                "valor": linha.get("valor"),
                "tempo": linha.get("tempo"),
                "rep": linha.get("rep"),
                "stdout": linha.get("stdout"),
                "erro": linha.get("erro"),
                "timestamp": linha.get("timestamp")
            })

def salvar_resumo_json(caminho: str, resumo: dict):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

# -----------------------------------------------------------
# Função para perguntar sim/não
# -----------------------------------------------------------
def perguntar_sim_nao(pergunta: str) -> bool:
    """
    Faz uma pergunta ao usuário e retorna True se a resposta for 'sim' (ou 's'), False caso contrário.
    Aceita respostas em minúsculas ou maiúsculas.
    """
    while True:
        resposta = input(f"{pergunta} (sim/nao): ").strip().lower()
        if resposta in ['sim', 's', 'yes', 'y']:
            return True
        elif resposta in ['nao', 'n', 'no']:
            return False
        else:
            print("Resposta inválida. Por favor, responda com 'sim' ou 'nao'.")

# -----------------------------------------------------------
# Função principal
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Autotuner para modelo10.exe")
    parser.add_argument("--exec", default=DEFAULT_EXECUTAVEL, help="Caminho para o executável")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--optuna-trials", type=int, default=DEFAULT_OPTUNA_TRIALS)
    parser.add_argument("--optuna-timeout", type=int, default=DEFAULT_OPTUNA_TIMEOUT)
    parser.add_argument("--parallel", action="store_true", help="Avaliar em paralelo (ThreadPoolExecutor) para GA e PS")
    parser.add_argument("--methods", nargs="+", default=["ps", "ga", "optuna", "simplex"], help="Métodos a executar: ps ga optuna simplex")
    parser.add_argument("--out-csv", default="avaliacoes.csv")
    parser.add_argument("--out-json", default="resumo_resultados.json")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    try:
        checar_executavel(args.exec)
    except FileNotFoundError as e:
        log(str(e))
        sys.exit(1)

    log("=== Comparador de Otimização - Início ===")
    inicio_total = time.time()
    resultados = []

    if "ps" in args.methods:
        if perguntar_sim_nao("Executar o método Pattern Search?"):
            try:
                res_ps = pattern_search(args.exec, args.replicates, args.retries, args.timeout, parallel=args.parallel)
                resultados.append(res_ps)
                log(f"✔ PS concluído: {res_ps['melhor_valor']:.6g} (tempo {res_ps['tempo']:.1f}s)")
            except Exception as e:
                log(f"❌ Erro no Pattern Search: {e}")
        else:
            log("Pattern Search pulado pelo usuário.")

    if "ga" in args.methods:
        if perguntar_sim_nao("Executar o método Algoritmo Genético (GA)?"):
            try:
                res_ga = algoritmo_genetico(args.exec, args.replicates, args.retries, args.timeout, parallel=args.parallel)
                resultados.append(res_ga)
                log(f"✔ GA concluído: {res_ga['melhor_valor']:.6g} (tempo {res_ga['tempo']:.1f}s)")
            except Exception as e:
                log(f"❌ Erro no Algoritmo Genético: {e}")
        else:
            log("Algoritmo Genético pulado pelo usuário.")

    if "optuna" in args.methods:
        if perguntar_sim_nao("Executar o método Otimização Bayesiana (Optuna)?"):
            try:
                storage_uri = "sqlite:///optuna_study.db"
                res_opt = otimizacao_bayesiana(args.exec, args.replicates, args.retries, args.timeout, n_trials=args.optuna_trials, overall_timeout=args.optuna_timeout, storage_path=storage_uri)
                resultados.append(res_opt)
                log(f"✔ Optuna concluído: {res_opt['melhor_valor']:.6g} (tempo {res_opt['tempo']:.1f}s)")
            except Exception as e:
                log(f"❌ Erro na Otimização Bayesiana: {e}")
        else:
            log("Otimização Bayesiana (Optuna) pulada pelo usuário.")

    if "simplex" in args.methods:
        if perguntar_sim_nao("Executar o método Simplex / Nelder-Mead?"):
            try:
                res_sx = simplex_search(args.exec, args.replicates, args.retries, args.timeout)
                resultados.append(res_sx)
                if res_sx["melhor_valor"] != -math.inf:
                    log(f"✔ Simplex concluído: {res_sx['melhor_valor']:.6g} (tempo {res_sx['tempo']:.1f}s)")
            except Exception as e:
                log(f"❌ Erro no Simplex: {e}")
        else:
            log("Simplex pulado pelo usuário.")

    # escolher vencedor entre resultados válidos
    validos = [r for r in resultados if r.get("melhor_valor") not in (None, -math.inf)]
    vencedor = max(validos, key=lambda r: r["melhor_valor"]) if validos else None

    # salvar
    salvar_avaliacoes_csv(args.out_csv)
    resumo = {
        "timestamp": time.time(),
        "resultados": resultados,
        "vencedor": vencedor,
        "total_avaliacoes": len(_avaliacoes),
        "tempo_total_s": time.time() - inicio_total
    }
    salvar_resumo_json(args.out_json, resumo)

    # imprimir resumo
    log("\n=== RESUMO FINAL ===")
    for r in resultados:
        log(f"{r['metodo']:<25} | melhor: {r['melhor_valor']:.6g} | tempo: {r['tempo']:.1f}s | params: {r['parametros']}")
    if vencedor:
        log(f"\n>> VENCEDOR: {vencedor['metodo']} -> {vencedor['melhor_valor']} {vencedor['parametros']}")
    log(f"Avaliações registradas em: {args.out_csv}")
    log(f"Resumo salvo em: {args.out_json}")
    log(f"Tempo total: {resumo['tempo_total_s']:.1f}s")
    log("=== FIM ===")

if __name__ == "__main__":
    main()
