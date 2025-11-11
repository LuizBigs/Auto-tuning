import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List

def parse_initial_point(point_str: str, dimension: int) -> List[float]:
    """
    Parse do ponto inicial a partir de string
    """
    if point_str:
        try:
            point = [float(x.strip()) for x in point_str.split(',')]
            if len(point) != dimension:
                print(f"Aviso: Dimensão do ponto inicial ({len(point)}) não corresponde à dimensão especificada ({dimension}). Usando o primeiro valor para todas as dimensões.")
                # Se não corresponder, usa o primeiro valor para todas as dimensões
                if point:
                    point = [point[0]] * dimension
                else:
                    return [1.0] * dimension # Padrão se a string estiver vazia mas presente
            return point
        except ValueError:
            print("Erro: Ponto inicial em formato inválido. Usando ponto inicial padrão.")
    
    # Ponto padrão se não especificado
    return [1.0] * dimension

class ParameterTunneling:
    """
    Sistema de auto-tunneling para leitura de parâmetros
    (Atividade 1: Preparar o auto-tuning para ler parâmetros)
    """
    
    def __init__(self):
        self.parameters = {}
        self.sources = []
        
    def add_command_line(self):
        """Adiciona parâmetros da linha de comando"""
        parser = argparse.ArgumentParser(description='Pattern Search Optimization')
        
        # Parâmetros gerais
        parser.add_argument('--config', type=str, help='Arquivo de configuração JSON')
        parser.add_argument('--function', type=str, required=True, 
                          help='Função objetivo (sphere, rosenbrock, etc.)')
        parser.add_argument('--dimension', type=int, default=2, 
                          help='Dimensão do problema')
        
        # Parâmetros do algoritmo
        parser.add_argument('--initial_step', type=float, default=1.0,
                          help='Tamanho inicial do passo')
        parser.add_argument('--step_reduction', type=float, default=0.5,
                          help='Fator de redução do passo')
        parser.add_argument('--tolerance', type=float, default=1e-6,
                          help='Tolerância para convergência')
        parser.add_argument('--max_iterations', type=int, default=1000,
                          help='Número máximo de iterações')
        
        # Ponto inicial
        parser.add_argument('--initial_point', type=str,
                          help='Ponto inicial como string separada por vírgulas')
        
        args = parser.parse_args(args=sys.argv[1:]) # Garante que funciona bem em ambientes de teste
        
        # Converte argumentos para dicionário
        cmd_params = vars(args)
        self.parameters.update(cmd_params)
        self.sources.append('command_line')
        
        return self
    
    def add_config_file(self, config_file: str = None):
        """Adiciona parâmetros de arquivo de configuração"""
        if config_file is None:
            config_file = self.parameters.get('config')
            
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config_params = json.load(f)
                
                # Atualiza parâmetros (prioridade: Command Line > Config File)
                for key, value in config_params.items():
                    if key not in self.parameters or self.parameters.get(key) is None:
                        self.parameters[key] = value
                
                self.sources.append('config_file')
            except Exception as e:
                print(f"Erro ao ler arquivo de configuração: {e}")
        
        return self
    
    def add_environment_variables(self, prefix: str = "PS_"):
        """Adiciona parâmetros de variáveis de ambiente"""
        
        env_params = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                param_name = key[len(prefix):].lower()
                
                # Tenta converter para número se possível
                try:
                    if '.' in value:
                        env_params[param_name] = float(value)
                    else:
                        env_params[param_name] = int(value)
                except ValueError:
                    env_params[param_name] = value
        
        # Atualiza parâmetros (prioridade: Command Line > Config File > Environment Variables)
        for key, value in env_params.items():
            if key not in self.parameters or self.parameters.get(key) is None:
                self.parameters[key] = value
        
        if env_params:
            self.sources.append('environment_variables')
        
        return self
    
    def add_defaults(self, default_params: Dict):
        """Adiciona valores padrão para parâmetros faltantes (Menor prioridade)"""
        for key, value in default_params.items():
            if key not in self.parameters or self.parameters.get(key) is None:
                self.parameters[key] = value
        
        self.sources.append('defaults')
        return self
    
    def get_parameters(self) -> Dict:
        """Retorna todos os parâmetros coletados"""
        return self.parameters.copy()
    
    def print_sources(self):
        """Imprime as fontes de parâmetros utilizadas"""
        print("Fontes de parâmetros utilizadas:")
        for source in self.sources:
            print(f"  - {source}")