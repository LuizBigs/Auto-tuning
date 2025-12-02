# 🚀 Sistema de Auto-tuning para Otimização de Parâmetros

Sistema inteligente de otimização automática que utiliza múltiplos algoritmos para encontrar os melhores parâmetros para executáveis externos.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Algoritmos Disponíveis](#algoritmos-disponíveis)
- [Arquivos Gerados](#arquivos-gerados)
- [Parâmetros de Linha de Comando](#parâmetros-de-linha-de-comando)
- [Exemplos de Uso](#exemplos-de-uso)
- [Estrutura de Arquivos](#estrutura-de-arquivos)

---

## 🎯 Visão Geral

O **Auto-tuning** é um sistema de otimização que automaticamente encontra os melhores valores para 10 parâmetros (de 1 a 1000) que maximizam ou minimizam a saída de um programa executável externo (`provab2.exe`).

### Como Funciona?

1. **Entrada**: 10 parâmetros numéricos (P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
2. **Processamento**: Algoritmos de otimização testam diferentes combinações
3. **Execução**: Cada combinação é executada no programa externo
4. **Avaliação**: O sistema analisa os resultados
5. **Saída**: Melhor combinação de parâmetros encontrada

```
┌─────────────────┐
│  Tunador.py     │
│  (Otimizador)   │
└────────┬────────┘
         │
         ↓ [P1, P2, P3, P4, P5]
┌─────────────────┐
│  simulado.exe   │
│  (Seu programa) │
└────────┬────────┘
         │
         ↓ Valor de saída
┌─────────────────┐
│  Resultado      │
│  Otimizado      │
└─────────────────┘
```

---

## ✨ Características

### 🔍 **Múltiplos Algoritmos**
- **Pattern Search (PS)**: Busca exploratória sistemática
- **Algoritmo Genético (GA)**: Evolução populacional
- **Método Combinado**: PS → GA híbrido

### 📊 **Monitoramento em Tempo Real**
- Progresso exibido a cada 50 avaliações
- Mostra melhor valor encontrado instantaneamente
- Indicadores visuais com emojis (🔍 🧬 ✨ 🏁 📈)

### ⚡ **Execução Paralela**
- Avaliação simultânea de múltiplos parâmetros
- Usa ThreadPoolExecutor para acelerar o processo
- Ativado por padrão

### 📁 **Relatórios Detalhados**
- Arquivos CSV com todas as avaliações
- Resumos JSON estruturados
- Relatórios em texto formatado
- **Arquivos individuais por algoritmo**

### 🎯 **Estratégia Inteligente**
- Explora valores altos primeiro (para maximização)
- Step adaptativo (começa grande, diminui gradualmente)
- Diversificação com múltiplos pontos iniciais

---

## 📦 Requisitos

### Sistema Operacional
- ✅ Windows (testado)
- ✅ Linux
- ✅ macOS

### Software Necessário
```bash
Python 3.7+
```

### Bibliotecas Python (Incluídas no Python padrão)
- `argparse`
- `csv`
- `json`
- `math`
- `random`
- `subprocess`
- `time`
- `concurrent.futures`

### Bibliotecas Opcionais
```bash
# Para algoritmo Simplex/Nelder-Mead
pip install scipy

# Para otimização Bayesiana
pip install optuna
```

---

## 🔧 Instalação

### 1. Clone o Repositório
```bash
git clone https://github.com/LuizBigs/Auto-tuning.git
cd Auto-tuning
```

### 2. Certifique-se que tem Python
```bash
python --version
```

### 3. Coloque seu Executável
Coloque o arquivo `provab2.exe` no mesmo diretório do `Tunador.py`

---

## 🚀 Como Usar

### Execução Básica

```bash
python Tunador.py
```

O sistema irá perguntar qual modo você deseja executar:

```
ESCOLHA O MODO DE OTIMIZAÇÃO:
================================================================================
1. Pattern Search (PS) - Busca exploratória sistemática
2. Algoritmo Genético (GA) - Evolução populacional
3. COMBINADO (PS + GA) - Híbrido com melhor dos dois mundos
================================================================================
Digite sua escolha [1/2/3]:
```

### Durante a Execução

Você verá o progresso em tempo real:

```
[2025-12-02 10:30:15] 🔍 PS Start 1/2 - Inicial: [500, 500, 500, 500, 500, 500, 500, 500, 500, 500] = 5000.00
[2025-12-02 10:30:20] ✨ NOVO MELHOR: [800, 800, 800, 800, 800, 800, 800, 800, 800, 800] = 8000.00
[2025-12-02 10:30:45] 📈 Progresso: 50 avaliações, 30.5s decorridos, step=125, atual=8000.00
[2025-12-02 10:35:10] ✨ NOVO MELHOR: [850, 820, 880, 840, 860, 830, 870, 810, 890, 825] = 8475.00
[2025-12-02 10:40:00] 🏁 Pattern Search finalizado: Melhor=8475.00 em [850, 820, 880, 840, 860, 830, 870, 810, 890, 825]
```

### Interromper a Execução

Pressione `Ctrl+C` a qualquer momento para parar graciosamente. O sistema salvará os resultados parciais.

---

## 🧠 Algoritmos Disponíveis

### 1. **Pattern Search (PS)** 🔍

**Como funciona:**
- Começa de um ponto inicial
- Testa vizinhos em todas as direções
- Move-se para o melhor vizinho encontrado
- Reduz o tamanho do passo gradualmente

**Parâmetros:**
- `ps_max_iter`: 700 iterações
- `ps_multistarts`: 2 pontos iniciais
- `step_size`: Começa em 250, reduz pela metade

**Estratégia de Inicialização:**
- Start 0: [500, 500, 500, 500, 500, 500, 500, 500, 500, 500] - Meio do espaço
- Start 1: [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000] - Valores máximos
- Start 2+: Aleatório [700-1000] - Valores altos variados

**Melhor para:**
- Exploração sistemática do espaço
- Encontrar ótimos locais
- Convergência rápida

---

### 2. **Algoritmo Genético (GA)** 🧬

**Como funciona:**
- Cria população inicial de soluções
- Seleciona os melhores indivíduos (elitismo)
- Cruza indivíduos para gerar filhos
- Aplica mutações aleatórias
- Evolui por múltiplas gerações

**Parâmetros:**
- `ga_pop_size`: 40 indivíduos
- `ga_generations`: 70 gerações
- `mutation_rate`: 12%
- `elite_ratio`: 20%

**Operadores:**
- **Seleção**: Torneio (k=3)
- **Cruzamento**: Uniforme (escolhe aleatoriamente de cada pai)
- **Mutação**: Substitui um parâmetro aleatório

**Melhor para:**
- Exploração global do espaço
- Evitar ótimos locais
- Diversidade de soluções

---

### 3. **Método Combinado (PS + GA)** 🔥

**Como funciona:**
1. Executa Pattern Search rápido para encontrar uma boa região
2. Usa o resultado do PS como "semente" para o GA
3. GA refina e explora ao redor da semente
4. Retorna o melhor entre PS e GA

**Parâmetros:**
- PS: 80 iterações, 1 start
- GA: 20 indivíduos, 25 gerações

**Melhor para:**
- Combinar exploração e refinamento
- Convergência rápida com qualidade
- Aproveitar pontos fortes de ambos

---

## 📁 Arquivos Gerados

### Arquivos Gerais (Todos os Métodos)

#### `avaliacoes.csv`
Todas as avaliações de todos os métodos executados.

**Colunas:**
- `metodo`: Nome do algoritmo
- `tipo`: Tipo de configuração (sempre "default")
- `params`: Valores dos 5 parâmetros (separados por vírgula)
- `valor`: Resultado da avaliação
- `tempo`: Tempo de execução (segundos)
- `rep`: Número da réplica
- `stdout`: Saída do programa
- `erro`: Mensagem de erro (se houver)
- `timestamp`: Momento da avaliação

**Exemplo:**
```csv
metodo,tipo,params,valor,tempo,rep,stdout,erro,timestamp
Pattern Search,default,"800,800,800,800,800,800,800,800,800,800",8000.0,0.15,0,8000.00,,1701518400.123
```

---

#### `resumo_resultados.json`
Resumo comparativo de todos os métodos.

**Estrutura:**
```json
{
  "timestamp": 1701518400.123,
  "modo_selecionado": "ps",
  "tempo_execucao_minutos": 20,
  "resultados": [
    {
      "metodo": "Pattern Search",
      "melhor_valor": 8475.00,
      "parametros": ["default", [850, 820, 880, 840, 860, 830, 870, 810, 890, 825]],
      "tempo": 1200.5
    }
  ],
  "vencedor": {
    "metodo": "Pattern Search",
    "melhor_valor": 8475.00
  },
  "total_avaliacoes": 583,
  "tempo_total_s": 1200.5,
  "objetivo": "max"
}
```

---

#### `relatorio_otimizacao.txt`
Relatório formatado em texto com todos os resultados.

**Conteúdo:**
```
================================================================================
RELATÓRIO DETALHADO DE OTIMIZAÇÃO
================================================================================
Data/Hora: 2025-12-02 10:40:00
Tempo Total de Execução: 1200.50 segundos (20.01 minutos)
Objetivo: MAXIMIZAR
Total de Avaliações do Modelo: 583
================================================================================

RESULTADOS POR MÉTODO:
--------------------------------------------------------------------------------

1. Pattern Search
   Melhor Valor: 8475.0
   Número de Tentativas/Avaliações: 583
   Tempo de Execução: 1200.50 segundos (20.01 minutos)
   Tipo: default
   Parâmetros: [850, 820, 880, 840, 860, 830, 870, 810, 890, 825]

--------------------------------------------------------------------------------

🏆 MELHOR RESULTADO GERAL:
--------------------------------------------------------------------------------
Método Vencedor: Pattern Search
Melhor Valor: 8475.0
Número de Tentativas/Avaliações: 583
Tempo de Execução: 1200.50 segundos (20.01 minutos)
Tipo: default
Parâmetros Ótimos: [850, 820, 880, 840, 860, 830, 870, 810, 890, 825]

================================================================================
FIM DO RELATÓRIO
================================================================================
```

---

### Arquivos Individuais por Algoritmo

Para cada método executado, são gerados 3 arquivos separados:

#### `avaliacoes_pattern_search.csv`
Apenas as avaliações do Pattern Search

#### `resumo_pattern_search.json`
Resumo específico do Pattern Search
```json
{
  "timestamp": 1701518400.123,
  "metodo": "Pattern Search",
  "resultado": {
    "metodo": "Pattern Search",
    "melhor_valor": 8475.00,
    "parametros": ["default", [850, 820, 880, 840, 860, 830, 870, 810, 890, 825]],
    "tempo": 1200.5
  },
  "total_avaliacoes": 583,
  "tempo_total_s": 1200.5,
  "objetivo": "max"
}
```

#### `relatorio_pattern_search.txt`
Relatório detalhado apenas do Pattern Search

**Mesma estrutura para outros algoritmos:**
- `avaliacoes_algoritmo_genético.csv`
- `resumo_algoritmo_genético.json`
- `relatorio_algoritmo_genético.txt`

---

## ⚙️ Parâmetros de Linha de Comando

### Parâmetros Básicos

```bash
python Tunador.py [opções]
```

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--exec` | Caminho para o executável | `provab2.exe` |
| `--execution-time` | Tempo de execução (minutos) | `20` |
| `--goal` | Objetivo: `max` ou `min` | `max` |
| `--parallel` | Ativa execução paralela | `True` |
| `--seed` | Seed para números aleatórios | `42` |

### Parâmetros de Avaliação

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--replicates` | Réplicas por avaliação | `1` |
| `--retries` | Tentativas em caso de falha | `2` |
| `--timeout` | Timeout por execução (segundos) | `12.0` |

### Parâmetros de Saída

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--out-csv` | Arquivo CSV de avaliações | `avaliacoes.csv` |
| `--out-json` | Arquivo JSON de resumo | `resumo_resultados.json` |
| `--out-report` | Arquivo de relatório | `relatorio_otimizacao.txt` |

---

## 💡 Exemplos de Uso

### Exemplo 1: Execução Padrão (20 minutos)
```bash
python Tunador.py
```

### Exemplo 2: Execução Rápida (5 minutos)
```bash
python Tunador.py --execution-time 5
```

### Exemplo 3: Minimização (encontrar valor mínimo)
```bash
python Tunador.py --goal min
```

### Exemplo 4: Com executável personalizado
```bash
python Tunador.py --exec meu_programa.exe
```

### Exemplo 5: Sem paralelização
```bash
python Tunador.py --parallel False
```

### Exemplo 6: Com timeout maior
```bash
python Tunador.py --timeout 30
```

### Exemplo 7: Arquivos de saída personalizados
```bash
python Tunador.py --out-csv meus_dados.csv --out-json meu_resumo.json --out-report meu_relatorio.txt
```

---

## 📂 Estrutura de Arquivos

```
Auto-tuning/
│
├── Tunador.py                      # Script principal
├── provab2.exe                     # Executável a ser otimizado
├── README.md                       # Este arquivo
│
├── avaliacoes.csv                  # Todas as avaliações (geral)
├── resumo_resultados.json          # Resumo comparativo (geral)
├── relatorio_otimizacao.txt        # Relatório formatado (geral)
│
├── avaliacoes_pattern_search.csv  # Avaliações do PS
├── resumo_pattern_search.json     # Resumo do PS
├── relatorio_pattern_search.txt   # Relatório do PS
│
├── avaliacoes_algoritmo_genético.csv   # Avaliações do GA
├── resumo_algoritmo_genético.json      # Resumo do GA
└── relatorio_algoritmo_genético.txt    # Relatório do GA
```

---

## 🔬 Detalhes Técnicos

### Formato de Entrada do Executável

O executável deve aceitar 10 parâmetros numéricos:

```bash
provab2.exe P1 P2 P3 P4 P5 P6 P7 P8 P9 P10
```

Onde cada parâmetro é um inteiro entre 1 e 1000.

### Formato de Saída do Executável

O executável deve imprimir um valor numérico na saída padrão (stdout):

```
8475.00
```

O sistema extrairá automaticamente o primeiro número encontrado.

---

## 🎛️ Configurações Avançadas

### Ajustar Parâmetros do Pattern Search

Edite no código `Tunador.py` linha ~857:

```python
ps_max_iter = 700  # Número de iterações
ps_multistarts = 2  # Pontos iniciais
```

### Ajustar Parâmetros do Algoritmo Genético

Edite no código `Tunador.py` linha ~876:

```python
ga_pop_size = 40    # Tamanho da população
ga_generations = 70  # Número de gerações
```

### Ajustar Step Inicial do Pattern Search

Edite no código `Tunador.py` linha ~268:

```python
step_size = 250  # Tamanho inicial do passo (proporcional a 1-1000)
```

---

## 🐛 Solução de Problemas

### Problema: "Executável não encontrado"
**Solução:** Certifique-se que `provab2.exe` está no mesmo diretório ou use `--exec` com caminho completo.

### Problema: "Timeout excedido"
**Solução:** Aumente o timeout com `--timeout 30` (em segundos).

### Problema: "Nenhum resultado válido"
**Solução:** Verifique se o executável está funcionando manualmente e retornando valores numéricos.

### Problema: Execução muito lenta
**Solução:** 
- Certifique-se que `--parallel True` está ativado
- Reduza `--execution-time`
- Reduza o número de iterações/gerações no código

---

## 📊 Interpretando os Resultados

### Valor Melhor vs. Tempo

- **Pattern Search**: Convergência rápida, mas pode ficar preso em ótimos locais
- **Algoritmo Genético**: Exploração global, mas pode demorar mais para convergir
- **Combinado**: Melhor dos dois mundos

### Analisando os Parâmetros Ótimos

Os parâmetros encontrados são os valores que maximizam (ou minimizam) a saída do seu programa. Use-os como ponto de partida para análises futuras.

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abrir um Pull Request

---

## 📝 Licença

Este projeto é de código aberto e está disponível sob a licença MIT.

---

## 👨‍💻 Autor

**Luiz Meneses**
- GitHub: [@LuizBigs](https://github.com/LuizBigs)

---

## 📧 Suporte

Se tiver dúvidas ou problemas:

1. Abra uma [Issue no GitHub](https://github.com/LuizBigs/Auto-tuning/issues)
2. Descreva o problema detalhadamente
3. Inclua logs e arquivos de saída se possível

---

## 🎉 Agradecimentos

Obrigado por usar o Sistema de Auto-tuning! 

**Happy Optimizing! 🚀**
