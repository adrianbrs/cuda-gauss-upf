#!/bin/bash

# --- Configuração de Segurança ---
set -e
set -o pipefail

# --- 1. Verificação de Parâmetros ---
# Este script precisa de 2 parâmetros: n e times
if [ "$#" -ne 2 ]; then
    echo "Uso: $0 <n> <times>"
    echo "Exemplo: ./benchmark_seq.sh 3000 5"
    exit 1
fi

N=$1
TIMES=$2

# --- 2. Definição dos Nomes de Arquivo ---
BENCHMARK_DIR="benchmarks"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${BENCHMARK_DIR}/benchmark-seq-n${N}-t${TIMES}-${TIMESTAMP}.log"
# Arquivo CSV no mesmo formato usado por benchmark.sh
CSV_FILE="${BENCHMARK_DIR}/results-seq-n${N}-t${TIMES}-${TIMESTAMP}.csv"

echo "Criando diretório de benchmarks..."
mkdir -p ${BENCHMARK_DIR}

echo "Iniciando benchmark sequencial..."
echo "Tamanho (n): $N"
echo "Repetições (times): $TIMES"
echo "------------------------------------"
echo "Logs completos serão salvos em: $LOG_FILE"
echo "Resultados CSV incrementais serão salvos em: $CSV_FILE"
echo "------------------------------------"

# --- 3. Loop de Execução e Coleta de Dados ---
declare -a solve_times_list

# Limpa o arquivo de log para esta nova execução
> "$LOG_FILE"

# Cria/zera o CSV e escreve cabeçalho (mesmo formato de benchmark.sh)
echo "n,run,result,times,log,status,timestamp" > "$CSV_FILE"

# --- Coleta informações da máquina (Processador, RAM, GPU) ---
CPU_INFO="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed -E 's/.*: //')"
if [ -z "$CPU_INFO" ]; then
    CPU_INFO="$(lscpu 2>/dev/null | grep 'Model name' | sed -E 's/.*: //')"
fi

RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
if [ -n "$RAM_KB" ]; then
    # Converte KB -> GiB e arredonda para inteiro mais próximo
    RAM_GB=$(awk -v kb=$RAM_KB 'BEGIN{printf "%d", int((kb/1024/1024)+0.5)}')
    RAM_INFO="${RAM_GB}GiB"
else
    RAM_INFO="Unknown"
fi

GPU_INFO="Unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
elif command -v lspci >/dev/null 2>&1; then
    GPU_INFO="$(lspci 2>/dev/null | grep -iE 'vga|3d|display' | head -n1 | sed -E 's/^[^:]+: //')"
fi

# Escreve cabeçalho no log com informações da máquina
echo "==== Benchmark seq iniciado: $TIMESTAMP ====" >> "$LOG_FILE"
echo "Processador: $CPU_INFO" >> "$LOG_FILE"
echo "RAM: $RAM_INFO" >> "$LOG_FILE"
echo "Placa de Vídeo: $GPU_INFO" >> "$LOG_FILE"
echo "------------------------------------" >> "$LOG_FILE"

# Exibe no console também
echo "Informações da máquina:"
echo "  Processador: $CPU_INFO"
echo "  RAM: $RAM_INFO"
echo "  Placa de Vídeo: $GPU_INFO"
echo "------------------------------------"

# Compile uma vez antes do loop
echo "Compilando binário via make..."
make build-seq >> "$LOG_FILE" 2>&1 || { echo "Erro: falha ao compilar com make. Verifique $LOG_FILE para detalhes."; exit 2; }

for (( i=1; i<=$TIMES; i++ )); do
    echo "Executando $i/$TIMES..."
    echo "--- Execução $i/$TIMES ---" >> $LOG_FILE

    # Desabilita 'set -e' temporariamente para capturar o código de saída manualmente
    set +e
    output=$(make run-seq n=$N 2>&1 | tee -a "$LOG_FILE")
    execution_exit_code=${PIPESTATUS[0]}
    set -e

    # Adiciona uma linha em branco ao log para legibilidade
    echo "" >> "$LOG_FILE"

    # Verifica se o comando falhou
    if [ $execution_exit_code -ne 0 ]; then
        echo "Erro na execução $i: comando 'make run-seq' falhou com código de saída $execution_exit_code."
        echo "A saída (parcial ou completa) foi registrada em $LOG_FILE."
        run_ts=$(date +%s)
        # Campos bsmode,bmulti,tfactor ficam vazios para o benchmark sequencial
        printf '%s,%s,%s,%s,%s,%s,%s\n' "$N" "$i" "" "$TIMES" "$LOG_FILE" "ERROR" "$run_ts" >> "$CSV_FILE"
        continue
    fi
    
    # Captura "Tempo total:"
    solve_time=$(echo "$output" | grep "Tempo total:" | awk '{print $4}')
    
    if [ -z "$solve_time" ]; then
        echo "Erro na execução $i: Não foi possível capturar o tempo total."
        echo "O executável 'gauss_mod' (de 'make build-seq') precisa ter as mesmas métricas de printf."
        echo "Veja $LOG_FILE para a saída completa."
        exit 2
    fi
    
    solve_times_list+=($solve_time)
    
    # Grava resultado incremental no CSV (mantendo colunas compatíveis com benchmark.sh)
    run_ts=$(date +%s)
    # Campos bsmode,bmulti,tfactor ficam vazios para o benchmark sequencial
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$N" "$i" "$solve_time" "$TIMES" "$LOG_FILE" "OK" "$run_ts" >> "$CSV_FILE"
done

echo "------------------------------------"
echo "Benchmark concluído. Calculando estatísticas..."

# --- 4. Cálculo de Média e Desvio Padrão com AWK ---
# Adicionado 'LC_NUMERIC=C' para corrigir o bug do separador decimal (ponto vs vírgula)
stats_output=$(printf "%s\n" "${solve_times_list[@]}" | LC_NUMERIC=C awk '
    {
        sum += $1;
        sumsq += $1 * $1;
    }
    END {
        n = NR; # Número de Repetições
        if (n > 0) {
            avg = sum / n;
            # Fórmula do desvio padrão populacional
            stddev = sqrt( (sumsq / n) - (avg * avg) );
            printf "%.6f\n%.6f\n", avg, stddev;
        } else {
            printf "0\n0\n";
        }
    }
')

average=$(echo "$stats_output" | head -n 1)
std_dev=$(echo "$stats_output" | tail -n 1)

echo "Média: $average s"
echo "Desvio Padrão: $std_dev s"
echo "------------------------------------"