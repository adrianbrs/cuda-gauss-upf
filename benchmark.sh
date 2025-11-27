#!/bin/bash

# --- Configuração de Segurança ---
set -e
set -o pipefail

# --- 1. Verificação de Parâmetros ---
if [ "$#" -ne 3 ]; then
    echo "Uso: $0 <n> <bsmode> <times>"
    echo "Exemplo: ./benchmark.sh 3000 device 5"
    exit 1
fi

N=$1
BSMODE=$2
TIMES=$3

# --- 2. Definição dos Nomes de Arquivo ---
BENCHMARK_DIR="benchmarks"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${BENCHMARK_DIR}/benchmark-n${N}-bsmode${BSMODE}-t${TIMES}-${TIMESTAMP}.log"
JSON_FILE="${BENCHMARK_DIR}/benchmark_n${N}_bsmode${BSMODE}.json"

echo "Criando diretório de benchmarks..."
mkdir -p ${BENCHMARK_DIR}

echo "Iniciando benchmark..."
echo "Tamanho (n): $N"
echo "BS mode: $BSMODE"
echo "Repetições (times): $TIMES"
echo "------------------------------------"
echo "Logs completos serão salvos em: $LOG_FILE"
echo "Resultados JSON salvos em: $JSON_FILE"
echo "------------------------------------"

# --- 3. Loop de Execução e Coleta de Dados ---
declare -a solve_times_list

# Limpa o arquivo de log para esta nova execução
> "$LOG_FILE"

# --- Coleta informações da máquina (Processador, RAM, GPU) ---
CPU_INFO="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed -E 's/.*: //')"
if [ -z "$CPU_INFO" ]; then
    CPU_INFO="$(lscpu 2>/dev/null | grep 'Model name' | sed -E 's/.*: //')"
fi

RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
if [ -n "$RAM_KB" ]; then
    # Converte KB -> GiB e arredonda para inteiro mais próximo (ex.: 15.5 -> 16)
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

# Sanitiza strings para JSON (escapa aspas se houver)
CPU_INFO_JSON=$(printf '%s' "$CPU_INFO" | sed 's/"/\\"/g')
RAM_INFO_JSON=$(printf '%s' "$RAM_INFO" | sed 's/"/\\"/g')
GPU_INFO_JSON=$(printf '%s' "$GPU_INFO" | sed 's/"/\\"/g')

# Escreve cabeçalho no log com informações da máquina
echo "==== Benchmark iniciado: $TIMESTAMP ====" >> "$LOG_FILE"
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
make build >> "$LOG_FILE" 2>&1 || { echo "Erro: falha ao compilar com make. Verifique $LOG_FILE para detalhes."; exit 2; }

for (( i=1; i<=$TIMES; i++ )); do
    echo "Executando $i/$TIMES..."
    echo "--- Execução $i/$TIMES ---" >> "$LOG_FILE"

    # Desabilita 'set -e' temporariamente para capturar saída e código
    set +e

    # Executa o binário diretamente, passando n e bsmode como argumentos
    output=$(make run n=$N bsmode=$BSMODE 2>&1 | tee -a "$LOG_FILE")
    execution_exit_code=${PIPESTATUS[0]}

    # Reabilita 'set -e'
    set -e

    echo "" >> "$LOG_FILE"

    if [ $execution_exit_code -ne 0 ]; then
        echo "Erro na execução $i: execução via make retornou código $execution_exit_code."
        echo "A saída (parcial ou completa) foi registrada em $LOG_FILE."
        exit 2
    fi

    # Captura a linha formatada pelo gauss_mod.cu: "Tempo total: <valor> s"
    solve_time=$(echo "$output" | grep "Tempo total" | awk '{print $4}')

    if [ -z "$solve_time" ]; then
        echo "Erro na execução $i: Não foi possível capturar 'Tempo total'."
        echo "Veja $LOG_FILE para a saída completa."
        exit 2
    fi

    solve_times_list+=($solve_time)
done

echo "------------------------------------"
echo "Benchmark concluído. Calculando estatísticas..."

# --- 4. Cálculo de Média e Desvio Padrão com AWK ---
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

echo "Média (solveLinearSystem): $average s"
echo "Desvio Padrão (solveLinearSystem): $std_dev s"

# --- 5. Geração do JSON ---

# Formata a lista de tempos do bash para uma lista JSON
json_list=$(printf "%s, " "${solve_times_list[@]}")
json_list="[${json_list%, }]" # Remove a vírgula final

# Escreve o arquivo JSON
cat << EOF > $JSON_FILE
{
    "parametros": {
        "n": $N,
        "bsmode": "$BSMODE",
        "times": $TIMES
    },
    "log": "$LOG_FILE",
    "info_maquina": {
        "cpu": "${CPU_INFO_JSON}",
        "ram": "${RAM_INFO_JSON}",
        "gpu": "${GPU_INFO_JSON}"
    },
    "tempos": $json_list,
    "estatisticas": {
        "media": $average,
        "desvio_padrao": $std_dev
    }
}
EOF

echo "------------------------------------"
echo "Resultados JSON salvos em: $JSON_FILE"
