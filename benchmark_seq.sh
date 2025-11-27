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
JSON_FILE="${BENCHMARK_DIR}/benchmark_seq_n${N}.json"

echo "Criando diretório de benchmarks..."
mkdir -p ${BENCHMARK_DIR}

echo "Iniciando benchmark sequencial..."
echo "Tamanho (n): $N"
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

# Sanitiza strings para JSON (escapa aspas se houver)
CPU_INFO_JSON=$(printf '%s' "$CPU_INFO" | sed 's/"/\\"/g')
RAM_INFO_JSON=$(printf '%s' "$RAM_INFO" | sed 's/"/\\"/g')
GPU_INFO_JSON=$(printf '%s' "$GPU_INFO" | sed 's/"/\\"/g')

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
        exit 2
    fi
    
    # Captura "Tempo total:"
    solve_time=$(echo "$output" | grep "Tempo total:" | awk '{print $5}')
    
    if [ -z "$solve_time" ]; then
        echo "Erro na execução $i: Não foi possível capturar o tempo total."
        echo "O executável 'gauss_mod' (de 'make build-seq') precisa ter as mesmas métricas de printf."
        echo "Veja $LOG_FILE para a saída completa."
        exit 2
    fi
    
    solve_times_list+=($solve_time)
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

# --- 5. Geração do JSON ---

json_list=$(printf "%s," "${solve_times_list[@]}")
json_list="[${json_list%,}]"

cat << EOF > $JSON_FILE
{
  "parametros": {
    "n": $N,
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