#!/bin/bash

# --- Configuração de Segurança ---
set -e
set -o pipefail

# --- 1. Verificação de Parâmetros ---
if [ "$#" -ne 5 ]; then
    echo "Uso: $0 <n> <bsmode> <bmulti> <tfactor> <times>"
    echo "Exemplo: ./benchmark.sh 3000 device 4 9 5"
    exit 1
fi

N=$1
BSMODE=$2
BMULTI=$3
TFACTOR=$4
TIMES=$5

# --- 2. Definição dos Nomes de Arquivo ---
BENCHMARK_DIR="benchmarks"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${BENCHMARK_DIR}/benchmark-n${N}-bsmode${BSMODE}-bmulti${BMULTI}-tfactor${TFACTOR}-t${TIMES}-${TIMESTAMP}.log"
JSON_FILE="${BENCHMARK_DIR}/benchmark_n${N}_bsmode${BSMODE}-bmulti${BMULTI}-tfactor${TFACTOR}.json"

echo "Criando diretório de benchmarks..."
mkdir -p ${BENCHMARK_DIR}

echo "Iniciando benchmark..."
echo "Tamanho (n): $N"
echo "Back-Substitution Mode: $BSMODE"
echo "Block Multiplier: $BMULTI"
echo "Thread Factor: $TFACTOR"
echo "Repetições (times): $TIMES"
echo "------------------------------------"
echo "Logs completos serão salvos em: $LOG_FILE"
echo "Resultados JSON salvos em: $JSON_FILE"
echo "------------------------------------"

# --- 3. Loop de Execução e Coleta de Dados ---
declare -a solve_times_list

# (logs por combinação serão criados separadamente)

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

# Escreve cabeçalho no log mestre com informações da máquina
MASTER_LOG="${BENCHMARK_DIR}/benchmark-master-${TIMESTAMP}.log"
echo "==== Benchmark iniciado: $TIMESTAMP ====" > "$MASTER_LOG"
echo "Processador: $CPU_INFO" >> "$MASTER_LOG"
echo "RAM: $RAM_INFO" >> "$MASTER_LOG"
echo "Placa de Vídeo: $GPU_INFO" >> "$MASTER_LOG"
echo "------------------------------------" >> "$MASTER_LOG"

# Exibe no console também
echo "Informações da máquina:" | tee -a "$MASTER_LOG"
echo "  Processador: $CPU_INFO" | tee -a "$MASTER_LOG"
echo "  RAM: $RAM_INFO" | tee -a "$MASTER_LOG"
echo "  Placa de Vídeo: $GPU_INFO" | tee -a "$MASTER_LOG"
echo "------------------------------------" | tee -a "$MASTER_LOG"

# Compile uma vez antes do loop
echo "Compilando binário via make..." | tee -a "$MASTER_LOG"
make build >> "$MASTER_LOG" 2>&1 || { echo "Erro: falha ao compilar com make. Verifique $MASTER_LOG para detalhes."; exit 2; }

# Parse CSV lists
IFS=',' read -r -a BSMODES <<< "$BSMODE"
IFS=',' read -r -a BMULTIS <<< "$BMULTI"
IFS=',' read -r -a TFACTORS <<< "$TFACTOR"

combo_id=0
CSV_FILE="${BENCHMARK_DIR}/results-n${N}-t${TIMES}-${TIMESTAMP}.csv"
# Cabeçalho CSV (cria/reescreve no início da execução)
echo "n,bsmode,bmulti,tfactor,run,result,times,log,status,timestamp" > "$CSV_FILE"

# Progress bar helpers
print_progress() {
  # args: current total
  local cur=$1
  local tot=$2
  local width=40
  local pct=0
  if [ "$tot" -gt 0 ]; then
    pct=$(( cur * 100 / tot ))
  fi
  local filled=$(( cur * width / tot ))
  local empty=$(( width - filled ))
  local bar=""
  for ((i=0;i<filled;i++)); do bar+="#"; done
  for ((i=0;i<empty;i++)); do bar+="."; done
  # Print progress to stdout but keep logs as well
  printf "\rProgress: %d/%d [%s] %d%%\n" "$cur" "$tot" "$bar" "$pct"
}

# Global counters for progress
# Compute totals for the progress bar
global_total_combos=$(( ${#BSMODES[@]} * ${#BMULTIS[@]} * ${#TFACTORS[@]} ))
global_total_runs=$(( global_total_combos * TIMES ))
global_run_counter=0
# Print initial empty progress
print_progress 0 "$global_total_runs"

for bsmode in "${BSMODES[@]}"; do
  for bmulti in "${BMULTIS[@]}"; do
    for tfactor in "${TFACTORS[@]}"; do
      combo_id=$((combo_id+1))
      LOG_FILE_C="${BENCHMARK_DIR}/benchmark-n${N}-bsmode${bsmode}-bmulti${bmulti}-tfactor${tfactor}-t${TIMES}-${TIMESTAMP}.log"
      
      echo "=== Combo #${combo_id}: bsmode=${bsmode}, BLOCK_MULTIPLIER=${bmulti}, THREAD_FACTOR=${tfactor} ===" | tee -a "$MASTER_LOG"
      > "$LOG_FILE_C"

      # Executa TIMES repetições para esta combinação
      solve_times_list=()
      for (( run_i=1; run_i<=$TIMES; run_i++ )); do
        clear
        echo "=== Combo #${combo_id}: bsmode=${bsmode}, BLOCK_MULTIPLIER=${bmulti}, THREAD_FACTOR=${tfactor} ==="
        echo "[$combo_id] Execução $run_i/$TIMES..." | tee -a "$LOG_FILE_C"
        
        cmd="make run n=${N} bsmode=${bsmode} bmulti=${bmulti} tfactor=${tfactor}"
        echo "> $cmd"

        set +e
        output=$($cmd 2>&1 | tee -a "$LOG_FILE_C")
        execution_exit_code=${PIPESTATUS[0]}
        # update global run counter and progress bar regardless of success
        global_run_counter=$((global_run_counter+1))
        print_progress $global_run_counter "$global_total_runs"

        set -e

        echo "" >> "$LOG_FILE_C"

        if [ $execution_exit_code -ne 0 ]; then
          echo "Erro na execução combo ${combo_id} run ${run_i}: retorno ${execution_exit_code}. Veja $LOG_FILE_C." | tee -a "$MASTER_LOG"
          break
        fi

        run_ts=$(date +%s)

        solve_time=$(echo "$output" | grep "Tempo total" | awk '{print $4}')
        if [ -z "$solve_time" ]; then
          echo "Não foi possível capturar 'Tempo total' para combo ${combo_id} run ${run_i}. Veja $LOG_FILE_C." | tee -a "$MASTER_LOG"
          printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$N" "$bsmode" "$bmulti" "$tfactor" "$run_i" "" "$TIMES" "$LOG_FILE_C" "ERROR" "$run_ts" >> "$CSV_FILE"
          break
        fi

        solve_times_list+=($solve_time)

        echo "Tempo: ${solve_time} s"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$N" "$bsmode" "$bmulti" "$tfactor" "$run_i" "$solve_time" "$TIMES" "$LOG_FILE_C" "OK" "$run_ts" >> "$CSV_FILE"
      done

      if [ ${#solve_times_list[@]} -gt 0 ]; then
        stats_output=$(printf "%s\n" "${solve_times_list[@]}" | LC_NUMERIC=C awk '
            {
                sum += $1;
                sumsq += $1 * $1;
            }
            END {
                n = NR;
                if (n > 0) {
                    avg = sum / n;
                    stddev = sqrt( (sumsq / n) - (avg * avg) );
                    printf "%.6f\n%.6f\n", avg, stddev;
                } else {
                    printf "0\n0\n";
                }
            }
        ')

        average=$(echo "$stats_output" | head -n 1)
        std_dev=$(echo "$stats_output" | tail -n 1)

        json_list=$(printf "%s, " "${solve_times_list[@]}")
        json_list="[${json_list%, }]"

        echo "Combo ${combo_id} concluído." | tee -a "$MASTER_LOG"
      else
        echo "Combo ${combo_id} falhou sem medições válidas. Veja $LOG_FILE_C" | tee -a "$MASTER_LOG"
      fi
    done
  done
done

# Finalize progress bar line
printf '\n'
echo "Benchmark completo. Resultados incrementais em: $CSV_FILE" | tee -a "$MASTER_LOG"
