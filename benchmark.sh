#!/bin/bash

set -e
set -o pipefail

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

BENCHMARK_DIR="benchmarks"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MASTER_LOG="${BENCHMARK_DIR}/benchmark-master-n${N}-t${TIMES}-${TIMESTAMP}.log"
LOG_FILE="${BENCHMARK_DIR}/benchmark-n${N}-bsmode${BSMODE}-bmulti${BMULTI}-tfactor${TFACTOR}-t${TIMES}-${TIMESTAMP}.log"
CSV_FILE="${BENCHMARK_DIR}/results-n${N}-t${TIMES}-${TIMESTAMP}.csv"

echo "Criando diretório de benchmarks..."
mkdir -p ${BENCHMARK_DIR}

declare -a solve_times_list
# Contadores para média global incremental (usada para estimar ETA)
declare -i global_total_measured_runs=0
global_average_time=0.0
global_sum_measured_runs=0.0

# Janela para estimador robusto (trimmed mean). Ajuste conforme necessário.
RECENT_WINDOW_SIZE=30
TRIM_FRACTION=0.10
declare -a recent_runs

# --- Coleta informações da máquina (Processador, RAM, GPU) ---
echo "Coletando informações da máquina..."

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

# Compile uma vez antes do loop
echo "Compilando binário via make..." | tee -a "$MASTER_LOG"
make build >> "$MASTER_LOG" 2>&1 || { echo "Erro: falha ao compilar com make. Verifique $MASTER_LOG para detalhes."; exit 2; }

# Parse CSV lists
IFS=',' read -r -a BSMODES <<< "$BSMODE"
IFS=',' read -r -a BMULTIS <<< "$BMULTI"
IFS=',' read -r -a TFACTORS <<< "$TFACTOR"

set_id=0
# Cabeçalho CSV (cria/reescreve no início da execução)
echo "n,bsmode,bmulti,tfactor,run,result,times,log,status,timestamp" > "$CSV_FILE"

print_head() {
  echo "==== Benchmark iniciado: $TIMESTAMP ===="
  echo "Tamanho (n): $N"
  echo "Back-Substitution Mode: $BSMODE"
  echo "Block Multiplier: $BMULTI"
  echo "Thread Factor: $TFACTOR"
  echo "Repetições (times): $TIMES"
  echo "------------------------------------"
  echo "Logs completos serão salvos em: $LOG_FILE"
  echo "Resultados CSV incrementais serão salvos em: $CSV_FILE"
  echo "Informações da máquina:"
  echo "  Processador: $CPU_INFO"
  echo "  RAM: $RAM_INFO"
  echo "  Placa de Vídeo: $GPU_INFO"
  echo "------------------------------------"
}

print_head > $MASTER_LOG

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
  # Calculate ETA based on average time per execution so far and remaining runs
  local remaining=$(( tot - cur ))
  local eta_display="N/A"
  if [ "$remaining" -gt 0 ]; then
    if [ "$global_total_measured_runs" -gt 0 ]; then
      # Escolhe estimador por-rodada:
      # - Se não temos janela cheia, usa a média global
      # - Se temos janela cheia, usa trimmed mean da janela recente (robusto a outliers)
      local per_run_est="0"
      if [ ${#recent_runs[@]} -lt "$RECENT_WINDOW_SIZE" ]; then
        per_run_est="$global_average_time"
      else
        # calcular trimmed mean da janela recente
        per_run_est=$(printf "%s\n" "${recent_runs[@]}" | sort -n | awk -v tf="$TRIM_FRACTION" '
          { a[NR]=$1 }
          END {
            n=NR;
            if(n==0){ print "0"; exit }
            trim=floor(n*tf+0.0000001);
            start=trim+1; end=n-trim;
            if(end<start){ start=1; end=n }
            sum=0; cnt=0;
            for(i=start;i<=end;i++){ sum+=a[i]; cnt++ }
            if(cnt>0) printf "%.6f", sum/cnt; else print "0";
          }')
      fi

      # estima segundos restantes (arredondado)
      local est_sec
      est_sec=$(awk -v a="$per_run_est" -v r="$remaining" 'BEGIN{printf "%d", (a*r + 0.5)}')
      local hh=$((est_sec/3600))
      local mm=$(( (est_sec%3600)/60 ))
      local ss=$((est_sec%60))
      eta_display=$(printf "%02d:%02d:%02d" "$hh" "$mm" "$ss")
    fi
  fi

  # Print progress to stdout but keep logs as well
  printf "\rProgress: %d/%d [%s] %d%% ETA: %s\n" "$cur" "$tot" "$bar" "$pct" "$eta_display"
}

# Global counters for progress
# Compute totals for the progress bar
global_total_combos=$(( ${#BSMODES[@]} * ${#BMULTIS[@]} * ${#TFACTORS[@]} ))
global_total_runs=$(( global_total_combos * TIMES ))
global_run_counter=0

print_step() {
  clear
  print_head
  print_progress "$global_run_counter" "$global_total_runs"
}

print_step

for bsmode in "${BSMODES[@]}"; do
  for bmulti in "${BMULTIS[@]}"; do
    for tfactor in "${TFACTORS[@]}"; do
      set_id=$((set_id+1))
      LOG_FILE_C="${BENCHMARK_DIR}/benchmark-n${N}-bsmode${bsmode}-bmulti${bmulti}-tfactor${tfactor}-t${TIMES}-${TIMESTAMP}.log"
      
      echo "=== Combo #${set_id}: bsmode=${bsmode}, BLOCK_MULTIPLIER=${bmulti}, THREAD_FACTOR=${tfactor} ===" | tee -a "$MASTER_LOG"
      > "$LOG_FILE_C"

      # Executa TIMES repetições para esta combinação
      solve_times_list=()
      for (( run_i=1; run_i<=$TIMES; run_i++ )); do
        echo "[$set_id] Execução $run_i/$TIMES..." | tee -a "$LOG_FILE_C"
        
        cmd="make run n=${N} bsmode=${bsmode} bmulti=${bmulti} tfactor=${tfactor}"
        echo "> $cmd"

        set +e
        output=$($cmd 2>&1 | tee -a "$LOG_FILE_C")
        execution_exit_code=${PIPESTATUS[0]}
        # update global run counter and progress bar regardless of success
        global_run_counter=$((global_run_counter+1))

        print_step

        set -e

        echo "" >> "$LOG_FILE_C"

        if [ $execution_exit_code -ne 0 ]; then
          echo "Erro na execução combo ${set_id} run ${run_i}: retorno ${execution_exit_code}. Veja $LOG_FILE_C." | tee -a "$MASTER_LOG"
          break
        fi

        run_ts=$(date +%s)

        solve_time=$(echo "$output" | grep "Tempo total" | awk '{print $4}')
        if [ -z "$solve_time" ]; then
          echo "Não foi possível capturar 'Tempo total' para combo ${set_id} run ${run_i}. Veja $LOG_FILE_C." | tee -a "$MASTER_LOG"
          printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$N" "$bsmode" "$bmulti" "$tfactor" "$run_i" "" "$TIMES" "$LOG_FILE_C" "ERROR" "$run_ts" >> "$CSV_FILE"
          break
        fi

        solve_times_list+=($solve_time)

        echo "Tempo: ${solve_time} s"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$N" "$bsmode" "$bmulti" "$tfactor" "$run_i" "$solve_time" "$TIMES" "$LOG_FILE_C" "OK" "$run_ts" >> "$CSV_FILE"

        # Atualiza estruturas para estimativa robusta
        # atualiza soma e contador global incremental
        global_sum_measured_runs=$(awk -v s="$global_sum_measured_runs" -v v="$solve_time" 'BEGIN{printf "%.6f", s+v}')
        global_total_measured_runs=$((global_total_measured_runs + 1))
        global_average_time=$(awk -v s="$global_sum_measured_runs" -v c="$global_total_measured_runs" 'BEGIN{ if(c>0) printf "%.6f", s/c; else print "0" }')

        # adiciona à janela recente (circular)
        recent_runs+=("$solve_time")
        # se excedeu a janela, remove o mais antigo (shift)
        if [ ${#recent_runs[@]} -gt $RECENT_WINDOW_SIZE ]; then
          recent_runs=("${recent_runs[@]:1}")
        fi
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
        echo "Conjunto ${set_id} concluído." | tee -a "$MASTER_LOG"
      else
        echo "Conjunto ${set_id} falhou sem medições válidas. Veja $LOG_FILE_C" | tee -a "$MASTER_LOG"
      fi
    done
  done
done

# Finalize progress bar line
printf '\n'
echo "Benchmark completo. Resultados incrementais em: $CSV_FILE" | tee -a "$MASTER_LOG"
