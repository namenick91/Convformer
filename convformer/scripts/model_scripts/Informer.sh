# Usage (when sourced):
#   run_informer "<root_path>" "<data_path>" "<data>"
# Example:
#   run_informer "./data/raw/ETT-small/" "ETTh1.csv" "ETTh1"

run_informer() {
  local root_path="${1:?root_path required}"
  local data_path="${2:?data_path required}"
  local data="${3:?data required}"

  # Prediction horizons to sweep
  local -a PRED_LENS=(24 48 168 336 720)  # ETT exchange_rate weather
  # local -a PRED_LENS=(24 48 168 336)    # ECL traffic
  # local -a PRED_LENS=(24 36 48 60)      # illness

  # --------------------------------------------------------------------------------------------------------------------
  # | mixed | mixed |  mixed |  mixed |  mixed |  mixed |  mixed |  mixed |  mixed |  mixed |  mixed |  mixed |  mixed | 
  # --------------------------------------------------------------------------------------------------------------------

  # Shared args (dataset-specific are injected via the function args above)
  # ETT setup
  local -a COMMON_ARGS=(
    --is_training 1
    --model Informer
    --root_path "$root_path"
    --data_path "$data_path"
    --data "$data"
    --features MS 
    --target OT
    --freq h                 
    --seq_len 96
    --label_len 48
    --e_layers 2
    --d_layers 1
    --factor 3 
    --enc_in 7  
    --dec_in 7 
    --c_out 1  
    --use_torch_compile False
    --itr 3
  )

  # exchange_rate setup
  # local -a COMMON_ARGS=(
  #   --is_training 1
  #   --model Informer
  #   --root_path "$root_path"
  #   --data_path "$data_path"
  #   --data "$data"
  #   --features MS 
  #   --target OT
  #   --freq h                 
  #   --seq_len 96
  #   --label_len 48
  #   --e_layers 2
  #   --d_layers 1
  #   --factor 3 
  #   --enc_in 8  
  #   --dec_in 8 
  #   --c_out 1  
  #   --use_torch_compile False
  #   --itr 3
  # )

  # ----------------------------------------------------------------------------------------------------------------------
  # | univariate | univariate | univariate | univariate | univariate | univariate | univariate | univariate | univariate |
  # ----------------------------------------------------------------------------------------------------------------------

  # Shared args (dataset-specific are injected via the function args above)
  # ETT, exchange_rate setup
  # local -a COMMON_ARGS=(
  #   --is_training 1
  #   --model Informer
  #   --root_path "$root_path"
  #   --data_path "$data_path"
  #   --data "$data"
  #   --features S 
  #   --target OT
  #   --freq h                 
  #   --seq_len 96
  #   --label_len 48
  #   --e_layers 2
  #   --d_layers 1
  #   --factor 3 
  #   --enc_in 1  
  #   --dec_in 1 
  #   --c_out 1  
  #   --use_torch_compile False
  #   --itr 3
  # )

  # -------------------------------------------------------------------------------------------------------------------------
  # | multivariate | multivariate | multivariate | multivariate | multivariate | multivariate | multivariate | multivariate | 
  # -------------------------------------------------------------------------------------------------------------------------

  # Shared args (dataset-specific are injected via the function args above)
  # ETT setup
  # local -a COMMON_ARGS=(
  #   --is_training 1
  #   --model Informer
  #   --root_path "$root_path"
  #   --data_path "$data_path"
  #   --data "$data"
  #   --features M 
  #   --target OT
  #   --freq h                 
  #   --seq_len 96
  #   --label_len 48
  #   --e_layers 2
  #   --d_layers 1
  #   --factor 5 
  #   --enc_in 7  
  #   --dec_in 7 
  #   --c_out 7  
  #   --use_torch_compile False
  #   --itr 3
  # )

  # ECL setup
  # local -a COMMON_ARGS=(
  #   --is_training 1
  #   --model Informer
  #   --root_path "$root_path"
  #   --data_path "$data_path"
  #   --data "$data"
  #   --features M
  #   --target OT
  #   --freq h                 
  #   --seq_len 96
  #   --label_len 48
  #   --e_layers 2
  #   --d_layers 1
  #   --factor 5 #
  #   --enc_in 321
  #   --dec_in 321
  #   --c_out 321
  #   --use_torch_compile False
  #   --itr 3 
  # )

  # # exchange_rate setup
  # local -a COMMON_ARGS=(
  #   --is_training 1
  #   --model Informer
  #   --root_path "$root_path"
  #   --data_path "$data_path"
  #   --data "$data"
  #   --features M
  #   --target OT
  #   --freq h                 
  #   --seq_len 96
  #   --label_len 48
  #   --e_layers 2
  #   --d_layers 1
  #   --factor 3 #
  #   --enc_in 8
  #   --dec_in 8
  #   --c_out 8
  #   --use_torch_compile False
  #   --itr 3 
  # )

  # # illness
  # local -a COMMON_ARGS=(
  #   --is_training 1
  #   --model Informer
  #   --root_path "$root_path"
  #   --data_path "$data_path"
  #   --data "$data"
  #   --features M
  #   --target OT
  #   --freq h                 
  #   --seq_len 36 
  #   --label_len 18
  #   --e_layers 2
  #   --d_layers 1
  #   --factor 3
  #   --enc_in 7
  #   --dec_in 7
  #   --c_out 7
  #   --use_torch_compile False
  #   --itr 3 
  # )

  # # traffic
  # local -a COMMON_ARGS=(
  #   --is_training 1
  #   --model Informer
  #   --root_path "$root_path"
  #   --data_path "$data_path"
  #   --data "$data"
  #   --features M
  #   --target OT
  #   --freq h                 
  #   --seq_len 96
  #   --label_len 48
  #   --e_layers 2
  #   --d_layers 1
  #   --factor 3 #
  #   --enc_in 862
  #   --dec_in 862
  #   --c_out 862
  #   --use_torch_compile False
  #   --itr 3 \
  #   --train_epochs 3
  # )

  # # weather
  # local -a COMMON_ARGS=(
  #   --is_training 1
  #   --model Informer
  #   --root_path "$root_path"
  #   --data_path "$data_path"
  #   --data "$data"
  #   --features M
  #   --target OT
  #   --freq h                 
  #   --seq_len 96
  #   --label_len 48
  #   --e_layers 2
  #   --d_layers 1
  #   --factor 3 #
  #   --enc_in 21
  #   --dec_in 21
  #   --c_out 21
  #   --use_torch_compile False
  #   --itr 3 \
  # )

  # MODEL
  for pl in "${PRED_LENS[@]}"; do
    local model_id="Informer_96_${pl}_ETTh1" # manual change
    local des="informer_${pl}"

    local logfile="../logs/informer/${model_id}.log"
    mkdir -p "$(dirname "$logfile")"

    echo "==> Running ${des} (data=ETTh1, pred_len=${pl})" # manual change
    python -u ../run.py \
      "${COMMON_ARGS[@]}" \
      --pred_len "${pl}" \
      --model_id "${model_id}" \
      --des "${des}" \
      | tee "$logfile"
  done
}

# Allow running this file directly as well:
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <root_path> <data_path> <data>" >&2
    exit 64
  fi
  run_informer "$1" "$2" "$3"
fi
