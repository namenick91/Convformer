# pyenv activate science-venv && cd Documents/VScodeProjects/LSTF-Transformers-paper/convformer/scripts && bash main.sh

# Resolve this script’s directory so relative paths are robust
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Path to ablation scripts
ABLAT1_SH="${SCRIPT_DIR}/Ablations/ablation1.sh"
ABLAT2_SH="${SCRIPT_DIR}/Ablations/ablation2.sh"
ABLAT3_SH="${SCRIPT_DIR}/Ablations/ablation3.sh"

source "${SCRIPT_DIR}/model_scripts/Autoformer.sh"
source "${SCRIPT_DIR}/model_scripts/Convformer.sh"
source "${SCRIPT_DIR}/model_scripts/Performer.sh"
source "${SCRIPT_DIR}/model_scripts/Informer.sh"


# -------------------------------------------------------------
# | ABLATION1 | ABLATION1 | ABLATION1 | ABLATION1 | ABLATION1 |
# -------------------------------------------------------------

# Custom
# bash "$ABLAT1_SH" "${SCRIPT_DIR}/../data/raw/synth/" "LOCAL_MOTIFS.csv" "custom"

# ETTh1 (hourly)
# bash "$ABLAT1_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1" # ~1h43m  № 1 ------------------------ DONE

# ETTh2 (hourly)
# bash "$ABLAT1_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"

# -------------------------------------------------------------
# | ABLATION2 | ABLATION2 | ABLATION2 | ABLATION2 | ABLATION2 |
# -------------------------------------------------------------

# Custom
# bash "$ABLAT2_SH" "${SCRIPT_DIR}/../data/raw/synth/" "LOCAL_DELAY.csv" "custom"

# ETTh1 (hourly)
# bash "$ABLAT2_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1" # ~3h12m  № 2 ------------------------ DONE

# ETTh2 (hourly)
# bash "$ABLAT2_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"

# -------------------------------------------------------------
# | ABLATION3 | ABLATION3 | ABLATION3 | ABLATION3 | ABLATION3 |
# -------------------------------------------------------------

# Custom
# bash "$ABLAT3_SH" "${SCRIPT_DIR}/../data/raw/synth/" "MULTISCALE.csv" "custom"  # ~4h55m № 3 ------------------------ DONE

# ETTh1 (hourly)
# bash "$ABLAT3_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"  # ~4h55m № 4 ------------------------ DONE

# ETTh2 (hourly)
# bash "$ABLAT3_SH" "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"

# -------------------------------------------------------------------------------------------------------------------------
# | ETT-small | ETT-small | ETT-small | ETT-small | ETT-small | ETT-small | ETT-small | ETT-small | ETT-small | ETT-small |
# -------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------
# | INFORMER | INFORMER | INFORMER | INFORMER | INFORMER |
# --------------------------------------------------------

# ETTh1 (hourly)
run_informer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"  # ------------------------ DONE 1 

# ETTh2 (hourly)
# run_informer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2" # ~1h43m № 6 ------------------------ DONE 1 

# ------------------------------------------------------------------
# | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER |
# ------------------------------------------------------------------

# ETTh1 (hourly)
run_autoformer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"  # ------------------------ DONE 1 

# ETTh2 (hourly)
# run_autoformer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"  # ~1h43m № 7 ------------------------ DONE 1

# ------------------------------------------------------------------
# | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER |
# ------------------------------------------------------------------

# ETTh1 (hourly)
run_convformer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"  # ------------------------ DONE 1 

# ETTh2 (hourly)
# run_convformer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"  # ~1h43m № 8 ------------------------ DONE 1

# -------------------------------------------------------------
# | PERFORMER | PERFORMER | PERFORMER | PERFORMER | PERFORMER |
# -------------------------------------------------------------

# ETTh1 (hourly)
run_performer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh1.csv" "ETTh1"  # ------------------------ DONE 1 

# ETTh2 (hourly)
# run_performer "${SCRIPT_DIR}/../data/raw/ETT-small/" "ETTh2.csv" "ETTh2"  # ~1h43m № 9 ------------------------ DONE 1

# -------------------------------------------------------------------------------------------------------------------------------
# | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL | ECL |
# -------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------
# | INFORMER | INFORMER | INFORMER | INFORMER | INFORMER |
# --------------------------------------------------------

# run_informer "${SCRIPT_DIR}/../data/raw/electricity/" "electricity.csv" "custom"  # ------------------------ DONE 1 except for 720

# ------------------------------------------------------------------
# | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER |
# ------------------------------------------------------------------

# run_autoformer "${SCRIPT_DIR}/../data/raw/electricity/" "electricity.csv" "custom"  # ------------------------ DONE 1

# ------------------------------------------------------------------
# | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER |
# ------------------------------------------------------------------

# run_convformer "${SCRIPT_DIR}/../data/raw/electricity/" "electricity.csv" "custom"  # ------------------------ DONE 1 except for 720

# -------------------------------------------------------------
# | PERFORMER | PERFORMER | PERFORMER | PERFORMER | PERFORMER |
# -------------------------------------------------------------

# run_performer "${SCRIPT_DIR}/../data/raw/electricity/" "electricity.csv" "custom"  # ------------------------ DONE 1 except for 720

# ---------------------------------------------------------------------------------------------------------------------------------
# | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate | exchange_rate |
# ---------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------
# | INFORMER | INFORMER | INFORMER | INFORMER | INFORMER |
# --------------------------------------------------------

# run_informer "${SCRIPT_DIR}/../data/raw/exchange_rate/" "exchange_rate.csv" "custom"  # ------------------------ DONE 1

# ------------------------------------------------------------------
# | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER |
# ------------------------------------------------------------------

# run_autoformer "${SCRIPT_DIR}/../data/raw/exchange_rate/" "exchange_rate.csv" "custom"  # ------------------------ DONE 1

# ------------------------------------------------------------------
# | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER |
# ------------------------------------------------------------------

# run_convformer "${SCRIPT_DIR}/../data/raw/exchange_rate/" "exchange_rate.csv" "custom"  # ------------------------ DONE 1

# -------------------------------------------------------------
# | PERFORMER | PERFORMER | PERFORMER | PERFORMER | PERFORMER |
# -------------------------------------------------------------

# run_performer "${SCRIPT_DIR}/../data/raw/exchange_rate/" "exchange_rate.csv" "custom"  # ------------------------ DONE 1

# -----------------------------------------------------------------------------------------------------------------------------------
# | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness | illness |
# -----------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------
# | INFORMER | INFORMER | INFORMER | INFORMER | INFORMER |
# --------------------------------------------------------

# run_informer "${SCRIPT_DIR}/../data/raw/illness/" "national_illness.csv" "custom"  # ------------------------ DONE 1

# ------------------------------------------------------------------
# | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER |
# ------------------------------------------------------------------

# run_autoformer "${SCRIPT_DIR}/../data/raw/illness/" "national_illness.csv" "custom"  # ------------------------ DONE 1

# ------------------------------------------------------------------
# | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER |
# ------------------------------------------------------------------

# run_convformer "${SCRIPT_DIR}/../data/raw/illness/" "national_illness.csv" "custom"  # ------------------------ DONE 1

# -------------------------------------------------------------
# | PERFORMER | PERFORMER | PERFORMER | PERFORMER | PERFORMER |
# -------------------------------------------------------------

# run_performer "${SCRIPT_DIR}/../data/raw/illness/" "national_illness.csv" "custom"  # ------------------------ DONE 1

# -----------------------------------------------------------------------------------------------------------------------------------
# | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic | traffic |
# -----------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------
# | INFORMER | INFORMER | INFORMER | INFORMER | INFORMER |
# --------------------------------------------------------

# run_informer "${SCRIPT_DIR}/../data/raw/traffic/" "traffic.csv" "custom"  # ------------------------ DONE 1 except for 720

# ------------------------------------------------------------------
# | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER |
# ------------------------------------------------------------------

# run_autoformer "${SCRIPT_DIR}/../data/raw/traffic/" "traffic.csv" "custom"  # ------------------------ DONE 1 except for 168, 336, 720

# ------------------------------------------------------------------
# | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER |
# ------------------------------------------------------------------

# run_convformer "${SCRIPT_DIR}/../data/raw/traffic/" "traffic.csv" "custom"  # ------------------------ DONE 1 except for 720

# -------------------------------------------------------------
# | PERFORMER | PERFORMER | PERFORMER | PERFORMER | PERFORMER |
# -------------------------------------------------------------

# run_performer "${SCRIPT_DIR}/../data/raw/traffic/" "traffic.csv" "custom"  # ------------------------ DONE 1 except for 720

# -----------------------------------------------------------------------------------------------------------------------------------
# | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather | weather |
# -----------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------
# | INFORMER | INFORMER | INFORMER | INFORMER | INFORMER |
# --------------------------------------------------------

# run_informer "${SCRIPT_DIR}/../data/raw/weather/" "weather.csv" "custom"  # ------------------------ DONE 1

# ------------------------------------------------------------------
# | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER | AUTOFORMER |
# ------------------------------------------------------------------

# run_autoformer "${SCRIPT_DIR}/../data/raw/weather/" "weather.csv" "custom"  # ------------------------ DONE 1

# ------------------------------------------------------------------
# | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER | CONVFORMER |
# ------------------------------------------------------------------

# run_convformer "${SCRIPT_DIR}/../data/raw/weather/" "weather.csv" "custom"  # ------------------------ DONE 1

# -------------------------------------------------------------
# | PERFORMER | PERFORMER | PERFORMER | PERFORMER | PERFORMER |
# -------------------------------------------------------------

# run_performer "${SCRIPT_DIR}/../data/raw/weather/" "weather.csv" "custom"  # ------------------------ DONE 1
