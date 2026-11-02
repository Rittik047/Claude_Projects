#!/usr/bin/env bash
# Submit the whole development pipeline with SLURM dependencies (run from bmets-forecast/).
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$BMF_SCRATCH"
sub() { sbatch --parsable -p "$1" "${@:2}"; }
J0=$(sub "$JUNO_PARTITION_CPU" "$BMF_HOME/juno/01b_index.sbatch")
J1=$(sub "$JUNO_PARTITION_CPU" --dependency=afterok:$J0 "$BMF_HOME/juno/02_preprocess.sbatch")
J2=$(sub "$JUNO_PARTITION_CPU" --dependency=afterok:$J1 "$BMF_HOME/juno/03a_seg_export.sbatch")
J3=$(sub "$JUNO_PARTITION_GPU" --dependency=afterok:$J2 "$BMF_HOME/juno/03_seg_train.sbatch")
J4=$(sub "$JUNO_PARTITION_CPU" --dependency=afterok:$J3 "$BMF_HOME/juno/03b_seg_eval.sbatch")
J5=$(sub "$JUNO_PARTITION_GPU" --dependency=afterok:$J1 "$BMF_HOME/juno/04_embed.sbatch")
J6=$(sub "$JUNO_PARTITION_GPU" --dependency=afterok:$J5 "$BMF_HOME/juno/05_train_eval.sbatch")
J7=$(sub "$JUNO_PARTITION_CPU" --dependency=afterok:$J6:$J4 "$BMF_HOME/juno/06_figures_report.sbatch")
echo "submitted: preprocess=$J1 segexp=$J2 nnunet=$J3 segeval=$J4 embed=$J5 train=$J6 report=$J7"
echo "External validation (07_external.sbatch) is submitted manually after the analysis is frozen."
