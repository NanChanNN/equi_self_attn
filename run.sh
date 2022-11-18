#!/bin/bash

# make the script stop when error (non-true exit code) is occured
set -e

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<

conda activate et

for s in 4042 6594 1619 1940 9742 4165 6729 8186 7213 6440
do
    python nbody_run.py --batch_size 128 --num_channels 4 --num_layers 4 --data_str 5_new --seed $s
done
