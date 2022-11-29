#!/bin/zsh

for task in 'beta_gamma' 'beta_sharpness' 'phase' 'pac' 'cross_pac' 'phase_shift' 'burst_length'
do
    python main_synthetic.py --name $task --n_jobs 8 &&
done
