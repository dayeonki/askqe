#!/bin/sh

#SBATCH --job-name=run_bt
#SBATCH --output=/fs/nexus-scratch/dayeonki/QATransEval/log/run_bt.out
#SBATCH --error=/fs/nexus-scratch/dayeonki/QATransEval/log/run_bt.error
#SBATCH --time=04:00:00
#SBATCH --mem=64gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:2


python -u run_bt.py