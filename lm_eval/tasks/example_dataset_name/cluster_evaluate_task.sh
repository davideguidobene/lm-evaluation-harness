#!/bin/bash

#SBATCH --nodes=1 
#SBATCH --time=48:00:00
#SBATCH --ntasks=1 
#SBATCH --job-name=benchmark
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100GB
#SBATCH --output=/cluster/home/dguidobene/logs/lmh/lmh.out
#SBATCH --error=/cluster/home/dguidobene/logs/lmh/lmh.err
#SBATCH --gpus=rtx_4090:1
#SBATCH --tmp=500G

module load--ignore_cache eth_proxy

export HF_HOME=/cluster/scratch/dguidobene
source /cluster/scratch/dguidobene/venvs/lmharness/bin/activate
lm_eval \
  --model hf \
  --model_args pretrained=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --tasks example_task \
  --device gpu \
  --batch_size 1