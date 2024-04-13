#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=72G
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-80gb:1

module load ml-gpu
cd /work/LAS/cjquinn-lab/zefuh/selectivity/interpretability-of-source-code-transformers/POS\ Code/Experiments
ml-gpu /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python run_neurox1.py --extract=False --this_model pretrained_RoBERTa --language java > ./running_scripts_java/RoBERTa.log
 