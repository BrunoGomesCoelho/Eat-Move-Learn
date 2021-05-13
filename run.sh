#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=geese
#SBATCH --mail-type=END
#SBATCH --mail-user=zy2043@nyu.edu
#SBATCH --output=geese%j.out

conda activate geese

cd /scratch/zy2043/Eat-Move-Learn/notebooks/

python geese_dqn_1opponent.py 
#python geese_dqn_2opponents.py 
#python geese_dqn_3opponents.py 
#python geese_dqn_multiopponents.py 


