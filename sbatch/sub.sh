#!/bin/bash
#SBATCH --job-name=ML
#SBATCH --mem-per-cpu=6000 
#SBATCH --time=1:00:00      
#SBATCH --ntasks=1            
#SBATCH --nodes=1 
#SBATCH --array=1-1000           
#SBATCH --output=out_py.txt
#SBATCH --open-mode=append


bin=../scripts/run.py 

python $bin $SLURM_ARRAY_TASK_ID 

exit 0
