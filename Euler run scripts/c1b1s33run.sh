#!/usr/bin/env bash

#SBATCH --partition=slurm_shortgpu

#SBATCH --job-name=SEc1b1s33
#SBATCH --output=c1b1s33job_output-%j.txt

#SBATCH --time=0-00:30:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:7

date
hostname
rm -rf model logs
python3 c1b1s33.py
result=$?

if [ $result -ge 0 ]; then
    echo “job $SLURM_JOB_ID completed wrongly”
    exit $result
fi