#!/bin/bash
#SBATCH -A cs175_class_gpu    ## Account to charge
#SBATCH --time=240:00:00              ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=gpu       ## Partition name
#SBATCH --mem=30GB            ## Allocated Memory
#SBATCH --cpus-per-task 16    ## Number of CPU cores
#SBATCH --gres=gpu:V100:1     ## Type and the number of GPUs

Xvfb :43 -screen 0 1024x768x24 & # Create virtual display so the remote machine can run the python script
export DISPLAY=:43              # Redirect Unity to the virtual screen created

python train_obt.py --worker 43
