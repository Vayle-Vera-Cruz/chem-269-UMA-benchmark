#!/bin/bash
#SBATCH --job-name=Morse_Geodesic_Interpolation
#SBATCH -p debug
#SBATCH --output=Interpolation.out
#SBATCH --error=Interpolation.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # <-- 8 CPU cores for 1 Python process

source /opt/slurm_scripts/setup_mlp_geodesic.sh

# Let NumPy / BLAS / PyTorch use all 8 threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"

#This batch script is used to accelerate the creation of your multiframe .xyz file for use in the Geodesic Model

#The only things you need to change are the [integer] value to a number of your choice for the number of images you want,
#followed by you input file, which is a multiframe xyz file containing your starting and end geometry,
#Lastly you may name your output.xyz file       

python /opt/mlp_geodesic/morse_geodesic/cli.py \
        --nimages 9 \
        --tol 1e-5 \
        Multiframe.xyz \
        output.xyz
