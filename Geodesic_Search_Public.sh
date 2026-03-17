#!/bin/bash
#SBATCH --job-name=UMA_Geodesic_Test
#SBATCH -p debug
#SBATCH --output=UMA_GeodesicTest.out
#SBATCH --error=UMA_Geodesic_Test.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # <-- 8 CPU cores for 1 Python process

source /opt/slurm_scripts/setup_mlp_geodesic.sh

# Let NumPy / BLAS / PyTorch use all 8 threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"

#The following line calls a python script from the /opt/mlp_geodesic path and the UMA potential file. Edit the line after the .pt file as follows:
#After the .pt file you input your multiframe .xyz file followed by your name for your output .xyz file       
python /opt/mlp_geodesic/cli.py --climb --device cpu --model-path /opt/mlp_geodesic/uma-s-1.pt morse_geodesic_path_HCN_to_HNC.xyz Test_Finished.xyz &> Test.out
