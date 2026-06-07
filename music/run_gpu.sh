#!/bin/bash
#SBATCH --account=def-baptista
#SBATCH --job-name=DiscreteDiffusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 # CPU cores/threads
#SBATCH --gpus-per-node=1
#SBATCH --time=9:59:59
#SBATCH --output=%x_%j.txt
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=r.baptista@utoronto.ca

module load cuda/12.9 # 12.9 or 12.6
module load python/3.12   

source /home/baptista/ML/bin/activate # activate virtual environment

echo "==============================================="
echo " Job started on $(hostname) at $(date)"
echo " SLURM Job ID: $SLURM_JOB_ID"
echo " CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo " GPUs assigned by Slurm: $SLURM_JOB_GPUS"
echo "==============================================="
echo "Detected GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# ------------------------------
# Run the script
# ------------------------------
echo "Running script on GPU..."
start_time=$(date +%s)

srun python /scratch/baptista/GenerativeDiscreteDiffusionBPS/music/train.py 
#srun python /scratch/baptista/GenerativeDiscreteDiffusionBPS/music/post_process.py --model_run ./runs/run_20260505_230717

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "Training completed in ${runtime} seconds."

