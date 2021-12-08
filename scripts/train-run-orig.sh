#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 50G
#SBATCH --time 72:00:00
#SBATCH --account vita
#SBATCH --gres gpu:1
#SBATCH --output train-run-orig

cd ../
module load gcc python py-torchvision py-torch
source ../ven*/bin/activate

# You can change G to different options --netG spade, --netG asapnets, --netG pix2pixhd

python train.py --name spade_orig --dataset_mode cityscapes --netG spade \
--checkpoints_dir /scratch/izar/saeedsa/spade/checkpoints --dataroot /scratch/izar/saeedsa/pix2pixHD/datasets/cityscape \
--niter_decay 100 --niter 100 \
--aspect_ratio 1 --load_size 256 --crop_size 256 --batchSize 16 --gpu_ids 0
