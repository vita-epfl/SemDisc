#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 50G
#SBATCH --time 72:00:00
#SBATCH --account vita
#SBATCH --gres gpu:1
#SBATCH --output train-run2

cd ../
module load gcc python py-torchvision py-torch
source ../ven*/bin/activate

# You can change G to different options --netG spade, --netG asapnets, --netG pix2pixhd

python train.py --name spade_semdisc --dataset_mode cityscapes --netG spade --c2f_sem_rec --normalize_smaps \
--checkpoints_dir /scratch/izar/saeedsa/spade/checkpoints --dataroot /scratch/izar/saeedsa/pix2pixHD/datasets/cityscape \
--lambda_seg 1 --lambda_rec 1 --lambda_GAN 35 --lambda_feat 10 --lambda_vgg 10 --fine_grained_scale 0.05 \
--niter_decay 0 --niter 100 \
--aspect_ratio 1 --load_size 256 --crop_size 256 --batchSize 16 --gpu_ids 0

python train.py --name spade_semdisc --dataset_mode cityscapes --netG spade --c2f_sem_rec --normalize_smaps \
--checkpoints_dir /scratch/izar/saeedsa/spade/checkpoints --dataroot /scratch/izar/saeedsa/pix2pixHD/datasets/cityscape \
--lambda_seg 1 --lambda_rec 1 --lambda_GAN 35 --lambda_feat 10 --lambda_vgg 10 --fine_grained_scale 0.05 \
--niter_decay 100 --niter 100 --continue_train --active_GSeg \
--aspect_ratio 1 --load_size 256 --crop_size 256 --batchSize 16 --gpu_ids 0
