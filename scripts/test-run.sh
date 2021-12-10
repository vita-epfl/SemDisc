#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 30G
#SBATCH --time 5:00:00
#SBATCH --account vita
#SBATCH --gres gpu:1
#SBATCH --output test-run

cd ../
module load gcc python py-torchvision py-torch
source ../ven*/bin/activate

# You can change G to different options --netG spade, --netG asapnets, --netG pix2pixhd

name='spade' # or spade_orig
python test.py --name $name --dataset_mode cityscapes \
--checkpoints_dir /scratch/izar/saeedsa/spade/checkpoints \
--dataroot /scratch/izar/saeedsa/pix2pixHD/datasets/cityscape \
--results_dir /scratch/izar/saeedsa/spade/results/ \
--which_epoch latest --aspect_ratio 1 --load_size 256 --crop_size 256 \
--netG spade --how_many 496

# FID calculation
python fid/pytorch-fid/fid_score.py \
/scratch/izar/saeedsa/spade/results/$name/test_latest/images/GT_image \
/scratch/izar/saeedsa/spade/results/$name/test_latest/images/synthesized_image >> results/fid_$name.txt
