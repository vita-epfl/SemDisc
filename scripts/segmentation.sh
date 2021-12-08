#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 30G
#SBATCH --time 5:00:00
#SBATCH --account vita
#SBATCH --gres gpu:1
#SBATCH --output segmentation-run

cd ../
module load gcc python py-torchvision py-torch
source ../ven*/bin/activate

name='tmp'
cd datasets/cityscapes/
convert -sample 512X256\! "/scratch/izar/saeedsa/spade/results/"$name"/test_latest/images/synthesized_image/*.png" -set filename:base "%[base]" "synthesized_image/%[filename:base].png"
find synthesized_image/ -maxdepth 3 -name "*_leftImg8bit.png" | sort > val_images.txt
cd ../..
python3 segment.py test -d datasets/cityscapes/ -c 19 --arch drn_d_105 --pretrained drn-d-105_ms_cityscapes.pth --phase val --batch-size 1 --ms >> ../results/seg_$name.txt
