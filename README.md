# A Shared Representation for Photorealistic Driving Simulators

The official code for the paper: "A Shared Representation for Photorealistic Driving Simulators", [ITS paper](https://ieeexplore.ieee.org/abstract/document/9635715), [arXiv](https://arxiv.org/abs/2108.10879)

> __A Shared Representation for Photorealistic Driving Simulators__<br />
> _[Saeed Saadatnejad](https://scholar.google.com/citations?user=PBdhgFYAAAAJ&hl=en), [Siyuan Li](https://scholar.google.ch/citations?user=80_DZiwAAAAJ&hl=en), [Taylor Mordan](https://dblp.org/pid/203/8404.html), [Alexandre Alahi](https://scholar.google.com/citations?user=UIhXQ64AAAAJ&hl=en)_, 2021.
> A powerful simulator highly decreases the need for real-world tests when training and evaluating autonomous vehicles.
> Data-driven simulators flourished with the recent advancement of conditional Generative Adversarial Networks (cGANs), providing high-fidelity images.
> The main challenge is synthesizing photo-realistic images while following given constraints.
> In this work, we propose to improve the quality of generated images by rethinking the discriminator architecture. 
> The focus is on the class of problems where images are generated given semantic inputs, such as scene segmentation maps or human body poses.
> We build on successful cGAN models to propose a new semantically-aware discriminator that better guides the generator.
> We aim to learn a shared latent representation that encodes enough information to jointly do semantic segmentation, content reconstruction, along with a coarse-to-fine grained adversarial reasoning.
> The achieved improvements are generic and simple enough to be applied to any architecture of conditional image synthesis. 
> We demonstrate the strength of our method on the scene, building, and human synthesis tasks across three different datasets.
> 

## Example

<p align="center">
  <a href="url"><img src="imgs/results.png"  height="512" width="1024" ></a>
</p>  

 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

1. Clone this repo.
```
git clone https://github.com/vita-epfl/SemDisc.git
```
```
cd ./SemDisc
```
### Prerequisites

2. Please install dependencies by

```
pip install -r requirements.txt
```

### Dataset Preparation

3. The cityscapes dataset can be downloaded from here: [cityscapes](https://www.cityscapes-dataset.com/dataset-overview/)

For the experiment, you will need to download  [gtFine_trainvaltest.zip] and [leftImg8bit_trainvaltest.zip] and unzip them.

gtFine contains the semantics segmentations. 
leftImg8bit contains the dashcam photographs


## Training the network

After preparing all necessary environments and the dataset. We can start to train the network by entering:

### Training the semantic-aware discriminator

```
python train.py --name cityscapes_semanticD --dataset_mode cityscapes --dataroot /PATH TO your data folder/ --load_size 256 --crop_size 256  --gpu_ids 0 --tf_log --batchSize 16 --niter 100 --niter_decay 100 --semantic --gan_mode ls
```
### Training the original multiscale discriminator

```
python train.py --name cityscapes --dataset_mode cityscapes --dataroot /PATH TO your data folder/ --load_size 256 --crop_size 256  --gpu_ids 0 --tf_log --batchSize 16 --niter 100 --niter_decay 100 
```
Then, please put the model under folder 'models'.

When the main training process finishes, you need to restart your training with segmentation enabled.
```
python train.py --name cityscapes --dataset_mode cityscapes --dataroot /PATH TO your data folder/ --load_size 256 --crop_size 256  --gpu_ids 0 --tf_log --batchSize 16 --niter 100 --niter_decay 100 --fine_tune --continue_train --semantic --gan_mode ls
```

### Running the tests

After you have full trained the network, you can run the test as following:

```
python test.py --name cityscapes_semanticD --dataset_mode cityscapes --dataroot /PATH TO your data folder/ --load_size 256 --crop_size 256  --gpu_ids 0
```
## The code structure

The base of the code is borrowed from SPADE. Please refer to [SPADE](https://github.com/NVlabs/SPADE) to see the details.

We adapt the 'pix2pix_model.py',  'cityscape_dataset.py' and 'loss.py' to add semantic-aware discriminator.



## Citation

```
@article{saadatnejad2021semdisc,
  author={Saadatnejad, Saeed and Li, Siyuan and Mordan, Taylor and Alahi, Alexandre},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={A Shared Representation for Photorealistic Driving Simulators}, 
  year={2021},
  doi={10.1109/TITS.2021.3131303}}
```
