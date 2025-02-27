# PseTNet

This repo is an implementation of the paper: Learning Task Level Pseudo-Text Prompt for Improved Medical Image Segmentation.

The detailed structure of the PseTNet model is shown in the following figure:

<p align="center">
  <img src="Figures/main figure.jpg" width="700"/>
</p>

## Dataset

|   Dataset   |                        Download                         | Comment                                                                                                           |
|:-----------:| :-----------------------------------------------------: |-------------------------------------------------------------------------------------------------------------------|
| Spineweb-16 |    [Link](https://aasce19.grand-challenge.org/Home/)    | Train and val: random split, <br />Test: official split.                                                          |
|     DFU     | [Link](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge) | Train: official split, <br />Val and test: random split. <br />(The offical test set does not have image labels.) |
|    ACDC     |    [Link](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)    | Train, val and test: random split. <br />(Please divide the different images according to the patient ID.)        |

Please prepare the dataset in the following format to facilitate the use of the code:

```angular2html
├── datasets
   ├── Spineweb-16
   │   ├── image
   |   |   ├── train
   |   |   ├── val
   |   |   └── test
   │   └── mask
   |       ├── train
   |       ├── val
   |       └── test
   |  
   └── DFU
       ├── image
       ......
```

## Parameter preparation

Please [download](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) the pre-training parameters for the text encoder of the CLIP model to the /model/PseTNet/text_part/CLIP folder.

## Train model

```bash
python train.py
```
If you want to train the PseTNet model on different datasets, please modify the parameters of DATASET in the train_config.yaml file.

## Requirements

+ CUDA/CUDNN
+ pytorch>=1.7.1
+ torchvision>=0.8.2

## Citation
```
@inproceedings{he2024learning,
  title={Learning Task-Level Pseudo-Text Prompt for Improved Medical Image Segmentation},
  author={He, Zhu and Liu, Yaru and Yang, Guangjing and Bao, Xueqi and Chai, Yufei and Lao, Qicheng},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={3262--3267},
  year={2024},
  organization={IEEE}
}
```
