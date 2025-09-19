# Convolution Neural Network to classify galaxies based on their shape

## The Galaxy Zoo Project
 Decision tree of classifications (Willett et al. 2013)
![Decision tree](https://github.com/matheenm/GalaxyClassification/blob/main/Images/Decision_tree.png)

## Dataset
I used the dataset from  Galaxy Zoo - The Galaxy Challenge on [kaggle.com](www.kaggle.com):

https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data
*kaggle competitions download galaxy-zoo-the-galaxy-challenge*

The code uses the **images_training_rev1.zip** which has the images and **training_solutions_rev1.zip** which has the labelled soultions.

## Labelled Examples

## Convolution Neural Network

The model is small enough that it can be run on a CPU itself. 

## Code
The code is broken down into two parts. The preprocessing which crops the images into a region of interest and generates 64x64 images (**preprocess.py**) 
The second part trains the neural network (**train.py**).

## Loss
epoch 10, batch 26 - loss: 0.0179
![Loss](https://github.com/matheenm/GalaxyClassification/blob/main/Images/training_loss.png)

Validation sample RMSE error =  0.1074
Test sample RMSE error =  0.1069

## Example classifications

## Reference
Willett K.W., Lintott C.J., Bamford S.P., Masters K.L., Simmons B.D., Casteels K.R.V., Edmondson E.M., et al., 2013, MNRAS, 435, 2835. [doi:10.1093/mnras/stt1458](https://doi.org/10.1093/mnras/stt1458)

## Acknowledgement
