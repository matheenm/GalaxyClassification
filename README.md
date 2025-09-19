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

![Example](https://github.com/matheenm/GalaxyClassification/blob/main/Images/Galaxy_Example.png)

## Convolution Neural Network
 Input<br />
  │<br />
  ▼<br />
  ┌───────────────────────────────────────────────────┐<br />
   Conv2d (cin= 3, cout= 64, k= 9, str= 1, pad= 1)   <br />
   → BN → ReLU → MaxPool2d(k= 2, str= 2)             <br />
  └───────────────────────────────────────────────────┘<br />
  │<br />
  ▼<br />
  ┌───────────────────────────────────────────────────┐<br />
   Conv2d (cin= 64, cout= 128, k= 5, str= 1, pad= 1) <br />
   → BN → ReLU → MaxPool2d(k= 2, str= 2)             <br />
  └───────────────────────────────────────────────────┘<br />
  │<br />
  ▼<br />
  ┌───────────────────────────────────────────────────┐<br />
   Conv2d (cin= 128, cout= 256, k= 3, str= 1, pad= 1)<br />
   → BN → ReLU → MaxPool2d (k= 2, str= 2)            <br />
  └───────────────────────────────────────────────────┘<br />
  │<br />
  ▼<br />
  ┌───────────────────────────────────────────────────┐<br />
   Conv2d (cin=256, cout=512, k= 1, str= 1, padd= 1) <br />
   → BN → ReLU → MaxPool2d(k= 2, str= 2)             <br />
  └───────────────────────────────────────────────────┘<br />
  │<br />
  ▼<br />
  Flatten (to 1‑D vector)<br />
  │<br />
  ▼<br />
  ┌────────────────────────────────┐<br />
   Linear (fin= 8192, fout= 1024) <br />
   → Dropout → ReLU 								      <br />
  └────────────────────────────────┘<br />
  │<br />
  ▼<br />
  ┌────────────────────────────────────┐<br />
   Linear (fin= 1024, fout= 2048)     <br />
   → Dropout → ReLU 			          		   <br />
  └────────────────────────────────────┘<br />
  │<br />
  ▼<br />
  ┌───────────────────────────────────────┐<br />
   Linear (fin= 128, fout=NUM_CLASSES)   <br />
   → Sigmoid      								               <br />
  └───────────────────────────────────────┘<br />
  │<br />
  ▼<br />
  Output<br />



The model is small enough that it can be run on a CPU itself. 

## Code
The code is broken down into two parts. The preprocessing which crops the images into a region of interest and generates 64x64 images (**preprocess.py**) 
The second part trains the neural network (**train.py**).

There is also a helper program **plot_image.py** which can plot out any galaxy image given an ID.
## Loss
epoch 10, batch 26 - loss: 0.0179
![Loss](https://github.com/matheenm/GalaxyClassification/blob/main/Images/training_loss.png)

## Example classifications

![test1](https://github.com/matheenm/GalaxyClassification/blob/main/Images/test_1.png)
![test2](https://github.com/matheenm/GalaxyClassification/blob/main/Images/test_2.png)

## Reference
Willett K.W., Lintott C.J., Bamford S.P., Masters K.L., Simmons B.D., Casteels K.R.V., Edmondson E.M., et al., 2013, MNRAS, 435, 2835. [doi:10.1093/mnras/stt1458](https://doi.org/10.1093/mnras/stt1458)
