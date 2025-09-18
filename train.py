import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Augmentor
import pandas as pd
import numpy as np
import scipy
import random
import cv2

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

SEED = 65536
random.seed(SEED)

TRAINING_IMAGES = "galaxy_data/images_training_rev1/"
os.makedirs("galaxy_data/processed", exist_ok=True)
DIR_PROCESSED = 'galaxy_data/processed/'

PREPROCESSED_IMAGE_DIM = 64

def root_mean_square_error(golden, prediction):
    # Element-wise RMSE score. 
    return np.sqrt( np.mean( np.square( np.array(golden).flatten() - np.array(prediction).flatten() )))

def conv_output_width(width, kernel, padding, stride):
    return int((width + 2 * padding - kernel - 1) / stride + 1)

def zipdir(directory, zip_file):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Preserve directory structure in the zip
            arcname = os.path.relpath(file_path, os.path.dirname(directory))
            zip_file.write(file_path, arcname)

def weighted_mean_square_error(output, targets, weights):
    loss = (output - targets) ** 2
    loss = loss * weights.expand(loss.shape)
    loss = loss.mean(dim=0).sum()
    return loss

images = [f for f in os.listdir(TRAINING_IMAGES) if os.path.isfile(os.path.join(TRAINING_IMAGES, f))]
labels = pd.read_csv('galaxy_data/training_solutions_rev1.csv')
labels.GalaxyID = labels.GalaxyID.apply(lambda id: str(int(id)) + '.png')

# Smooth
# Featured or disc
# Star or artifact
# Edge on
# Not edge on
# Bar through center
# No bar
# Spiral
# No Spiral
# No bulge
# Just noticeable bulge
# Obvious bulge
# Dominant bulge
# Odd Feature
# No Odd Feature
# Completely round
# In between
# Cigar shaped
# Ring (Oddity)
# Lens or arc (Oddity)
# Disturbed (Oddity)
# Irregular (Oddity)
# Other (Oddity)
# Merger (Oddity)
# Dust lane (Oddity)
# Rounded bulge
# Boxy bulge
# No bulge
# Tightly wound arms
# Medium wound arms
# Loose wound arms
# 1 Spiral Arm
# 2 Spiral Arms
# 3 Spiral Arms
# 4 Spiral Arms
# More than four Spiral Arms
# Can't tell

# pick 80% of objects as the training set
x_train, x_test, y_train, y_test = train_test_split(labels.GalaxyID,labels[labels.columns[1:]].to_numpy(),test_size = 0.20,random_state = SEED)
#further split the test set into validation and test sets
x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,test_size = 0.50,random_state = SEED)

print(f'Total number of images: {len(images)}')
print(f'Number of classes for classification: {labels.shape[1]-1}')
print(f'Size of training set: {x_train.shape[0]}')
print(f'ize of validation set: {x_val.shape[0]}')
print(f'Size of test set: {x_test.shape[0]}')


def batch_generator(pics, labels, batch_size, rotate=False):
    '''
    Given a list of object IDs, generate batches of PyTorch tensors
    '''
    angles = np.array([0,90,180,270])
    labels = torch.tensor(labels, dtype=torch.float32)
    l = len(pics)
    batches = int(l/batch_size)
    leftover = l % batch_size
    for batch in range(batches):
        start = batch * batch_size
        this_batch = pics[start:start+batch_size]
        batch_labels = labels[start:start+batch_size,:]
        
        if rotate:
            yield torch.tensor([scipy.ndimage.rotate(
                                plt.imread(DIR_PROCESSED + pic, format='png'),
                                reshape=False,
                                angle=np.random.randint(0,360)
                                )
                            for pic in this_batch], dtype=torch.float32).permute(0, 3, 1, 2), batch_labels, this_batch
        else:
            yield torch.tensor([ plt.imread(DIR_PROCESSED + pic, format='png') for pic in this_batch], dtype=torch.float32).permute(0, 3, 1, 2), batch_labels, this_batch
    start = batches * batch_size
    this_batch = pics[start:start+leftover]
    batch_labels = labels[start:start+leftover,:]
    if rotate:
        yield torch.tensor([scipy.ndimage.rotate(
                            plt.imread(DIR_PROCESSED + pic, format='png'),
                            reshape=False,
                            angle=np.random.randint(0,360)
                            )
                        for pic in this_batch], dtype=torch.float32).permute(0, 3, 1, 2), batch_labels, this_batch
    
    yield torch.tensor([ plt.imread(DIR_PROCESSED + pic, format='png') for pic in this_batch], dtype=torch.float32).permute(0, 3, 1, 2), batch_labels, this_batch
    


classes = labels.columns[1:]
print(f"possible classes {classes}")

in_width = PREPROCESSED_IMAGE_DIM


c1_in = 3 # RGB
c1_kernel = 9
c1_out = 64
pool_kernel = 2

c1_conv_width = conv_output_width(in_width, c1_kernel, c1_kernel/2, 1)
c1_pooled_width = conv_output_width(c1_conv_width, pool_kernel/2, 0, pool_kernel)
print("C1: ", c1_conv_width)
print('P1: ', c1_pooled_width)

c2_kernel = 5
c2_out = 128
c2_conv_width = conv_output_width(c1_pooled_width, c2_kernel, c2_kernel/2, 1)
c2_pooled_width = conv_output_width(c2_conv_width, pool_kernel, pool_kernel/2, pool_kernel)
print("C2: ", c2_conv_width)
print('P2: ', c2_pooled_width)

c3_kernel = 3
c3_out = 256
c3_conv_width = conv_output_width(c2_pooled_width, c3_kernel, c3_kernel/2, 1)
c3_pooled_width = conv_output_width(c3_conv_width, pool_kernel, pool_kernel/2, pool_kernel)
print("C3: ", c3_conv_width)
print('P3: ', c3_pooled_width)

c4_kernel = 1
c4_out = 512
c4_conv_width = conv_output_width(c3_pooled_width, c4_kernel, c4_kernel/2, 1)
c4_pooled_width = conv_output_width(c4_conv_width, pool_kernel, pool_kernel/2, pool_kernel)
print("C4: ", c4_conv_width)
print('P4: ', c4_pooled_width)

full_1_in = c4_out * c4_pooled_width * c4_pooled_width
full_1_out = 1024
full_2_out = 256
full_3_out = len(classes)


class Net(nn.Module):
#
#X -> 4X(Conv -> BN -> Relu -> MaxPool) -> X
#X -> 3X(Linear -> Dropout -> Relu) -> SoftMax -> Out
#
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=c1_in, 
                               out_channels=c1_out, 
                               kernel_size=c1_kernel,
                               stride=1,
                               padding= int(c1_kernel/2),
                               padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=c1_out,
                               out_channels=c2_out, 
                               kernel_size=c2_kernel,
                               stride=1,
                               padding= int(c2_kernel/2),
                               padding_mode='zeros') 
        self.conv3 = nn.Conv2d(in_channels=c2_out, 
                               out_channels=c3_out, 
                               kernel_size=c3_kernel,
                               stride=1,
                               padding= int(c3_kernel/2),
                               padding_mode='zeros')
        self.conv4 = nn.Conv2d(in_channels=c3_out,
                               out_channels=c4_out,
                               kernel_size=c4_kernel,
                               stride=1,
                               padding= int(c4_kernel/2),
                               padding_mode='zeros')
        self.pool1 = nn.MaxPool2d(pool_kernel)
        self.pool2 = nn.MaxPool2d(pool_kernel)
        self.pool3 = nn.MaxPool2d(pool_kernel)
        self.pool4 = nn.MaxPool2d(pool_kernel)

        self.conv1_bn = nn.BatchNorm2d(c1_out)
        self.conv2_bn = nn.BatchNorm2d(c2_out)
        self.conv3_bn = nn.BatchNorm2d(c3_out)
        self.conv4_bn = nn.BatchNorm2d(c4_out)

        self.linear1 = nn.Linear(full_1_in, full_1_out)
        self.linear2 = nn.Linear(full_1_out, full_2_out)
        self.linear3 = nn.Linear(full_2_out, full_3_out)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        # Conv layers
        x = self.pool1(F.relu(self.conv1_bn(x)))
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool4(F.relu(self.conv4_bn(self.conv4(x))))
        # flatten before fuly connected linear layers
        x = x.reshape(x.size(0), -1)
        # Fully connected linear layers
        x = F.relu(self.dropout(self.linear1(x)))
        x = F.relu(self.dropout(self.linear2(x)))
        x = F.sigmoid(self.linear3(x))
        return x

net = Net()

loss_weight = torch.tensor(np.sum(y_train, axis=0) / np.sum(y_train))
optimizer = optim.Adam(net.parameters())


num_epochs = 10
batch_size = 2048 
mini_batch_size = 128
loss_tracking = []
print(f"Start - num_epochs = {num_epochs}  batch_size = {batch_size} min_batch = {mini_batch_size} total_training_steps = {batch_size/mini_batch_size}")

for epoch in range(num_epochs): 
    batch_id = 1
    train_data = batch_generator(x_train,y_train,batch_size=batch_size,rotate=True)
    
    for images_pre, targets_pre, _ in train_data:
        # jumble the images in the batch
        jumbled_order = torch.randperm(len(images_pre))
        images = images_pre[jumbled_order]
        targets = targets_pre[jumbled_order]

        for mini_batch in range(int(batch_size/mini_batch_size)):
            mini_batch_start_id = mini_batch*mini_batch_size
            mini_batch_end_id = mini_batch_start_id + mini_batch_size
            if mini_batch_start_id >= images.shape[0]:
                break

            ###traning loop###
            #1. Forward
            fwd = net(images[mini_batch_start_id:mini_batch_end_id])
            #2. loss estimation
            loss = weighted_mean_square_error(fwd, targets[mini_batch_start_id:mini_batch_end_id], loss_weight)
            #3. Backward
            optimizer.zero_grad()   
            loss.backward()
            #4. weight update
            optimizer.step()
        
        print("epoch %d, batch %d - loss: %.4f" % (epoch+1, batch_id, loss.item()))
        batch_id += 1
        loss_tracking.append((epoch, batch_id, loss.item()))



plt.title('Training Loss over batches')
plt.xlabel('Batch')
plt.ylabel('weighted mean square error Loss')
plt.plot([error[2] for error in loss_tracking])
plt.savefig('training_loss.png')

val_data = batch_generator(x_val,y_val,batch_size=1000)
predicted_labels = np.empty((0,len(classes)), float)
for val_images, val_labels , _ in val_data:
    outputs = net(val_images)
    predicted_labels = np.append(predicted_labels, outputs.detach().numpy(), axis=0)
prediction_error = root_mean_square_error(y_val, predicted_labels)
print("Validation sample RMSE error = ",prediction_error)

test_data = batch_generator(x_test,y_test,batch_size=1000)
predicted_labels = np.empty((0,len(classes)), float)
for test_images, test_labels , _ in test_data:
    outputs = net(test_images)
    predicted_labels = np.append(predicted_labels, outputs.detach().numpy(), axis=0)
prediction_error = root_mean_square_error(y_test, predicted_labels)
print("Test sample RMSE error = ",prediction_error)



