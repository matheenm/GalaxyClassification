import os
import zipfile
import shutil
import pandas as pd
import random
import cv2 
import matplotlib.pyplot as plt


def zipdir(directory, zip_file):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Preserve directory structure in the zip
            zip_name = os.path.relpath(file_path, os.path.dirname(directory))
            zip_file.write(file_path, zip_name)

SEED = 65536
random.seed(SEED)
images_zip =  'images_training_rev1.zip'
solutions_zip = 'training_solutions_rev1.zip'

try:
    os.makedirs("galaxy_data", exist_ok=True)
except Exception as e:
    print(f"Error: {e}")

shutil.copy(images_zip, "galaxy_data")
shutil.copy(solutions_zip, "galaxy_data")

TRAINING_IMAGES = "galaxy_data/images_training_rev1/"
os.makedirs("galaxy_data/processed", exist_ok=True)
DIR_PROCESSED = 'galaxy_data/processed/'


zip_extraction_path = 'galaxy_data/' 
# Unpack training and validation datasets
zip_path = "galaxy_data/images_training_rev1.zip"
# Extract all contents to the current directory
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(path=zip_extraction_path)
    
# Unpack training and validation datasets
zip_path = "galaxy_data/training_solutions_rev1.zip"
# Extract all contents to the current directory
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(path=zip_extraction_path)
    
images = [f for f in os.listdir(TRAINING_IMAGES) if os.path.isfile(os.path.join(TRAINING_IMAGES, f))]
labels = pd.read_csv('galaxy_data/training_solutions_rev1.csv')
labels.GalaxyID = labels.GalaxyID.apply(lambda id: str(int(id)) + '.jpg')

galaxy_classes = ['Smooth','Featured or disc','Star or artifact','Edge on','Not edge on','Bar through center','No bar','Spiral','No Spiral','No bulge','Just noticeable bulge','Obvious bulge','Dominant bulge','Odd Feature','No Odd Feature','Completely round','In between','Cigar shaped','Ring (Oddity)','Lens or arc (Oddity)','Disturbed (Oddity)','Irregular (Oddity)','Other (Oddity)','Merger (Oddity)','Dust lane (Oddity)','Rounded bulge','Boxy bulge','No bulge','Tightly wound arms','Medium wound arms','Loose wound arms','1 Spiral Arm','2 Spiral Arms','3 Spiral Arms','4 Spiral Arms','More than four Spiral Arms',"Can't tell"]
print(f'Total number of images: {len(images)}')
print(f'Number of classes for classification: {labels.shape[1]-1}')

     
size = 15
fig = plt.figure('Image Examples', figsize=[size,size*10/4])
#plt.suptitle("Top Image for Each Feature", fontsize=20)
plt.axis('off')
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
for i, feature in enumerate(labels.columns[1:]):
    fig.add_subplot(10, 4, i+1)
    plt.title(feature + "\n" + galaxy_classes[i])
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    file = labels.iloc[labels[feature].idxmax()].GalaxyID
    img = plt.imread(TRAINING_IMAGES + file, format='jpg')
    plt.imshow(img, aspect='auto')
    #plt.xlabel(str(i))
fig.tight_layout(pad=1.0)
plt.savefig('Image_Examples.png')  


padding_size = 0.2

for image in labels.GalaxyID:

    im = cv2.imread(TRAINING_IMAGES + image)
    im2 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) 
    # I read the docs and found this uses human perception of luminance already. 
    ret, thresh = cv2.threshold(im2, 10, 255, 0) 
    # The numbers are upper and lower thresholds
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ROI = (0,0,0,0) # Region of interest
    ROI_area = 0
    for contour in contours: # cv.RETR_LIST exports contrours as a list. 
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height 
        if area > ROI_area:
            ROI_area = area 
            ROI = (x,y,width,height) 
    
    x, y, width, height = ROI

    if width > height:
        pad = int(width * padding_size)
    else:
        pad = int(height * padding_size)
    
    if (y-pad >= 0 and 
        x-pad >= 0 and 
        y + max(width, height) + pad < im.shape[1] and 
        x + max(width, height) + pad < im.shape[0]):

        crop = im[y-pad:y+max(width,height)+pad,x-pad:x+max(width,height)+pad]
    else:
        crop = im
    
    image = image.replace('jpg','png')


    # 64x64 
    cv2.imwrite(
        DIR_PROCESSED + image, # OpenCV adheres to file extension formats 
        cv2.resize(crop, (64,64), interpolation=cv2.INTER_AREA)
    )


size = 15
fig = plt.figure('Image Examples color', figsize=[size,size*10/4])
plt.axis('off')
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
for i, feature in enumerate(labels.columns[1:]):
    fig.add_subplot(10, 4, i+1)
    plt.title(feature + "\n" + galaxy_classes[i])
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    file = labels.iloc[labels[feature].idxmax()].GalaxyID
    img = plt.imread(DIR_PROCESSED + file.replace('jpg','png'), format='png')
    plt.imshow(img, aspect='auto', cmap='gray')
    plt.xlabel(str(i))
fig.tight_layout(pad=1.0)
plt.savefig('Image_Example_color.png')  

