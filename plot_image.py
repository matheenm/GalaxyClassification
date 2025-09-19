import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

TRAINING_IMAGES = "galaxy_data/images_training_rev1/"
labels = pd.read_csv('galaxy_data/training_solutions_rev1.csv')

classes = ['Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2','Class5.3', 'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3',       'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6','Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3',  'Class11.4', 'Class11.5', 'Class11.6']
galaxy_shape = ['Smooth','Featured or disc','Star or artifact','Edge on','Not edge on','Bar through center','No bar','Spiral','No Spiral','No bulge','Just noticeable bulge','Obvious bulge','Dominant bulge','Odd Feature','No Odd Feature','Completely round','In between','Cigar shaped','Ring (Oddity)','Lens or arc (Oddity)','Disturbed (Oddity)','Irregular (Oddity)','Other (Oddity)','Merger (Oddity)','Dust lane (Oddity)','Rounded bulge','Boxy bulge','No bulge','Tightly wound arms','Medium wound arms','Loose wound arms','1 Spiral Arm','2 Spiral Arms','3 Spiral Arms','4 Spiral Arms','More than four Spiral Arms',"Can't tell"]

#make tuples
galaxt_type_tupple=[]
for cl,gs in zip(classes,galaxy_shape):
    galaxt_type_tupple.append((cl,gs))


gal_id = 495381


fig = plt.figure('Galaxy_Example', figsize=(20, 20))
# Create axes for the image (left 80% of the figure width)
ax_img = fig.add_axes([0, 0, 0.8, 1])  # [left, bottom, width, height]
ax_img.axis('off')
ax_img.tick_params(axis='both', which='both', 
                   left=False, top=False, right=False, bottom=False,
                   labelleft=False, labeltop=False, labelright=False, labelbottom=False)

# Load and display the image
img = plt.imread(f'{TRAINING_IMAGES}{gal_id}.jpg', format='jpg')
ax_img.imshow(img, aspect='auto')
probs = row_array = labels[labels['GalaxyID'] == gal_id].values[0][1:]
text=''
it=0
for typ in classes:
    text = text + f'\n{galaxy_shape[it]} = {probs[it]:.2f}'
    it+=1

text = f'ID: {gal_id}{text}'

# Place text in the white space (right 20% of the figure)
fig.text(0.9, 0.5, text, fontsize= 20, ha='center', va='center')
plt.savefig('Galaxy_Example.png', bbox_inches='tight')
plt.close()



