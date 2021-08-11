import cv2
import numpy as np
import os
import random


# set the random seed to ensure repeatable results
SEED = 1

# random sample 150 positives and 150 negatives
TEST_N = 150


filepaths = {}

yes_folder = os.path.join('raw_data','yes')
no_folder = os.path.join('raw_data','no')

filepaths['yes'] = [os.path.join(yes_folder ,x)for x in next(os.walk(os.path.join(yes_folder)))[2]]
filepaths['no'] = [os.path.join(no_folder ,x)for x in next(os.walk(os.path.join(no_folder)))[2]]

filepaths['no'][0:5]

'''
Find the extreme points of the brain and crop and resize to 224 x 224
'''
def process_image(image, size=(224,224)):
    
    image = cv2.GaussianBlur(image, (5,5), 0)
    thresh = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    c = max(contours, key = cv2.contourArea)
    
    p0 = np.min(c,axis=0)[0]
    p1 = np.max(c,axis=0)[0]
    
    return cv2.resize(image[p0[1]:p1[1], p0[0]:p1[0]], size, interpolation=cv2.INTER_CUBIC)


# load and process the images
positives = [process_image(cv2.imread(x,0)) for x in filepaths['yes']]
negatives = [process_image(cv2.imread(x,0)) for x in filepaths['no']]


# create folders
try:
    os.mkdir(os.path.join('train_data'))
except FileExistsError:
    print("train_data exists")

try:
    os.mkdir(os.path.join('train_data', 'yes'))
except FileExistsError:
    print("train_data/yes exists")

try:
    os.mkdir(os.path.join('train_data', 'no'))
except FileExistsError:
    print("train_data/no exists")

try:
    os.mkdir(os.path.join('test_data'))
except FileExistsError:
    print("test_data exists")

try:
    os.mkdir(os.path.join('test_data', 'yes'))
except FileExistsError:
    print("test_data/yes exists")

try:
    os.mkdir(os.path.join('test_data', 'no'))
except FileExistsError:
    print("test_data/no exists")

# shuffle positives and negatives to randomly slice the test and training set
random.seed(SEED)

# shuffle positives and negatives
random.shuffle(positives)
random.shuffle(negatives)

# write the images to their correct folders
for i, image in enumerate(negatives[0:TEST_N]):
    cv2.imwrite(os.path.join('test_data', 'no', str(i).zfill(5)+'_no_test.png'), image)

for i, image in enumerate(negatives[TEST_N:]):
    cv2.imwrite(os.path.join('train_data', 'no', str(i).zfill(5)+'_no.png'), image)

for i, image in enumerate(positives[0:TEST_N]):
    cv2.imwrite(os.path.join('test_data', 'yes', str(i).zfill(5)+'_yes_test.png'), image)

for i, image in enumerate(positives[TEST_N:]):
    cv2.imwrite(os.path.join('train_data', 'yes', str(i).zfill(5)+'_yes.png'), image)
