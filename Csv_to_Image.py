import numpy as np
import pandas as pd
from PIL import Image
import os

train_path = './data/train/'
test_path = './data/test/'
data_path = './data/fer2013.csv'

def make_dir():
    for i in range(7):
        path1 = os.path.join(train_path,str(i))
        path2 = os.path.join(test_path,str(i))
        if not os.path.exists(path1):
            os.makedirs(path1)
        if not os.path.exists(path2):
            os.makedirs(path2)       

def save_images():
    df = pd.read_csv(data_path)
    train_i = [1 for i in range(0,7)]
    test_i = [1 for i in range(0,7)]
    for index in range(len(df)):
        emotion = df.loc[index][0]
        image = df.loc[index][1]
        usage = df.loc[index][2]
        data_array = list(map(float, image.split()))
        data_array = np.asarray(data_array)
        image = data_array.reshape(48, 48)
        im = Image.fromarray(image).convert('L')
        if(usage=='Training'):
            train_p = os.path.join(train_path,str(emotion),'{}.jpg'.format(train_i[emotion]))
            im.save(train_p)
            train_i[emotion] += 1
        else:
            test_p = os.path.join(test_path,str(emotion),'{}.jpg'.format(test_i[emotion]))
            im.save(test_p)
            test_i[emotion] += 1

make_dir()
save_images()
