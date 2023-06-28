import numpy as np
import h5py
from imageio import imread
import os
import time
import cv2


annot_dir = 'data/Tooth/annot'
img_dir = 'data/Tooth/images'

# assert os.path.exists(img_dir)
tooth, num_examples_train, num_examples_val = None, None, None


class Tooth:
    def __init__(self):
        print('loading data...')
        tic = time.time()

        labels = np.loadtxt(os.path.join(annot_dir, "kp_label.txt"),
                   dtype={'names': ('img_name', 'left_x', 'left_y', 'mid_x', 'mid_y', 'right_x', 'right_y'),
                          'formats': ('|S200', float, float, float, float, float, float)},
                   delimiter=';', skiprows=0)

        img_names = []
        parts = []
        visibles = []
        for i in range(len(labels)):
            img_names.append(str(labels['img_name'][i].decode("utf8")))
            parts.append([[labels['left_x'][i], labels['left_y'][i]], [labels['mid_x'][i], labels['mid_y'][i]], [labels['right_x'][i], labels['right_y'][i]]])
            visibles.append([1, 1, 1])

        self.imgname = img_names
        self.part = parts
        self.visible = visibles
        
        print('Done (t={:0.2f}s)'.format(time.time() - tic))
        
    def getAnnots(self, idx):
        """
        returns h5 file for train or val set
        """
        return self.imgname[idx], self.part[idx], self.visible[idx]
    
    def getLength(self):
        train_num = int(len(self.imgname) * 0.85)
        return train_num, len(self.imgname) - train_num


def init():
    global tooth, num_examples_train, num_examples_val
    tooth = Tooth()
    num_examples_train, num_examples_val = tooth.getLength()


# Part reference
parts = {'tooth': ['left', 'mid', 'right']}

flipped_parts = {'tooth': [2, 1, 0]}

part_pairs = {'tooth': [[0, 1], [1, 2]]}

pair_names = {'tooth': ['left_part', 'right_part']}


def setup_val_split():
    """
    returns index for train and validation imgs
    index for validation images starts after that of train images
    so that loadImage can tell them apart
    """
    valid = [i+num_examples_train for i in range(num_examples_val)]
    train = [i for i in range(num_examples_train)]
    return np.array(train), np.array(valid)


def get_img(idx):
    imgname, __, __ = tooth.getAnnots(idx)
    path = os.path.join(img_dir, imgname)
    img = cv2.imread(path)
    return img


def get_path(idx):
    imgname, __, __ = tooth.getAnnots(idx)
    path = os.path.join(img_dir, imgname)
    return path


def get_kps(idx):
    __, part, visible = tooth.getAnnots(idx)
    kp2 = np.insert(part, 2, visible, axis=1)
    kps = np.zeros((1, 3, 3))
    kps[0] = kp2
    return kps


if __name__ == "__main__":
    annot_dir = './annot'
    img_dir = './images'
    tooth = Tooth()
    num_examples_train, num_examples_val = tooth.getLength()
    print(num_examples_train, num_examples_val)
    data_0 = tooth.getAnnots(0)
    print(data_0)