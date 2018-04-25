import scipy
import scipy.misc
from glob import glob
import numpy as np
import random

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.img_paths = []

    def load_data(self, domain, batch_size=1, is_testing=False):
        self.img_paths = []
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        random.shuffle(path)

        # batch_images = np.random.choice(path, size=batch_size)
        batch_images = path[0:batch_size]
        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                # if np.random.random() > 0.5:
                #     img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)
            self.img_paths.append(img_path)
        if len(imgs[0].shape) <3:
            imgs = np.expand_dims(imgs,axis=-1)
        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='L')

    def get_imgs_path(self):
        return self.img_paths

    def get_imgs_name(self):
        img_names =self.img_paths
        for i,path in enumerate(img_names):
            print(path)
            b = path.split('/')
            img_names[i] = b[-1]
        return img_names
