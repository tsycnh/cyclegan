from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
from matplotlib.colors import NoNorm
from keras.models import model_from_json
import shutil
def create_new_empty_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
class CycleGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'plates'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0  # Cycle-consistency loss
        self.lambda_id = 0.0      # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        # 两个判别器，A和B
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generators
        # 两个生成器
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        self.g_AB.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.g_BA.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        # 图像翻译
        fake_B = self.g_AB(img_A)# 通过真实的图像A生成假的图像B
        fake_A = self.g_BA(img_B)# 通过真是的图像B生成假的图像A
        # Translate images back to original domain
        # 将刚刚生成好的假图像再翻译回去
        reconstr_A = self.g_BA(fake_B)#
        reconstr_B = self.g_AB(fake_A)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        # 判别翻译过来的假图质量咋样
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model([img_A, img_B], [valid_A, valid_B, fake_B, fake_A,reconstr_A, reconstr_B])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                                    loss_weights=[1, 1, self.lambda_id, self.lambda_id,self.lambda_cycle, self.lambda_cycle],
                                    optimizer=optimizer)
        self.save_structure()
    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Conv2D(self.df, kernel_size=4, strides=2, padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.8))
        model.add(Conv2D(self.df*2, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())
        model.add(Conv2D(self.df*4, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())
        model.add(Conv2D(self.df*8, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())
        model.add(Conv2D(1, kernel_size=4, strides=1, padding='same'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        half_batch = int(batch_size / 2)

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminators
            # ----------------------

            imgs_A = self.data_loader.load_data(domain="A", batch_size=half_batch)
            imgs_B = self.data_loader.load_data(domain="B", batch_size=half_batch)

            # Translate images to opposite domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)

            valid = np.ones((half_batch,) + self.disc_patch)
            fake = np.zeros((half_batch,) + self.disc_patch)

            # Train the discriminators (original images = real / translated = Fake)
            dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
            dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

            # Total disciminator loss
            d_loss = 0.5 * np.add(dA_loss, dB_loss)


            # ------------------
            #  Train Generators
            # ------------------

            # Sample a batch of images from both domains
            imgs_A = self.data_loader.load_data(domain="A", batch_size=batch_size)
            imgs_B = self.data_loader.load_data(domain="B", batch_size=batch_size)

            # The generators want the discriminators to label the translated images as real
            # ！！生成器需要认假为真
            valid = np.ones((batch_size,) + self.disc_patch)

            # Train the generators
            # combined 模型结构              Model([img_A, img_B], [valid_A, valid_B, fake_B, fake_A,reconstr_A, reconstr_B])

            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.epoch = epoch
                self.save_imgs(epoch)
                self.g_AB.save('./saved_model/model_epoch_'+str(epoch)+'.h5')
                create_new_empty_dir('predict_images/epoch_%d/'%(self.epoch))
                self.predicts_from_A_to_B()
    def save_structure(self):
        g_AB_json_string = self.g_AB.to_json()
        fh = open('./saved_model/g_AB_model.json', mode='w')
        fh.write(g_AB_json_string)
        fh.close()
    def save_imgs(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt],-1),cmap='gray',norm=NoNorm())#squeeze减少不必要维度，NoNorm去除默认的归一化
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

    def predicts_from_A_to_B(self,model_path=None):
        np.random.seed(0)
        if model_path != None:
            self.g_AB.load_weights(model_path)
        imgs_A = self.data_loader.load_data(domain='A',batch_size=100)
        imgs_B = self.data_loader.load_data(domain='B',batch_size=100)
        fake_B = self.g_AB.predict(imgs_A)
        r, c = 5, 3 # 5行2列
        candidates = []
        for ai in range(0,len(imgs_A)):
            candidates.append(imgs_A[ai]*0.5+0.5)
            candidates.append(fake_B[ai]*0.5+0.5)
            candidates.append(imgs_B[ai]*0.5+0.5)
            if (ai+1)%10==0:
                cnt = 0
                titles = ['Origin A', 'Translation B','Origin B']
                # candidates = np.concatenate(candidates)
                # candidates = 0.5 * candidates + 0.5
                fig,axs = plt.subplots(nrows=r,ncols=c,figsize=(10,10))
                for ti,title in enumerate(titles):
                    axs[0, ti].set_title(title)

                for i in range(r):
                    for j in range(c):
                        axs[i, j].imshow(np.squeeze(candidates[cnt], -1), cmap='gray',
                                         norm=NoNorm())  # squeeze减少不必要维度，NoNorm去除默认的归一化
                        axs[i, j].axis('off')
                        cnt += 1
                fig.savefig('predict_images/epoch_%d/%d.png'%(self.epoch,ai+1))
                candidates = []
if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=10001, batch_size=2, save_interval=100)
    # gan.predicts_from_A_to_B('good_model/model_epoch_1800.h5')