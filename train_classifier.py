
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

import keras
# 数据准备
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,# ((x/255)-0.5)*2  归一化到±1之间
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)


# In[ ]:


train_generator = train_datagen.flow_from_directory(directory='./datasets/plates/分类用/trainA',
                                  target_size=(128,128),
                                  batch_size=64)
val_generator = val_datagen.flow_from_directory(directory='./datasets/plates/分类用/testA',
                                target_size=(128,128),
                                batch_size=64)
TOTAL_CLASSES = 6


# In[ ]:


#ResNet Block
# 一个基本Block模块，是ResNet的最小单元
def resnet_block(inputs,num_filters=16,
                  kernel_size=3,strides=1,
                  activation='relu'):
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',
           kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    if(activation):
        x = Activation('relu')(x)
    return x


# In[ ]:


# 建一个20层的ResNet网络 
def resnet_v1(input_shape):
    inputs = Input(shape=input_shape)# Input层，用来当做占位使用
    # 第1层

    # x = Conv2D(filters=16,kernel_size=5,strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(inputs)
    x = resnet_block(inputs,kernel_size=5,strides=2)
    # output:128 * 128 * 3

    print('layer1,xshape:',x.shape)
    # 第2~7层
    for i in range(6):
        a = resnet_block(inputs = x)
        b = resnet_block(inputs=a,activation=None)
        x = keras.layers.add([x,b])
        x = Activation('relu')(x)
    # out：64*64*16
    # 第8~13层
    for i in range(6):
        if i == 0:
            a = resnet_block(inputs = x,strides=2,num_filters=32)
        else:
            a = resnet_block(inputs = x,num_filters=32)
        b = resnet_block(inputs=a,activation=None,num_filters=32)
        if i==0:
            x = Conv2D(32,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x,b])
        x = Activation('relu')(x)
    # out:32*32*32
    # 第14~19层
    for i in range(6):
        if i ==0 :
            a = resnet_block(inputs = x,strides=2,num_filters=64)
        else:
            a = resnet_block(inputs = x,num_filters=64)

        b = resnet_block(inputs=a,activation=None,num_filters=64)
        if i == 0:
            x = Conv2D(64,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x,b])# 相加操作，要求x、b shape完全一致
        x = Activation('relu')(x)
    # out:16*16*64

    for i in range(6):
        if i ==0 :
            a = resnet_block(inputs = x,strides=2,num_filters=64)
        else:
            a = resnet_block(inputs = x,num_filters=64)

        b = resnet_block(inputs=a,activation=None,num_filters=64)
        if i == 0:
            x = Conv2D(64,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x,b])# 相加操作，要求x、b shape完全一致
        x = Activation('relu')(x)
    # out:8*8*64

    # 第20层
    x = AveragePooling2D(pool_size=2)(x)
    # out:4*4*64
    y = Flatten()(x)
    # out:1024
    outputs = Dense(TOTAL_CLASSES,activation='softmax',
                    kernel_initializer='he_normal')(y)
    
    #初始化模型
    #之前的操作只是将多个神经网络层进行了相连，通过下面这一句的初始化操作，才算真正完成了一个模型的结构初始化
    model = Model(inputs=inputs,outputs=outputs)
    return model


# In[ ]:


model = resnet_v1((128,128,3))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])


# In[ ]:


tb = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=False,  # 是否存储网络结构图
                 write_grads=False, # 是否可视化梯度直方图
                 write_images=False,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
def lr_sch(epoch):
    #200 total
    if epoch <50:
        return 1e-3
    if 50<=epoch<100:
        return 1e-4
    if epoch>=100:
        return 1e-5
lr_scheduler = LearningRateScheduler(lr_sch)
checkpoint = ModelCheckpoint(filepath='./neuData_resnet_ckpt.h5',monitor='val_acc',
                             verbose=1,save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_acc',factor=0.2,patience=5,
                               mode='max',min_lr=1e-3)
callbacks = [tb,checkpoint,lr_scheduler,lr_reducer]


# In[ ]:


history_tl = model.fit_generator(generator=train_generator,
                    steps_per_epoch=1000,#800
                    epochs=200,#2
                    validation_data=val_generator,
                    validation_steps=12,#12
                    class_weight='auto',
                    callbacks=callbacks
                    )
model.save('./neuData_resnet.h5')

