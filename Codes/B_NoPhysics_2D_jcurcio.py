#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jason Curcio
@credit: Alexander Scheinker

"""


import numpy as np
import h5py


import tensorflow as tf
from tensorflow.keras import mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)


from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import concatenate, Add, MaxPool2D, UpSampling2D, Reshape, Multiply, MaxPool2D
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model

from tensorflow import keras

from tensorflow.keras.regularizers import l2 as l2_reg
from tensorflow.keras.regularizers import l1 as l1_reg
from tensorflow.keras.regularizers import l1_l2 as l1_l2_reg
  
import matplotlib.pyplot as plt
import matplotlib
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import random
from scipy.io import loadmat
from scipy import misc
import os
import csv
# from sklearn.preprocessing import QuantileTransformer, StandardScaler

from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import zoom
from scipy import ndimage

import gc
import pickle


PATH_TO_VOLUME_DATA  = '.../Volume_Data/'


#%%

# The 2D volumes have dimensions n_pixels*n_pixels
n_pixels = 128

# Import data (Small data set with first 2 time steps.)

# Charge density
Q = np.zeros([2,n_pixels,n_pixels,1])

# Non-zero charge density locations
Qnz = np.zeros([2,n_pixels,n_pixels,1])

# Electric field components
Ex = np.zeros([2,n_pixels,n_pixels,1])
Ey = np.zeros([2,n_pixels,n_pixels,1])
Ez = np.zeros([2,n_pixels,n_pixels,1])

# Magnetic field components
Bx = np.zeros([2,n_pixels,n_pixels,1])
By = np.zeros([2,n_pixels,n_pixels,1])
Bz = np.zeros([2,n_pixels,n_pixels,1])

# Current density components
Jx = np.zeros([2,n_pixels,n_pixels,1])
Jy = np.zeros([2,n_pixels,n_pixels,1])
Jz = np.zeros([2,n_pixels,n_pixels,1])

# Load the data
for n_load in np.arange(2):

    Q[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Q_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    
    Qnz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Qnz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    
    Ex[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ex_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    Ey[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ey_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    Ez[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ez_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    
    Bx[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Bx_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    By[n_load] = np.load(PATH_TO_VOLUME_DATA + f'By_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    Bz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Bz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    
    Jx[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jx_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    Jy[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jy_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    Jz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,1])
    

# Latent space inputs, un-used, all zeros
z_input = np.zeros([2,8,8,1]).astype(np.float32)

# Used to un-normalize PINN and no-physics CNN outputs
Bxyz_all_max = np.load(PATH_TO_VOLUME_DATA + 'Bxyz_max.npy')

# Normalize CNN inputs
J_max_max_all_128 = np.load(PATH_TO_VOLUME_DATA+'J_max_max_all_128.npy')

Jx = Jx/J_max_max_all_128
Jy = Jy/J_max_max_all_128

Bx = Bx/Bxyz_all_max
By = By/Bxyz_all_max

# Make 2D field data for the PINN and No-Physics CNNs for current density
Jxy = np.zeros([2,n_pixels,n_pixels,2])
Jxy[:,:,:,0] = Jx[:,:,:,0]
Jxy[:,:,:,1] = Jy[:,:,:,0]

Bxy = np.zeros([2,n_pixels,n_pixels,2])
Bxy[:,:,:,0] = Bx[:,:,:,0]
Bxy[:,:,:,1] = By[:,:,:,0]

# Non-zero charge density locations
Qnz_2D = np.zeros([2,n_pixels,n_pixels,2]).astype(np.float32)
Qnz_2D[:,:,:,0] = Qnz[:,:,:,0]
Qnz_2D[:,:,:,1] = Qnz[:,:,:,0]


#%%


def Field_model():
    
    # Regularlization
    l2w = 1e-6
    
    # Various resolution image inputs
    X_in = Input(shape = (128,128,2))
    Qnz_in = Input(shape = (128,128,1))
    ES_in = Input(shape = (8,8,1))
    
    def B_from_J(X_now,ES_now):
    
        qPxPy_128 = Conv2D(16, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(X_now)
        qPxPy_128 = BatchNormalization()(qPxPy_128)
        qPxPy_128 = layers.LeakyReLU(alpha=0.1)(qPxPy_128)
        qPxPy_128 = Conv2D(16, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(qPxPy_128)
        qPxPy_128 = BatchNormalization()(qPxPy_128)
        qPxPy_128 = layers.LeakyReLU(alpha=0.1)(qPxPy_128)
        # 128x128
        
        qPxPy_64 = MaxPool2D(pool_size=(2,2))(qPxPy_128)
        # 64x64
        
        qPxPy_64 = Conv2D(32, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(qPxPy_64)
        qPxPy_64 = BatchNormalization()(qPxPy_64)
        qPxPy_64 = layers.LeakyReLU(alpha=0.1)(qPxPy_64)
        qPxPy_64 = Conv2D(32, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(qPxPy_64)
        qPxPy_64 = BatchNormalization()(qPxPy_64)
        qPxPy_64 = layers.LeakyReLU(alpha=0.1)(qPxPy_64)
        # 64x64
        
        qPxPy_32 = MaxPool2D(pool_size=(2,2))(qPxPy_64)
        # 32x32
        
        qPxPy_32 = Conv2D(64, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(qPxPy_32)
        qPxPy_32 = BatchNormalization()(qPxPy_32)
        qPxPy_32 = layers.LeakyReLU(alpha=0.1)(qPxPy_32)
        qPxPy_32 = Conv2D(64, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(qPxPy_32)
        qPxPy_32 = BatchNormalization()(qPxPy_32)
        qPxPy_32 = layers.LeakyReLU(alpha=0.1)(qPxPy_32)
        # 32x32
        
        qPxPy_16 = MaxPool2D(pool_size=(2,2))(qPxPy_32)
        # 16x16
        
        qPxPy_16 = Conv2D(128, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(qPxPy_16)
        qPxPy_16 = BatchNormalization()(qPxPy_16)
        qPxPy_16 = layers.LeakyReLU(alpha=0.1)(qPxPy_16)
        qPxPy_16 = Conv2D(128, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(qPxPy_16)
        qPxPy_16 = BatchNormalization()(qPxPy_16)
        qPxPy_16 = layers.LeakyReLU(alpha=0.1)(qPxPy_16)
        # 16x16
        
        qPxPy_8 = MaxPool2D(pool_size=(2,2))(qPxPy_16)
        # 8x8
        
        qPxPy_8 = Conv2D(64, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(qPxPy_8)
        qPxPy_8 = BatchNormalization()(qPxPy_8)
        qPxPy_8 = layers.LeakyReLU(alpha=0.1)(qPxPy_8)
        qPxPy_8 = Conv2D(64, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(qPxPy_8)
        qPxPy_8 = BatchNormalization()(qPxPy_8)
        qPxPy_8 = layers.LeakyReLU(alpha=0.1)(qPxPy_8)
        # 8x8
        qPxPy_8 = Conv2D(1, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(qPxPy_8)
        qPxPy_8 = BatchNormalization()(qPxPy_8)
        qPxPy_8 = layers.LeakyReLU(alpha=0.1)(qPxPy_8)
        # 8x8x1
        
        adapt_in = Add()([qPxPy_8,ES_now])
        # 8x8x1
        
        A_8 = Conv2D(64, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(adapt_in)
        A_8 = BatchNormalization()(A_8)
        A_8 = layers.LeakyReLU(alpha=0.1)(A_8)
        A_8 = Conv2D(64, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(A_8)
        A_8 = BatchNormalization()(A_8)
        A_8 = layers.LeakyReLU(alpha=0.1)(A_8)
        # 8x8
        
        A_16 = Conv2DTranspose(64, kernel_size=3, strides=(2,2,2), padding='same', kernel_regularizer=l2_reg(l2w))(A_8)
        A_16 = BatchNormalization()(A_16)
        A_16 = layers.LeakyReLU(alpha=0.1)(A_16)
        # 16x16
        A_16 = Conv2D(64, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(A_16)
        A_16 = BatchNormalization()(A_16)
        A_16 = layers.LeakyReLU(alpha=0.1)(A_16)
        A_16 = Conv2D(64, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(A_16)
        A_16 = BatchNormalization()(A_16)
        A_16 = layers.LeakyReLU(alpha=0.1)(A_16)
        # 16x16
        
        A_32 = Conv2DTranspose(64, kernel_size=3, strides=(2,2,2), padding='same', kernel_regularizer=l2_reg(l2w))(A_16)
        A_32 = BatchNormalization()(A_32)
        A_32 = layers.LeakyReLU(alpha=0.1)(A_32)
        # 32x32
        A_32 = Conv2D(64, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(A_32)
        A_32 = BatchNormalization()(A_32)
        A_32 = layers.LeakyReLU(alpha=0.1)(A_32)
        A_32 = Conv2D(64, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(A_32)
        A_32 = BatchNormalization()(A_32)
        A_32 = layers.LeakyReLU(alpha=0.1)(A_32)
        # 32x32
        
        A_64 = Conv2DTranspose(32, kernel_size=3, strides=(2,2,2), padding='same', kernel_regularizer=l2_reg(l2w))(A_32)
        A_64 = BatchNormalization()(A_64)
        A_64 = layers.LeakyReLU(alpha=0.1)(A_64)
        # 64x64
        A_64 = Conv2D(32, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(A_64)
        A_64 = BatchNormalization()(A_64)
        A_64 = layers.LeakyReLU(alpha=0.1)(A_64)
        A_64 = Conv2D(32, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(A_64)
        A_64 = BatchNormalization()(A_64)
        A_64 = layers.LeakyReLU(alpha=0.1)(A_64)
        # Ax_64 = Conv2D(1, kernel_size=3, strides=(1,1,1), padding='same', activation='linear')(Ax_64)
        # 64x64
        
        A_128 = Conv2DTranspose(16, kernel_size=3, strides=(2,2,2), padding='same', kernel_regularizer=l2_reg(l2w))(A_64)
        A_128 = BatchNormalization()(A_128)
        A_128 = layers.LeakyReLU(alpha=0.1)(A_128)
        # 128x128
        A_128 = Conv2D(16, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(A_128)
        A_128 = BatchNormalization()(A_128)
        A_128 = layers.LeakyReLU(alpha=0.1)(A_128)
        A_128 = Conv2D(16, kernel_size=3, strides=(1,1,1), padding='same', kernel_regularizer=l2_reg(l2w))(A_128)
        A_128 = BatchNormalization()(A_128)
        A_128 = layers.LeakyReLU(alpha=0.1)(A_128)
        # 128x128
        
        B_component = Conv2D(1, kernel_size=3, strides=(1,1,1), padding='same', activation='linear', dtype='float32')(A_128)
        # 128x128
        
        return B_component, adapt_in

    B_x, adapt_in = B_from_J(X_in,ES_in)
    B_y, adapt_in = B_from_J(X_in,ES_in)
    B_z, adapt_in = B_from_J(X_in,ES_in)
    
    B_x = Multiply()([B_x, Qnz_in])
    B_y = Multiply()([B_y, Qnz_in])
    B_z = Multiply()([B_z, Qnz_in])
    
    B_xyz = concatenate([B_x, B_y, B_z])
    
    # Define the model
    CNN_model = Model(inputs=[X_in,ES_in,Qnz_in], outputs=[B_xyz,adapt_in])

    #Return the model
    return CNN_model



#%%

B_CNN_model = Field_model()
B_CNN_model.summary()


#%%

n_t_1 = 0
n_t_2 = 75
n_epochs = 20
n_batch= 4


lr = 1e-3
print(f'Learning rate updated to {lr}.')
opt = tf.keras.optimizers.Adam(learning_rate = lr)
B_CNN_model.compile(optimizer=opt,loss=['mse','mse'],loss_weights=[1.0,0.0])
history1 = B_CNN_model.fit([Jxyz[0:2], z_input[0:2], Qnz_3D[0:2]], [Bxyz[0:2], z_input[0:2]], batch_size=n_batch, epochs=n_epochs)