#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Alexander Scheinker
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
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Conv3D, Conv3DTranspose, UpSampling2D
from tensorflow.keras.layers import concatenate, Add, MaxPool3D, UpSampling3D, Reshape, Multiply, MaxPool2D
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

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


PATH_TO_VOLUME_DATA  = '/pscratch/sd/j/jcurcio/pcnn/Volume_Data/'

#%%

# The 3D volumes have dimensions n_pixels*n_pixels*n_pixels
n_pixels = 128

n_timesteps = 200

# Import data

# Charge density
Q = np.zeros([n_timesteps,n_pixels,n_pixels,n_pixels,1])

# Non-zero charge density locations
Qnz = np.random.randn(n_timesteps, n_pixels, n_pixels, n_pixels, 1) * 0.01

# Electric field components
Ex = np.zeros([n_timesteps,n_pixels,n_pixels,n_pixels,1])
Ey = np.zeros([n_timesteps,n_pixels,n_pixels,n_pixels,1])
Ez = np.zeros([n_timesteps,n_pixels,n_pixels,n_pixels,1])

# Magnetic field components
Bx = np.zeros([n_timesteps,n_pixels,n_pixels,n_pixels,1])
By = np.zeros([n_timesteps,n_pixels,n_pixels,n_pixels,1])
Bz = np.zeros([n_timesteps,n_pixels,n_pixels,n_pixels,1])

# Current density components
Jx = np.random.randn(n_timesteps, n_pixels, n_pixels, n_pixels, 1) * 0.01
Jy = np.random.randn(n_timesteps, n_pixels, n_pixels, n_pixels, 1) * 0.01
Jz = np.random.randn(n_timesteps, n_pixels, n_pixels, n_pixels, 1) * 0.01

# Load the data
for n_load in np.arange(n_timesteps):

    Q[n_load] = np.load(PATH_TO_VOLUME_DATA + f'q_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])

    Ex[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ex_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Ey[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ey_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Ez[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ez_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])

    Bx[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Bx_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    By[n_load] = np.load(PATH_TO_VOLUME_DATA + f'By_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Bz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Bz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])

    Jx[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jx_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Jy[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jy_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Jz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])


# Latent space inputs, un-used, all zeros
z_input = np.zeros([n_timesteps,8,8,8,1]).astype(np.float32)

# Used to un-normalize PINN and no-physics CNN outputs
Bxyz_all_max = np.load(PATH_TO_VOLUME_DATA + 'Bxyz_max.npy')

# Normalize CNN inputs
J_max_max_all_128 = np.load(PATH_TO_VOLUME_DATA+'J_max_max_all_128.npy')

Jx = Jx/J_max_max_all_128
Jy = Jy/J_max_max_all_128
Jz = Jz/J_max_max_all_128

Bx = Bx/Bxyz_all_max
By = By/Bxyz_all_max
Bz = Bz/Bxyz_all_max

# Make 3D field data for the PINN and No-Physics CNNs for current density
Jxyz = np.zeros([n_timesteps,n_pixels,n_pixels,n_pixels,3])
Jxyz[:,:,:,:,0] = Jx[:,:,:,:,0]
Jxyz[:,:,:,:,1] = Jy[:,:,:,:,0]
Jxyz[:,:,:,:,2] = Jz[:,:,:,:,0]

Bxyz = np.zeros([n_timesteps,n_pixels,n_pixels,n_pixels,3])
Bxyz[:,:,:,:,0] = Bx[:,:,:,:,0]
Bxyz[:,:,:,:,1] = By[:,:,:,:,0]
Bxyz[:,:,:,:,2] = Bz[:,:,:,:,0]

# Non-zero charge density locations
Qnz_3D = np.zeros([n_timesteps,n_pixels,n_pixels,n_pixels,3]).astype(np.float32)
Qnz_3D[:,:,:,:,0] = Qnz[:,:,:,:,0]
Qnz_3D[:,:,:,:,1] = Qnz[:,:,:,:,0]
Qnz_3D[:,:,:,:,2] = Qnz[:,:,:,:,0]


# In[11]:

# Define the physical space

# Some physics constants
u0 = tf.constant(4.0*np.pi*1e-7, dtype=DTYPE)
e0 = tf.constant(8.85*1e-12, dtype=DTYPE)
cc = tf.constant(2.99792e8, dtype=DTYPE)

# Physical size of the volume around the beam
x_max_all = 6.992009440856081e-05
x_min_all = -6.997896993744958e-05

y_max_all = 6.99964109005157e-05
y_min_all = -6.949648426854683e-05

z_max_all = 1.1389556140820954e-06
z_min_all = 4.0295594061272973e-10

# More physics constants
me = 9.109384e-31
ce = 2.99792458e8
qe = 1.602e-19

# Size of one pixel
dx = (x_max_all-x_min_all)/(n_pixels-1)
dy = (y_max_all-y_min_all)/(n_pixels-1)
dz = (z_max_all-z_min_all)/(n_pixels-1)

# Axis for plotting
x_axis = np.linspace(x_min_all,x_max_all,n_pixels)
y_axis = np.linspace(y_min_all,y_max_all,n_pixels)
z_axis = np.linspace(z_min_all,z_max_all,n_pixels)

# Time step between saved beam volumes
dt = 5e-11

# Defined filters for derivatives as a convolutional layer
d_dx = np.zeros([3,3,3])
d_dx[0,1,1] = -1
d_dx[2,1,1] = 1
d_dx = tf.keras.initializers.Constant(d_dx/2)

d_dy = np.zeros([3,3,3])
d_dy[1,0,1] = -1
d_dy[1,2,1] = 1
d_dy = tf.keras.initializers.Constant(d_dy/2)

d_dz = np.zeros([3,3,3])
d_dz[1,1,0] = -1
d_dz[1,1,2] = 1
d_dz = tf.keras.initializers.Constant(d_dz/2)

# Single layerr 3D CNNs for taking partial x, y, and z derivatives
def NN_ddx():
    X = Input(shape = (n_pixels,n_pixels,n_pixels,1))
    X_x = Conv3D(1, kernel_size=3, kernel_initializer=d_dx, strides=[1,1,1], padding='SAME', trainable=False)(X)
    d_dx_model = Model(inputs=[X], outputs=[X_x])
    return d_dx_model

def NN_ddy():
    X = Input(shape = (n_pixels,n_pixels,n_pixels,1))
    X_y = Conv3D(1, kernel_size=3, kernel_initializer=d_dy, strides=[1,1,1], padding='SAME', trainable=False)(X)
    d_dy_model = Model(inputs=[X], outputs=[X_y])
    return d_dy_model

def NN_ddz():
    X = Input(shape = (n_pixels,n_pixels,n_pixels,1))
    X_z = Conv3D(1, kernel_size=3, kernel_initializer=d_dz, strides=[1,1,1], padding='SAME', trainable=False)(X)
    d_dz_model = Model(inputs=[X], outputs=[X_z])
    return d_dz_model

# Partial Derivative CNNs
mNN_ddx = NN_ddx()
mNN_ddy = NN_ddy()
mNN_ddz = NN_ddz()

# Latent space inputs, un-used, all zeros
z_input = np.zeros([n_timesteps,8,8,8,1]).astype(np.float32)

#%%

def Field_model():

    # Regularlization
    l2w = 1e-6

    # Various resolution image inputs
    X_in = Input(shape = (None,None,None,1))
    ES_in = Input(shape = (None,None,None,1))

    # Define a function to add a block of layers
    def conv_block(input_tensor, num_filters, kernel_size=3, strides=1, padding='same'):
        x = Conv3D(num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=l2_reg(l2w))(input_tensor)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv3D(num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=l2_reg(l2w))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    def downsample_block(input_tensor, num_filters, pool_size=(2,2,2)):
        x = conv_block(input_tensor, num_filters)
        p = MaxPooling3D(pool_size=pool_size)(x)
        return x, p

    def upsample_block(input_tensor, skip_tensor, num_filters):
        x = UpSampling3D(size=(2,2,2))(input_tensor)
        x = conv_block(x, num_filters)
        x = Add()([x, skip_tensor])
        return x

    # Contracting path
    c1, p1 = downsample_block(X_in, 16)
    c2, p2 = downsample_block(p1, 32)
    c3, p3 = downsample_block(p2, 64)
    c4, p4 = downsample_block(p3, 128)

    # Bottleneck
    b1 = conv_block(p4, 256)

    # Expansive path
    u1 = upsample_block(b1, c4, 128)
    u1 = upsample_block(u1, c3, 64)
    u1 = upsample_block(u2, c2, 32)
    u1 = upsample_block(u3, c1, 16)

    # Final convolutional layer
    outputs = Conv3D(1, kernel_size=1, activation='linear', dtype='float32')(u4)

    # Global average pooling to handle varying input sizes
    A = GlobalAveragePooling3D()(outputs)
    A = Dense(64, activation='relu', kernel_regularizer=l2_reg(l2w))(A)
    A = Dense(1, activation='linear')(A)
    A = tf.expand_dims(tf.expand_dims(tf.expand_dims(A, axis=1), axis=1), axis=1)

    # Define the model
    CNN_model = Model(inputs=[X_in, ES_in], outputs=[A, outputs])

    # Return the model
    return CNN_model


# In[12]:


def B_fields(A_model,Jx_in1,Jy_in1,Jz_in1,A_cut_now1,ES_in1):

    # Calculate vector potential fields
    Ax1, Ax1_yL = A_model([Jx_in1,ES_in1])
    Ay1, Ay1_yL = A_model([Jy_in1,ES_in1])
    Az1, Az1_yL = A_model([Jz_in1,ES_in1])

    # Take derivatives
    Ax1_y = mNN_ddy(Ax1)/dy
    Ax1_z = mNN_ddz(Ax1)/dz

    Ay1_x = mNN_ddx(Ay1)/dx
    Ay1_z = mNN_ddz(Ay1)/dz

    Az1_x = mNN_ddx(Az1)/dx
    Az1_y = mNN_ddy(Az1)/dy

    # Magnetic Fields
    Bx1 = Az1_y - Ay1_z
    By1 = Ax1_z - Az1_x
    Bz1 = Ay1_x - Ax1_y

    Bx1 = Bx1*A_cut_now1
    By1 = By1*A_cut_now1
    Bz1 = Bz1*A_cut_now1

    return Ax1, Ay1, Az1, Bx1, By1, Bz1


# In[13]:


def A_fields_only(A_model,Jx_in1,Jy_in1,Jz_in1,A_cut_now1,ES_in1):

    # Calculate vector potential fields
    Ax1, Ax1_yL = A_model([Jx_in1,ES_in1])
    Ay1, Ay1_yL = A_model([Jy_in1,ES_in1])
    Az1, Az1_yL = A_model([Jz_in1,ES_in1])

    # Cut off
    Ax1 = Ax1*A_cut_now1
    Ay1 = Ay1*A_cut_now1
    Az1 = Az1*A_cut_now1

    return Ax1, Ay1, Az1


# In[14]:


def E_fields(V_model,Q_in2,Ax2_t,Ay2_t,Az2_t,A_cut_now2,ES_in1):

    # Calculate voltage fields
    V2, V2_yL = V_model([Q_in2,ES_in1])

    # Take derivatives
    V2_x = mNN_ddx(V2)/dx
    V2_y = mNN_ddy(V2)/dy
    V2_z = mNN_ddz(V2)/dz

    # Electric fields
    Ex2 = -Ax2_t - V2_x
    Ey2 = -Ay2_t - V2_y
    Ez2 = -Az2_t - V2_z

    Ex2 = Ex2*A_cut_now2
    Ey2 = Ey2*A_cut_now2
    Ez2 = Ez2*A_cut_now2

    return V2, Ex2, Ey2, Ez2



# In[15]:


def V_fields_only(V_model,Q_in2,A_cut_now2,ES_in1):

    # Calculate voltage fields
    V2, V2_yL = V_model([Q_in2,ES_in1])

    # Cut off
    V2 = V2*A_cut_now2

    return V2


# In[16]:


def A_Phi_constraint(A_model,ES_in1,
                     Jx_in1, Jy_in1, Jz_in1, A_cut_now1):

    # Calculate A and B fields
    Ax1, Ay1, Az1, Bx1, By1, Bz1 = B_fields(A_model,Jx_in1,Jy_in1,Jz_in1,A_cut_now1,ES_in1)

    return Bx1,By1,Bz1


# In[24]:


def compute_loss(A_model,ES_in1,
                     Jx_in1, Jy_in1, Jz_in1, A_cut_now1,
                     Bx1_tr, By1_tr, Bz1_tr
                 ):

    Bx1,By1,Bz1 = A_Phi_constraint(A_model,ES_in1,
                         Jx_in1, Jy_in1, Jz_in1, A_cut_now1)

    # B field
    loss_Bx = tf.reduce_mean(tf.square(Bx1-Bx1_tr))
    loss_By = tf.reduce_mean(tf.square(By1-By1_tr))
    loss_Bz = tf.reduce_mean(tf.square(Bz1-Bz1_tr))*100.0

    # Total loss
    loss_B = loss_Bx + loss_By + loss_Bz

    return loss_B, loss_Bx, loss_By, loss_Bz


# In[25]:


def get_grad(A_model,ES_in1,
                     Jx_in1, Jy_in1, Jz_in1, A_cut_now1,
                     Bx1_tr, By1_tr, Bz1_tr
                 ):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(A_model.trainable_variables)
        loss_B, loss_Bx, loss_By, loss_Bz = compute_loss(A_model,ES_in1,
                             Jx_in1, Jy_in1, Jz_in1, A_cut_now1,
                             Bx1_tr, By1_tr, Bz1_tr
                         )
    gA = tape.gradient(loss_B, A_model.trainable_variables)

    del tape

    return loss_B, gA, loss_Bx, loss_By, loss_Bz


# In[19]:

model_A = Field_model()
lr = 1e-4
optim_A = tf.keras.optimizers.Adam(learning_rate = lr)
model_A.summary()

#%%


def train_step(model_A, ES_in1,
                     Jx_in1, Jy_in1, Jz_in1, A_cut_now1,
                     Bx1_tr, By1_tr, Bz1_tr):

    loss_B, gA, loss_Bx, loss_By, loss_Bz = get_grad(model_A,ES_in1,
                         Jx_in1, Jy_in1, Jz_in1, A_cut_now1,
                         Bx1_tr, By1_tr, Bz1_tr
                     )

    optim_A.apply_gradients(zip(gA, model_A.trainable_variables))

    return loss_B, gA, loss_Bx, loss_By, loss_Bz


# In[ ]:


# Number of epochs
N_epochs = 10

# Number of training data points to look at
#Nt = int(0.75 * n_timesteps)
Nt = 2

hist_B = []
hist_Bx = []
hist_By = []
hist_Bz = []

t11 = time.time()

for n_ep in range(N_epochs):
    for n_t in range(Nt):
        print(f'Starting single step {n_t + 1}/{Nt} of epoch {n_ep + 1}/{N_epochs}.')
        t1 = time.time()

        loss_B, gA, loss_Bx, loss_By, loss_Bz = train_step(
            model_A,
            z_input[n_t:n_t + 1],
            Jx[n_t:n_t + 1],
            Jy[n_t:n_t + 1],
            Jz[n_t:n_t + 1],
            Qnz[n_t:n_t + 1],
            Bx[n_t:n_t + 1],
            By[n_t:n_t + 1],
            Bz[n_t:n_t + 1]
        )

        # print("Intermediate Values:")
        # print(f"z_input: {z_input[n_t:n_t + 1]}")
        # print(f"Jx: {Jx[n_t:n_t + 1]}")
        # print(f"Jy: {Jy[n_t:n_t + 1]}")
        # print(f"Jz: {Jz[n_t:n_t + 1]}")
        # print(f"Qnz: {Qnz[n_t:n_t + 1]}")
        # print(f"Bx: {Bx[n_t:n_t + 1]}")
        # print(f"By: {By[n_t:n_t + 1]}")
        # print(f"Bz: {Bz[n_t:n_t + 1]}")
        # print(f"loss_B: {loss_B}")
        # print(f"loss_Bx: {loss_Bx}")
        # print(f"loss_By: {loss_By}")
        # print(f"loss_Bz: {loss_Bz}")

        print('\n')
        print(f'Loss B = {loss_B:.11f}')
        print(f'Loss Bx = {loss_Bx:.11f}')
        print(f'Loss By = {loss_By:.11f}')
        print(f'Loss Bz = {loss_Bz:.11f}')
        print('\n')

        hist_B.append(loss_B.numpy())
        hist_Bx.append(loss_Bx.numpy())
        hist_By.append(loss_By.numpy())
        hist_Bz.append(loss_Bz.numpy())

        t2 = time.time()
        print(f'Step time: {t2 - t1:2f} seconds')

t22 = time.time()
print(f'Total time: {t22 - t11:2f} seconds')

# ------------------------------
# Plotting the results at the end
# Convert lists to numpy arrays for easier manipulation
hist_B = np.array(hist_B)
hist_Bx = np.array(hist_Bx)
hist_By = np.array(hist_By)
hist_Bz = np.array(hist_Bz)

# Generate an array representing each training step
steps = np.arange(1, len(hist_B) + 1)

# Plotting the results
plt.figure(figsize=(12, 8))

# Function to add annotations
def add_annotations(ax, steps, values):
    for i, (step, value) in enumerate(zip(steps, values)):
        ax.annotate(f'{value:.2f}', (step, value), textcoords="offset points", xytext=(0, 5), ha='center')

# Plot Loss B
ax1 = plt.subplot(4, 1, 1)
ax1.plot(steps, hist_B, label='Loss B', marker='o')
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Loss B')
ax1.set_title('Loss B over Training Steps')
ax1.grid(True)
add_annotations(ax1, steps, hist_B)

# Plot Loss Bx
ax2 = plt.subplot(4, 1, 2)
ax2.plot(steps, hist_Bx, label='Loss Bx', marker='o')
ax2.set_xlabel('Training Steps')
ax2.set_ylabel('Loss Bx')
ax2.set_title('Loss Bx over Training Steps')
ax2.grid(True)
add_annotations(ax2, steps, hist_Bx)

# Plot Loss By
ax3 = plt.subplot(4, 1, 3)
ax3.plot(steps, hist_By, label='Loss By', marker='o')
ax3.set_xlabel('Training Steps')
ax3.set_ylabel('Loss By')
ax3.set_title('Loss By over Training Steps')
ax3.grid(True)
add_annotations(ax3, steps, hist_By)

# Plot Loss Bz
ax4 = plt.subplot(4, 1, 4)
ax4.plot(steps, hist_Bz, label='Loss Bz', marker='o')
ax4.set_xlabel('Training Steps')
ax4.set_ylabel('Loss Bz')
ax4.set_title('Loss Bz over Training Steps')
ax4.grid(True)
add_annotations(ax4, steps, hist_Bz)

# Adjust layout
plt.tight_layout()
plt.show()
# ------------------------------