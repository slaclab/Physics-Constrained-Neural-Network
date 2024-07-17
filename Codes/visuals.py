#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Paths to your saved .npy files
PATH_TO_VOLUME_DATA = '/pscratch/sd/j/jcurcio/pcnn/Volume_Data/'

# Function to load and visualize 3D data
def visualize_data(filename):
    data = np.load(filename)
    print(f"Loaded {filename}: Shape {data.shape}")

    # Assuming data is structured as (128, 128, 128)
    n_pixels = data.shape[0]

    # Plot a 2D slice along the z-axis (middle slice)
    fig, ax = plt.subplots(figsize=(8, 8))

    slice_index = n_pixels // 2
    slice_data = data[:, :, slice_index]

    # Plotting the slice with adjusted colormap and scaling
    img = ax.imshow(slice_data, cmap='viridis', vmin=slice_data.min(), vmax=slice_data.max())
    ax.set_title(f'Slice at Z = {slice_index}')
    fig.colorbar(img, ax=ax)

    plt.suptitle(f'2D Slice Visualization of {filename}')
    plt.tight_layout()
    plt.show()

# Function to visualize 3D data with xyz components
def visualize_xyz_components(x_filename, y_filename, z_filename):
    x_data = np.load(x_filename)
    y_data = np.load(y_filename)
    z_data = np.load(z_filename)

    print(f"Loaded {x_filename}: Shape {x_data.shape}")
    print(f"Loaded {y_filename}: Shape {y_data.shape}")
    print(f"Loaded {z_filename}: Shape {z_data.shape}")

    # Assuming data is structured as (128, 128, 128)
    n_pixels = x_data.shape[0]

    # Plot slices along the xy, xz, and yz planes (middle slices)
    slice_index = n_pixels // 2

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    # XY plane slice
    axs[0, 0].imshow(x_data[:, :, slice_index], cmap='viridis', vmin=x_data.min(), vmax=x_data.max())
    axs[0, 0].set_title('X component (XY plane)')

    axs[0, 1].imshow(y_data[:, :, slice_index], cmap='viridis', vmin=y_data.min(), vmax=y_data.max())
    axs[0, 1].set_title('Y component (XY plane)')

    axs[0, 2].imshow(z_data[:, :, slice_index], cmap='viridis', vmin=z_data.min(), vmax=z_data.max())
    axs[0, 2].set_title('Z component (XY plane)')

    # XZ plane slice
    axs[1, 0].imshow(x_data[:, slice_index, :], cmap='viridis', vmin=x_data.min(), vmax=x_data.max())
    axs[1, 0].set_title('X component (XZ plane)')

    axs[1, 1].imshow(y_data[:, slice_index, :], cmap='viridis', vmin=y_data.min(), vmax=y_data.max())
    axs[1, 1].set_title('Y component (XZ plane)')

    axs[1, 2].imshow(z_data[:, slice_index, :], cmap='viridis', vmin=z_data.min(), vmax=z_data.max())
    axs[1, 2].set_title('Z component (XZ plane)')

    # YZ plane slice
    axs[2, 0].imshow(x_data[slice_index, :, :], cmap='viridis', vmin=x_data.min(), vmax=x_data.max())
    axs[2, 0].set_title('X component (YZ plane)')

    axs[2, 1].imshow(y_data[slice_index, :, :], cmap='viridis', vmin=y_data.min(), vmax=y_data.max())
    axs[2, 1].set_title('Y component (YZ plane)')

    axs[2, 2].imshow(z_data[slice_index, :, :], cmap='viridis', vmin=z_data.min(), vmax=z_data.max())
    axs[2, 2].set_title('Z component (YZ plane)')

    plt.suptitle('Visualization of 3D Data with XYZ Components')
    plt.tight_layout()
    plt.show()



n_pixels = 128

# Charge density
Q = np.zeros([2,n_pixels,n_pixels,n_pixels,1])

# Non-zero charge density locations
Qnz = np.zeros([2,n_pixels,n_pixels,n_pixels,1])

# Electric field components
Ex = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
Ey = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
Ez = np.zeros([2,n_pixels,n_pixels,n_pixels,1])

# Magnetic field components
Bx = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
By = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
Bz = np.zeros([2,n_pixels,n_pixels,n_pixels,1])

# Current density components
Jx = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
Jy = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
Jz = np.zeros([2,n_pixels,n_pixels,n_pixels,1])

# Load the data
for n_load in np.arange(2):

    Q[n_load] = np.load(PATH_TO_VOLUME_DATA + f'q_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])

    #Qnz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Qnz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])

    Ex[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ex_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Ey[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ey_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Ez[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ez_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])

    Bx[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Bx_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    By[n_load] = np.load(PATH_TO_VOLUME_DATA + f'By_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Bz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Bz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])

    Jx[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jx_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Jy[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jy_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Jz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])

    print(Q[n_load].shape)
    print(Qnz[n_load].shape)
    print(Ex[n_load].shape)
    print(Ey[n_load].shape)
    print(Ez[n_load].shape)
    print(Bx[n_load].shape)
    print(By[n_load].shape)
    print(Bz[n_load].shape)
    print(Jx[n_load].shape)
    print(Jy[n_load].shape)
    print(Jz[n_load].shape)



# Data path
data_path = PATH_TO_VOLUME_DATA

# Visualize the data for Q
visualize_data(data_path + 'q_3D_vol_128_0.npy')

# Visualize the data for B (as an example with xyz components)
visualize_xyz_components(
    data_path + 'Bx_3D_vol_128_0.npy',
    data_path + 'By_3D_vol_128_0.npy',
    data_path + 'Bz_3D_vol_128_0.npy'
)

# Load and print the max values
wannaSee = np.load(data_path + 'Bxyz_max.npy')
print(f'Bxyz_max: {wannaSee}')





















# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# # Import some libraries
# import numpy as np
# import matplotlib.pyplot as plt

# # Define constants
# n_pixels = 10
# data_path = '/global/homes/j/jcurcio/perlmutter/pcnn/Volume_Data/'

# # Load data
# Q = np.load(data_path + 'Q_3D_vol_10_0.npy').reshape([n_pixels, n_pixels, n_pixels])
# # Ex = np.load(data_path + 'Ex_3D_vol_128_0.npy').reshape([n_pixels, n_pixels, n_pixels])
# # Ey = np.load(data_path + 'Ey_3D_vol_128_0.npy').reshape([n_pixels, n_pixels, n_pixels])
# # Ez = np.load(data_path + 'Ez_3D_vol_128_0.npy').reshape([n_pixels, n_pixels, n_pixels])
# # Bx = np.load(data_path + 'Bx_3D_vol_128_0.npy').reshape([n_pixels, n_pixels, n_pixels])
# # By = np.load(data_path + 'By_3D_vol_128_0.npy').reshape([n_pixels, n_pixels, n_pixels])
# # Bz = np.load(data_path + 'Bz_3D_vol_128_0.npy').reshape([n_pixels, n_pixels, n_pixels])
# # Jx = np.load(data_path + 'Jx_3D_vol_128_0.npy').reshape([n_pixels, n_pixels, n_pixels])
# # Jy = np.load(data_path + 'Jy_3D_vol_128_0.npy').reshape([n_pixels, n_pixels, n_pixels])
# # Jz = np.load(data_path + 'Jz_3D_vol_128_0.npy').reshape([n_pixels, n_pixels, n_pixels])

# # DEBUGGING -----------------------------
# # wannaSee = np.load(data_path + 'J_max_max_all_128.npy')
# # print(wannaSee)

# print("Number of non-zero values in Q:", np.count_nonzero(Q))

# # ---------------------------------

# # Axis for plotting
# x_axis = np.linspace(-0.0012, 0.0012, n_pixels)
# y_axis = np.linspace(-0.0012, 0.0012, n_pixels)
# z_axis = np.linspace(-0.0022, 0.0022, n_pixels)

# # Plot slices
# def plot_slices(data, data_name, slice_index):
#     plt.figure(figsize=(12, 10))

#     # xy slice
#     plt.subplot(2, 2, 1)
#     plt.imshow(data[:, :, slice_index], extent=[x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()])
#     plt.title(f'{data_name} xy slice at z={z_axis[slice_index]:.4f}')
#     plt.colorbar()

#     # xz slice
#     plt.subplot(2, 2, 2)
#     plt.imshow(data[:, slice_index, :], extent=[x_axis.min(), x_axis.max(), z_axis.min(), z_axis.max()])
#     plt.title(f'{data_name} xz slice at y={y_axis[slice_index]:.4f}')
#     plt.colorbar()

#     # yz slice
#     plt.subplot(2, 2, 3)
#     plt.imshow(data[slice_index, :, :], extent=[y_axis.min(), y_axis.max(), z_axis.min(), z_axis.max()])
#     plt.title(f'{data_name} yz slice at x={x_axis[slice_index]:.4f}')
#     plt.colorbar()

#     plt.tight_layout()
#     plt.show()

# # Indices for slicing
# slice_index = n_pixels // 2

# # Plotting
# plot_slices(Q, 'Charge Density (Q)', slice_index)
# # plot_slices(Ex, 'Electric Field X (Ex)', slice_index)
# # plot_slices(Ey, 'Electric Field Y (Ey)', slice_index)
# # plot_slices(Ez, 'Electric Field Z (Ez)', slice_index)
# # plot_slices(Bx, 'Magnetic Field X (Bx)', slice_index)
# # plot_slices(By, 'Magnetic Field Y (By)', slice_index)
# # plot_slices(Bz, 'Magnetic Field Z (Bz)', slice_index)
# # plot_slices(Jx, 'Current Density X (Jx)', slice_index)
# # plot_slices(Jy, 'Current Density Y (Jy)', slice_index)
# # plot_slices(Jz, 'Current Density Z (Jz)', slice_index)
