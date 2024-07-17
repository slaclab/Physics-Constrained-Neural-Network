#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importing 3D plotting capabilities

# Paths to your saved .npy files
PATH_TO_VOLUME_DATA = '/global/homes/j/jcurcio/perlmutter/pcnn/Volume_Data/'

# Function to load and visualize 3D data
def visualize_data(filename):
    data = np.load(filename)
    print(f"Loaded {filename}: Shape {data.shape}")

    # Assuming data is structured as [128x128x128]x3
    n_arrays = data.shape[0]
    n_pixels = data.shape[1]

    # Plot each array separately
    fig, axs = plt.subplots(1, n_arrays, figsize=(15, 6))

    for i in range(n_arrays):
        ax = axs[i]

        # Extract data for the current array
        array_data = data[i]

        # Plot a 2D slice along the z-axis (middle slice)
        slice_index = n_pixels // 2
        slice_data = array_data[:, :, slice_index]

        # Plotting the slice with adjusted colormap and scaling
        img = ax.imshow(slice_data, cmap='viridis', vmin=slice_data.min(), vmax=slice_data.max())
        ax.set_title(f'Array {i+1}')
        fig.colorbar(img, ax=ax)

    plt.suptitle(f'2D Slice Visualization of {filename}')
    plt.tight_layout()
    plt.show()


data_path = '/global/homes/j/jcurcio/perlmutter/pcnn/Volume_Data/'

visualize_data(data_path + 'Q_3D_vol_128_0.npy')

wannaSee = np.load(data_path + 'Bxyz_max.npy')
print(wannaSee)


















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
