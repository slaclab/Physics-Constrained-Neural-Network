import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# Constants
PATH_TO_VOLUME_DATA = '/pscratch/sd/j/jcurcio/pcnn/Volume_Data/'
n_pixels = 128
n_predict = 10  # timestep, 0th indexed

# Load data
Jx = np.load(PATH_TO_VOLUME_DATA + f'Jx_3D_vol_{n_pixels}_{n_predict}.npy').reshape([1, n_pixels, n_pixels, n_pixels, 1])
Jy = np.load(PATH_TO_VOLUME_DATA + f'Jy_3D_vol_{n_pixels}_{n_predict}.npy').reshape([1, n_pixels, n_pixels, n_pixels, 1])
Jz = np.load(PATH_TO_VOLUME_DATA + f'Jz_3D_vol_{n_pixels}_{n_predict}.npy').reshape([1, n_pixels, n_pixels, n_pixels, 1])

# Latent space inputs, un-used, all zeros
z_input = np.zeros([1, 8, 8, 8, 1]).astype(np.float32)

# Load model
model_A = load_model('../Trained_Models/PCNN_10.h5')

# Predict Ax
inputs = [Jx, z_input]
Ax_predicted, _ = model_A.predict(inputs)

# Predict Ay
inputs = [Jy, z_input]
Ay_predicted, _ = model_A.predict(inputs)

# Predict Az
inputs = [Jz, z_input]
Az_predicted, _ = model_A.predict(inputs)

# Reshape the predictions to remove the single channel dimension
Ax_predicted = Ax_predicted[0, ..., 0]  # Shape: (128, 128, 128)
Ay_predicted = Ay_predicted[0, ..., 0]  # Shape: (128, 128, 128)
Az_predicted = Az_predicted[0, ..., 0]  # Shape: (128, 128, 128)

# Compute numerical derivatives to get the magnetic fields
def compute_derivatives(A, dx, dy, dz):
    A_y = np.gradient(A, axis=1) / dy
    A_z = np.gradient(A, axis=2) / dz
    A_x = np.gradient(A, axis=0) / dx
    return A_x, A_y, A_z

dx = dy = dz = 1  # Assuming unit spacing for simplicity; adjust if necessary

Ax1_x, Ax1_y, Ax1_z = compute_derivatives(Ax_predicted, dx, dy, dz)
Ay1_x, Ay1_y, Ay1_z = compute_derivatives(Ay_predicted, dx, dy, dz)
Az1_x, Az1_y, Az1_z = compute_derivatives(Az_predicted, dx, dy, dz)

# Magnetic Fields
Bx_predicted = Az1_y - Ay1_z  # Bx = ∂Az/∂y - ∂Ay/∂z
By_predicted = Ax1_z - Az1_x  # By = ∂Ax/∂z - ∂Az/∂x
Bz_predicted = Ay1_x - Ax1_y  # Bz = ∂Ay/∂x - ∂Ax/∂y

# Remove values below the precision threshold
precision_threshold = 0.0001
Bx_predicted = np.where(np.abs(Bx_predicted) < precision_threshold, 0, Bx_predicted)
By_predicted = np.where(np.abs(By_predicted) < precision_threshold, 0, By_predicted)
Bz_predicted = np.where(np.abs(Bz_predicted) < precision_threshold, 0, Bz_predicted)

print(Bx_predicted.dtype)

print('Number of zero elements in Bx: ', len(Bx_predicted.flatten()) - np.count_nonzero(Bx_predicted))
print('Number of zero elements in By: ', len(By_predicted.flatten()) - np.count_nonzero(By_predicted))
print('Number of zero elements in Bz: ', len(Bz_predicted.flatten()) - np.count_nonzero(Bz_predicted))

# PLOTTING
# ------------------------------------------------------------------------------------------------------------

# Compute the magnitude of the magnetic field vectors
B_magnitude_predicted = np.sqrt(Bx_predicted**2 + By_predicted**2 + Bz_predicted**2)

# Create a mask for non-zero magnitudes
mask_predicted = B_magnitude_predicted > 0

# 3D Scatter plot for non-zero values of the magnetic field magnitudes
def plot_magnetic_field_components(Bx, By, Bz, mask, title):
    fig = plt.figure(figsize=(15, 13))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for plotting
    x, y, z = np.meshgrid(np.arange(n_pixels), np.arange(n_pixels), np.arange(n_pixels), indexing='ij')

    # Flatten arrays for plotting
    x_flattened = x.flatten()
    y_flattened = y.flatten()
    z_flattened = z.flatten()
    Bx_flattened = Bx.flatten()
    By_flattened = By.flatten()
    Bz_flattened = Bz.flatten()

    # Get non-zero indices for the magnetic field components
    non_zero_indices = np.nonzero(mask.flatten())

    # Scatter plot of non-zero magnetic field components
    if len(non_zero_indices[0]) > 0:
        sc = ax.scatter(z_flattened[non_zero_indices], x_flattened[non_zero_indices],
                        y_flattened[non_zero_indices], c=Bx_flattened[non_zero_indices],
                        cmap='viridis', s=20, vmin=Bx.min(), vmax=Bx.max(), label='Bx')
        fig.colorbar(sc, label='Bx')
        sc = ax.scatter(z_flattened[non_zero_indices], x_flattened[non_zero_indices],
                        y_flattened[non_zero_indices], c=By_flattened[non_zero_indices],
                        cmap='viridis', s=20, vmin=By.min(), vmax=By.max(), label='By')
        fig.colorbar(sc, label='By')
        sc = ax.scatter(z_flattened[non_zero_indices], x_flattened[non_zero_indices],
                        y_flattened[non_zero_indices], c=Bz_flattened[non_zero_indices],
                        cmap='viridis', s=20, vmin=Bz.min(), vmax=Bz.max(), label='Bz')
        fig.colorbar(sc, label='Bz')
    else:
        # Add a placeholder plot for all-zero data
        ax.text2D(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Set axis limits
    ax.set_xlim(0, n_pixels)
    ax.set_ylim(0, n_pixels)
    ax.set_zlim(0, n_pixels)

    # Customize plot appearance
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_title(title)

    # Adjust viewing angle
    ax.view_init(elev=30, azim=-60)

    # Format the tick labels to reduce precision
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
    ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))

    plt.show()


# 2D Heatmap for a slice in the z direction
def plot_component_slice(B_component, slice_index, title, component_name):
    fig, ax = plt.subplots(figsize=(15, 13))

    # Plot the heatmap for the specified z slice
    c = ax.imshow(B_component[:, :, slice_index], cmap='viridis', origin='lower',
                  extent=(0, n_pixels, 0, n_pixels), aspect='auto')
    fig.colorbar(c, label=f'{component_name} Value')

    # Customize plot appearance
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title} (Z = {slice_index})')

    plt.show()


# Plot the predicted magnetic field components in a scatter plot
plot_magnetic_field_components(Bx_predicted, By_predicted, Bz_predicted, mask_predicted, 'Predicted Non-Zero Magnetic Field Components')

# Plot a slice in the z direction (middle slice) for each component
middle_slice_index = n_pixels // 2
plot_component_slice(Bx_predicted, middle_slice_index, 'Predicted Bx Component Slice in Z Direction', 'Bx')
plot_component_slice(By_predicted, middle_slice_index, 'Predicted By Component Slice in Z Direction', 'By')
plot_component_slice(Bz_predicted, middle_slice_index, 'Predicted Bz Component Slice in Z Direction', 'Bz')



# Load the original magnetic field data components
Bx_original = np.load(PATH_TO_VOLUME_DATA + 'original_Bx_data.npy')
By_original = np.load(PATH_TO_VOLUME_DATA + 'original_By_data.npy')
Bz_original = np.load(PATH_TO_VOLUME_DATA + 'original_Bz_data.npy')

# Compute the magnitude of the original magnetic field vectors
B_magnitude_original = np.sqrt(Bx_original**2 + By_original**2 + Bz_original**2)

# Create a mask for non-zero magnitudes for original data
mask_original = B_magnitude_original > 0

# Plot the original magnetic field magnitudes in a scatter plot
plot_magnetic_field(Bx_original, By_original, Bz_original, B_magnitude_original, mask_original, 'Original Non-Zero Magnetic Field Magnitude')

# Plot a slice in the z direction (z = 64) for original data
z_slice_index = 64
plot_slice(B_magnitude_original, z_slice_index, 'Original Magnetic Field Magnitude Slice at Z = 64')


print("Predicted magnetic field components for the 11th timestep:")
print("Bx:", Bx_predicted)
print("By:", By_predicted)
print("Bz:", Bz_predicted)