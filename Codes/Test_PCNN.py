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

# Remove values within the precision range threshold
precision_threshold_low = -0.8e8
precision_threshold_high = 0.8e8
Bx_predicted = np.where((Bx_predicted > precision_threshold_low) & (Bx_predicted < precision_threshold_high), 0, Bx_predicted)
By_predicted = np.where((By_predicted > precision_threshold_low) & (By_predicted < precision_threshold_high), 0, By_predicted)
Bz_predicted = np.where((Bz_predicted > precision_threshold_low) & (Bz_predicted < precision_threshold_high), 0, Bz_predicted)

print(Bx_predicted.dtype)

print('Number of zero elements in Bx: ', len(Bx_predicted.flatten()) - np.count_nonzero(Bx_predicted))
print('Number of zero elements in By: ', len(By_predicted.flatten()) - np.count_nonzero(By_predicted))
print('Number of zero elements in Bz: ', len(Bz_predicted.flatten()) - np.count_nonzero(Bz_predicted))

# PLOTTING
# ------------------------------------------------------------------------------------------------------------

# 3D PLOTTING

# Generate coordinate arrays
n_pixels = 128
x = np.linspace(0, 1, n_pixels)
y = np.linspace(0, 1, n_pixels)
z = np.linspace(0, 1, n_pixels)

# Create meshgrid for coordinates
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Apply the threshold
mask = np.abs(Bx_predicted) >= precision_threshold_high
X_filtered = X[mask]
Y_filtered = Y[mask]
Z_filtered = Z[mask]
Bx_filtered = Bx_predicted[mask]

# Get the original min and max values for the colormap
Bx_min = Bx_predicted.min()
Bx_max = Bx_predicted.max()

# Plot the 3D scatter plot
fig = plt.figure(figsize=(15, 13))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(Z_filtered, X_filtered, Y_filtered, c=Bx_filtered, cmap='viridis', s=20, vmin=Bx_min, vmax=Bx_max)
fig.colorbar(sc, label='B')

ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('y')
plt.title('3D Scatter Plot of Predicted Magnetic Field Bx (Filtered)')

plt.show()

# 2D SLICE PLOTTING

# Slice the data for a specific z-coordinate
z_slice_index = int(n_pixels / 2)
x_slice = X[:, :, z_slice_index]
y_slice = Y[:, :, z_slice_index]
Bx_slice = Bx_predicted[:, :, z_slice_index]

# Apply the threshold to the slice
mask_slice = np.abs(Bx_slice) >= precision_threshold_high
x_slice_filtered = x_slice[mask_slice]
y_slice_filtered = y_slice[mask_slice]
Bx_slice_filtered = Bx_slice[mask_slice]

# Plot the 2D scatter plot
fig = plt.figure(figsize=(15, 13))
ax = fig.add_subplot(111)

sc = ax.scatter(x_slice_filtered, y_slice_filtered, c=Bx_slice_filtered, cmap='viridis', s=20, vmin=Bx_min, vmax=Bx_max)
fig.colorbar(sc, label='B')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Slice at z = {z[z_slice_index]:.2f} (Filtered)')

plt.show()
