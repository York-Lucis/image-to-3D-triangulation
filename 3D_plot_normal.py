import cupy as cp
import cv2
import numpy as np  # Ensure NumPy is imported
import plotly.graph_objects as go
from scipy.spatial import Delaunay

# Load image and convert to grayscale
image_path = 'brazil_old_map.jpg'  # Replace with actual image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is not None and has sufficient size
if img is None or img.size == 0:
    raise ValueError("Image is empty or couldn't be loaded.")

# Normalize the grayscale image to simulate Z-values (move the array to GPU)
Z_gpu = cp.asarray(img.astype(float)) / 255.0

# Get the x, y coordinates (move arrays to GPU)
h, w = Z_gpu.shape
x_gpu = cp.arange(w)
y_gpu = cp.arange(h)
X_gpu, Y_gpu = cp.meshgrid(x_gpu, y_gpu)

# Move data back to CPU for Delaunay triangulation (scipy is CPU-based)
X_cpu = X_gpu.get()  # Convert cupy array to numpy array
Y_cpu = Y_gpu.get()  # Convert cupy array to numpy array
Z_cpu = Z_gpu.get()  # Convert cupy array to numpy array

# Flatten the arrays for triangulation
points2D_cpu = np.vstack([X_cpu.ravel(), Y_cpu.ravel()]).T  # np.vstack ensures this is numpy

# Perform Delaunay triangulation on the CPU using numpy arrays
tri = Delaunay(points2D_cpu)

# Create a Plotly 3D mesh plot
fig = go.Figure(data=[go.Mesh3d(
    x=points2D_cpu[:, 0],
    y=points2D_cpu[:, 1],
    z=Z_cpu.ravel(),
    i=tri.simplices[:, 0],  # Triangle vertex 1
    j=tri.simplices[:, 1],  # Triangle vertex 2
    k=tri.simplices[:, 2],  # Triangle vertex 3
    intensity=Z_cpu.ravel(),  # Color intensity based on Z values
    colorscale='Viridis',
    showscale=True
)])

# Update layout for better visualization
fig.update_layout(scene=dict(
    xaxis_title='X axis',
    yaxis_title='Y axis',
    zaxis_title='Z axis',
    aspectratio=dict(x=1, y=1, z=0.5),
), title="3D Triangular Mesh from Image (CUDA Accelerated)")

# Show the plot
print('3D triangulation is ready.')
fig.show()
