# ImageTo3DTriangulation

A Python application that converts 2D images into interactive 3D triangular mesh visualizations using CUDA-accelerated processing and Delaunay triangulation.

## Project Description

ImageTo3DTriangulation is a powerful visualization tool that transforms 2D images into stunning 3D triangular mesh representations. The application leverages CUDA acceleration through CuPy for high-performance image processing and uses Delaunay triangulation to create smooth, interactive 3D surfaces from grayscale images.

### Key Features

- **CUDA Acceleration**: High-performance image processing using GPU computing
- **Delaunay Triangulation**: Creates smooth, mathematically optimal triangular meshes
- **Interactive 3D Visualization**: Plotly-based interactive 3D plots with zoom, rotate, and pan capabilities
- **Multiple Image Formats**: Supports various image formats through OpenCV
- **Automatic Normalization**: Converts images to appropriate height values for 3D representation
- **High-Quality Rendering**: Professional-grade 3D visualization with customizable color schemes

## Installation Instructions

### Prerequisites

- Python 3.7 or higher
- NVIDIA GPU with CUDA support (for CuPy acceleration)
- CUDA Toolkit installed on your system

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/York-Lucis/image-to-3D-triangulation.git
   cd image-to-3D-triangulation
   ```

2. **Install CUDA dependencies**:
   ```bash
   # Install CuPy (adjust version for your CUDA version)
   pip install cupy-cuda11x  # For CUDA 11.x
   # or
   pip install cupy-cuda12x  # For CUDA 12.x
   ```

3. **Install other dependencies**:
   ```bash
   pip install opencv-python numpy plotly scipy
   ```

### Dependencies

- `cupy` - CUDA-accelerated NumPy-compatible array library
- `opencv-python` - Computer vision library for image processing
- `numpy` - Numerical computing library
- `plotly` - Interactive plotting library
- `scipy` - Scientific computing library (for Delaunay triangulation)

## Usage Guide

### Basic Usage

1. **Prepare your image**:
   - Place your image file in the project directory
   - Supported formats: JPG, PNG, BMP, TIFF, etc.

2. **Modify the image path**:
   ```python
   # Edit the image_path variable in 3D_plot_normal.py
   image_path = 'your_image.jpg'  # Replace with your image filename
   ```

3. **Run the application**:
   ```bash
python 3D_plot_normal.py
   ```

4. **Interact with the 3D visualization**:
   - Use mouse to rotate, zoom, and pan the 3D model
   - Hover over points to see coordinate information
   - Use the color scale to understand height values

### Example with Different Images

```python
# For a landscape photo
image_path = 'mountain_landscape.jpg'

# For a portrait
image_path = 'portrait_photo.png'

# For a map
image_path = 'topographic_map.jpg'
```

### Customization Options

#### Color Schemes
The application uses the 'Viridis' color scheme by default. You can modify this in the code:

```python
# Change color scheme in the Mesh3d configuration
colorscale='Viridis'  # Options: 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Jet'
```

#### Image Processing Parameters
```python
# Adjust normalization factor
Z_gpu = cp.asarray(img.astype(float)) / 255.0  # Standard normalization
# or
Z_gpu = cp.asarray(img.astype(float)) / 128.0  # Enhanced contrast
```

## Technical Details

### Processing Pipeline

1. **Image Loading**: Load image using OpenCV in grayscale mode
2. **GPU Transfer**: Convert image to CuPy array for GPU processing
3. **Normalization**: Normalize pixel values to 0-1 range for height mapping
4. **Coordinate Generation**: Create X, Y coordinate grids
5. **CPU Transfer**: Move data back to CPU for triangulation
6. **Delaunay Triangulation**: Create triangular mesh using scipy
7. **3D Visualization**: Generate interactive Plotly visualization

### CUDA Acceleration

The application uses CuPy for GPU-accelerated operations:
- **Memory Management**: Efficient GPU memory allocation and deallocation
- **Array Operations**: Fast mathematical operations on GPU
- **Data Transfer**: Optimized CPU-GPU data transfer

### Delaunay Triangulation

The Delaunay triangulation algorithm creates optimal triangular meshes by:
- Maximizing minimum angles in triangles
- Avoiding thin, elongated triangles
- Creating smooth surface representations
- Ensuring mathematical optimality

## Performance Considerations

### GPU Requirements
- **Minimum**: NVIDIA GPU with 2GB VRAM
- **Recommended**: NVIDIA GPU with 4GB+ VRAM
- **CUDA Version**: Compatible with CUDA 11.x or 12.x

### Image Size Limits
- **Small Images** (< 1000x1000): Fast processing, real-time visualization
- **Medium Images** (1000x1000 - 3000x3000): Good performance with some delay
- **Large Images** (> 3000x3000): May require significant processing time and memory

### Memory Usage
- GPU memory usage scales with image size
- CPU memory usage for triangulation can be substantial for large images
- Consider image resizing for very large files

## Development

### Architecture

The application follows a linear processing pipeline:

1. **Image Processing Module**: Handles image loading and preprocessing
2. **GPU Processing Module**: Manages CUDA operations and memory
3. **Triangulation Module**: Performs Delaunay triangulation
4. **Visualization Module**: Creates interactive 3D plots

### Key Functions

- **Image Loading**: `cv2.imread()` with grayscale conversion
- **GPU Processing**: CuPy array operations for normalization
- **Triangulation**: `scipy.spatial.Delaunay()` for mesh generation
- **Visualization**: `plotly.graph_objects.Mesh3d()` for 3D rendering

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with various image types and sizes
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure CUDA Toolkit is installed and CuPy is compatible
2. **Out of memory**: Reduce image size or use a GPU with more VRAM
3. **Slow performance**: Check CUDA installation and GPU utilization
4. **Import errors**: Verify all dependencies are installed correctly

### Performance Optimization

- Use images with power-of-2 dimensions when possible
- Consider image preprocessing to reduce noise
- Close other GPU-intensive applications during processing

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/York-Lucis/image-to-3D-triangulation) or create an issue.

---

**Author**: [York-Lucis](https://github.com/York-Lucis)  
**Repository**: [image-to-3D-triangulation](https://github.com/York-Lucis/image-to-3D-triangulation)