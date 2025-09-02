# Pothole Detection System

A comprehensive system for detecting and analyzing potholes in road surfaces using Intel RealSense cameras and deep learning.

## Features

- **Data Extraction**: Process Intel RealSense .bag files to extract RGB and depth images
- **Deep Learning Segmentation**: Use trained SegFormer model for accurate pothole detection
- **3D Analysis**: Calculate pothole dimensions and depth using point cloud processing
- **Tracking**: Group similar detections across video frames
- **Reporting**: Generate detailed CSV reports with measurements

## Project Structure

```
src/
├── data_extraction/     # .bag file processing
├── segmentation/        # Model loading and inference
├── analysis/           # Pothole measurement and analysis
├── tracking/           # Cross-frame pothole grouping
└── utils/              # Configuration and file management
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements_new.txt
   ```
3. Ensure you have the trained model file in the project root

## Usage

Run the main script:
```bash
python src/main.py
```

The GUI will prompt you to:
1. Select the folder containing .bag files
2. Choose an output directory for results

## Configuration

Modify `src/utils/config.py` to adjust:
- Model path and thresholds
- Processing parameters
- Tracking tolerances

## Dependencies

- PyTorch for deep learning
- Open3D for 3D processing
- OpenCV for image processing
- NumPy for numerical operations
- Intel RealSense SDK for camera data

## Output

The system generates:
- Extracted images in organized folders
- Annotated images with pothole contours
- CSV report with measurements
- Point cloud data (optional)

## Architecture

The system follows a modular architecture with clear separation of concerns:
- **Data Extraction**: Handles raw sensor data processing
- **Segmentation**: Applies ML model for detection
- **Analysis**: Performs geometric calculations
- **Tracking**: Manages temporal consistency
- **Utils**: Provides shared utilities and configuration
