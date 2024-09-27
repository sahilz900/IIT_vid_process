# Face Detection and Annotation

This project implements face detection and landmark annotation on video files using **MTCNN** (Multi-task Cascaded Convolutional Networks) and a **Graph Convolutional Network (GCN)** for refining facial landmarks. The video frames are processed and can be annotated based on specific conditions.

## Files Included

- **gcn.py**: Contains the implementation of the GCN for landmark refinement.
- **main.py**: Main script for processing video files.
- **static_img.py**: Script for processing static images.
- **video_testing.py**: Script for testing video processing with a webcam feed.
- **FaceDetection&Annotation.ipynb**: Jupyter Notebook for interactive face detection and annotation.

## How to Use

1. Clone the repository.
2. Install the required dependencies.
3. Update the video file path in `main.py`.
4. Run the `main.py` script to process your video file.

## Output

Processed videos are saved in MP4 format in the `output` directory.

## Requirements

- Python 3.x
- OpenCV
- MTCNN
- NumPy

## License

This project is licensed under the **MIT License**.

