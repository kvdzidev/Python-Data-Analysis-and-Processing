# Python-Data-Analysis-and-Processing
Project: Collection of Scripts for Data Analysis and Processing
Description
This repository contains three Python scripts implementing various algorithms and functions for data analysis, visualization, and machine learning.

Files
1. astar_test.py - A* Pathfinding Algorithm Implementation
This script implements the A* algorithm to find the shortest path between two points on a map. The user can select one of three heuristics:

Manhattan Metric
Euclidean Metric
Random Metric
The script loads a map image, identifies the start and goal points, and then draws the computed path on the map.

Use Cases:

Grid-based navigation
AI and game pathfinding algorithms
Route optimization
2. test_function_plot.py - Mathematical Function Visualization
This script generates 3D plots for various optimization functions, such as:

Sphere
Rosenbrock
Beale
Matyas
Rastrigin
Easom
It can be executed with a command-line argument specifying the function name.

Example Usage:

bash
Kopiuj
Edytuj
python test_function_plot.py sphere
The script creates a 3D surface visualization of the selected function in the range (-5, 5) for both variables.

Use Cases:

Visualization of optimization function landscapes
Analysis and comparison of test functions
3. autoenkoder_mnist.py - Autoencoder for MNIST Image Compression
This script implements an autoencoder neural network to reconstruct images from the MNIST dataset. It consists of two parts:

Encoder: compresses images into a lower-dimensional representation
Decoder: reconstructs the image from the encoded version
The model is trained for 10 epochs and then generates examples of original and reconstructed images.

Use Cases:

Image compression and reconstruction
Preprocessing data for deep learning
Noise reduction in images
Requirements
To run the scripts, install the required libraries:
pip install numpy matplotlib torch torchvision pillow
Author
Repository created by: kvdzidev
