# MNIST Neural Network Analysis

This project focuses on exploring key concepts in neural network design and optimization using the MNIST dataset. The analysis spans multiple aspects of neural network architecture, such as breadth vs. depth trade-offs, effects of activation functions and optimizers, and the comparison of dense layers versus convolutional neural networks (CNNs). Additionally, the project includes experiments with auto-encoders for dimensionality reduction and investigates how architectural choices affect performance metrics like training and testing loss. Each experiment is implemented and documented in a single Jupyter notebook with visualizations to highlight key trends and insights.


## Table of Contents
- [Features](#features)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Requirements](#requirements)
- [File Output](#file-output)
- [License](#license)
- [Important Note](#important-note)


## Features

- **Breadth vs. Depth Trade-Offs**:

  - Quantitative evaluation of network performance with varying layer counts and node distributions

  - Analysis of trends like overfitting, underfitting, and optimal parameter configurations

- **Training Effects**:
 
  - Impact of batch size, optimizers (Adam vs. SGD), and activation functions (ReLU, Tanh, Sigmoid, ELU)

  - Comparisons of training/test loss and convergence speed

- **CNN vs. Dense Layers**:

  - Performance comparison of dense networks versus CNNs with one or two convolutional layers

  - Exploration of optimal CNN architectures for the MNIST dataset

- **Auto-Encoders**:

  - Implementation of encoder-decoder architectures for dimensionality reduction

  - Reconstruction error analysis to estimate the intrinsic dimensionality of MNIST data

- **Bonus Experiments**: 

  - Custom experiments like varying learning rates for different layers and their impact on convergence
    
  - Analysis of classification errors to understand model behavior
  

## Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Kobe-Martinez/building-and-training-networks.git
   cd building-and-training-networks

2. **Open the Jupyter Notebook**:

  - Start Jupyter Notebook or JupyterLab
    
   ```bash
   jupyter notebook
   ```

  - Open the file ```bash mnist_analysis.ipynb ```

3. **Run the Notebook**:

  - Execute cells sequentially to reproduce the experiments and visualizations

4. **Customize Parameters**:

  - Modify hyperparameters (batch size, learning rates, layer sizes, etc.) directly in the notebook cells to explore additional configurations
    
## Code Structure

- **Notebook**:

  - `mnist_analysis.ipynb`: The central notebook containing all data preprocessing, model implementation, and experiment results

- **Utilities**:

  - Helper functions for visualization, metrics computation, and model evaluation are included within the notebook


## Requirements

- **Python**:

  - 3.8 or later

- **Required libraries**:

  - `torch`

  - `numpy`

  - `matplotlib`
 
- **Software**:

  - Jupyter Notebook or JupyterLab for running the `.ipynb` file

- **Install dependencies**:

  - `pip install torch numpy matplotlib`
 

## File Output

- **Generated Outputs**:

  - Plots visualizing training and testing loss trends, hyperparameter effects, and dimensionality reduction performance

  - Model logs and metrics saved within the notebook for easy reference
 

## License

This project is licensed under the MIT License. See the LICENSE file for details.


## Important Note

This code is intended for educational purposes only; Use this repository responsibly.
