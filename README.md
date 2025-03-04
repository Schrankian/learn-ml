# Own Neural Network implementation
This repository contains code I wrote to learn how neural networks work, using the MNIST image dataset for handwritten number recognition. 
I implemented everything from scratch, except for efficient matrix multiplication.

## Contents

### Data
I used PyTorch to download and split the MNIST dataset for training and evaluation, storing them in binary format. 
Methods to parse and load these images are provided. Additionally, I created two images with the same size as the training images, 
stored under `data/own/`, to test the model's ability to predict my handwriting.

### Inspiration
The `inspiration` directory contains a Jupyter notebook outlining the general implementation plan and serving as a reference for the steps followed.

### Models
The `models` directory stores the trained models. The current model, `model.bin`, has around 83% accuracy on the test dataset, which is satisfactory for a first attempt.

### Src
The `src` directory contains the neural network implementation, divided into three main parts:
- `autograd_matrix.rs`: Implements a custom Matrix type with automatic differentiation for efficient gradient calculation.
- `data_loader.rs`: Handles data loading (MNIST data, PNGs) and conversion into the model's requirements, providing batching logic.
- `neural_network.rs`: Contains wrappers around existing functionality to create an abstraction layer, keeping the `main.rs` file clean and readable.

### Prerequisites
- [OpenBLAS](http://www.openmathlib.org/OpenBLAS/docs/install/) installed on the system for efficient matrix multiplication.

### Dependencies
- `ndarray`: Base for the Matrix datatype.
- `blas-src`: Provides BLAS bindings to `ndarray`.
- `openblas-src`: Uses the system's BLAS library.
- `byteorder`: For loading the MNIST dataset correctly.
- `image`: Simplifies loading and converting images to grayscale.
- `rand`: Random number generator.
- `rand_distr`: Provides probability distributions for weight initialization.
- `num-traits`: Specifies a general "Float" type.
- `clap`: Command line parser.

### Run
The program can be started via Cargo, requiring some library linking. Use the following commands:
- `cargo run --release -- --help`: Show help menu.
- `cargo run --release -- train`: Train the model and save it to disk.
	- `--data-path <..>`: Specify custom training data path.
	- `--model-path <..>`: Specify custom model save location.
	- `--continue-training`: Continue training from an existing model.
	- `--epochs <..>`: Specify the number of training epochs.
- `cargo run --release -- eval`: Evaluate the model on the test dataset.
	- `--data-path <..>`: Specify custom test data path.
	- `--model-path <..>`: Specify model location.
- `cargo run --release -- normal`: Predict custom images.
	- `--model-path <..>`: Specify model location.
	- `--image-path <..>`: Specify image to load (must be a 28x28 PNG).

### Tests
Unit tests are defined for each module and can be run with `cargo test`. A PowerShell script, `./coverage.ps1`, generates a coverage report and opens it in the browser.

### Detailed Explanation
TODO

### Efficiency
Designed to work on the CPU, tested on an AMD Ryzen 9 7900X processor, achieving 35 seconds per epoch with a batch size of 64 and training size of 60,000. 
Ensure the `--release` flag is used for optimal performance.
