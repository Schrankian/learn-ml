/// This module handles data loading, preprocessing, and augmentation.
/// It provides utilities for managing datasets and preparing data for machine learning models.
/// How to use:
/// ```
/// use machine_learning::data_loader::DataLoader;
///
/// fn main() {
///     let train_dataset = DataLoader::from_dir("/path/to/idx-ubyte/files", "train", 1);
///
///     println!("Training data: {:?}", train_dataset);
/// }
/// ```
pub mod data_loader;

/// This module implements a matrix structure which uses automatic differentiation to calculate gradients.
/// It is NOT a general purpose matrix library, but a specialized one for machine learning.
/// How to use:
/// ```
/// use machine_learning::autograd_matrix::Matrix;
/// use ndarray::array;
///
/// fn main() {
///     let a = Matrix::from(array![[1.0, 2.0], [3.0, 4.0]]);
///     let b = Matrix::from(array![[5.0, 6.0], [7.0, 8.0]]);
///
///     let c = a.dot(&b).relu();
///
///     println!("Matrix C: {:?}", c);
/// }
/// ```
pub mod autograd_matrix;

/// This module is a simple wrapper around all things related to neural networks. It provides a simple API
/// for defining, training and using neural networks. It also provides wrappers around Loss and Optimizer functions.
/// And has predefined layers like Linear, ReLU, etc. (But also allows custom layers).
/// How to use:
/// ```
/// use machine_learning::neural_network::Model;
///
/// fn main() {
///     let mut model = Model::new(
///         vec![
///             Layer::linear((784, 784)),
///             Layer::relu(),
///             Layer::linear((784, 784)),
///             Layer::relu(),
///             Layer::linear((784, 10)),
///             Layer::relu(),
///         ],
///         Some(Loss::MSE),
///         Some(Optimizer::SGD),
///         Some(Meta {
///             num_epochs: 0,
///             learning_rate: 1e-3,
///             batch_size: 64,
///         }),
///     ),
/// 	let predictions = model.predict(Array2::zeros((1, 784)));
///     println!("Predictions: {:?}", predictions);
/// }
/// ```
pub mod neural_network;
