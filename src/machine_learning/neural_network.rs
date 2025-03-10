#![allow(dead_code)]
use crate::Matrix;
use ndarray::Array2;
use std::{
    fs::File,
    io::{Read, Write},
};

#[allow(non_snake_case)]
pub mod Optimizer {
    use crate::Layer;

    pub type Signature = fn(f32, &mut Vec<Layer>);

    /// Perform a gradient descent step on all weights in the model
    /// Does nothing if no weights are present
    /// To be honest, this is not true sgd, as it does support batch training (But still shuffles the dataset if used with the DataLoader)
    /// Something like Mini-batch gradient descent (MBGD. Batch size = ~64) or Full-batch gradient descent (FBGD. Batch size = full dataset) would be more accurate
    /// However, the term SGD is often used in literature to describe all of these
    pub const SGD: Signature = |lr, layers| {
        // The gradient descent call-order is currently as follows:
        // 1. This optimizer function is called
        // 2. It calls the grad_desc function of each layer
        // 3. Each layer calls the grad_desc function of its Matrix weights and bias
        // -> So the actual gradient descent is done in the Matrix struct, but the learning rate is passed from here
        layers.iter_mut().for_each(|layer| layer.grad_desc(lr));
    };
}

#[allow(non_snake_case)]
pub mod Loss {
    use crate::machine_learning::autograd_matrix::Matrix;
    use ndarray::Array2;

    pub type Signature = fn(Matrix, &Array2<f32>) -> Matrix;

    /// Mean Squared Error loss function
    /// Calculates the mean squared error between the result and the target
    pub const MSE: Signature = |result, target| result.mse(target);
}

/// Identifiers for saving and loading models to disk
#[derive(Debug, PartialEq)]
enum LayerType {
    Custom,
    Linear,
    ReLU,
}

pub struct Layer {
    // Weights and bias are optional to allow for layers without weights
    weights: Option<Matrix>,
    bias: Option<Matrix>,
    // The function to be called when the layer is activated
    act_fn: fn(Matrix, &Option<Matrix>, &Option<Matrix>) -> Matrix,
    // The type of the layer
    l_type: LayerType,
}

impl Layer {
    /// Create a new custom layer
    pub fn new(
        weights: Option<Matrix>,
        bias: Option<Matrix>,
        act_fn: fn(Matrix, &Option<Matrix>, &Option<Matrix>) -> Matrix,
    ) -> Self {
        Self {
            weights,
            bias,
            act_fn,
            l_type: LayerType::Custom,
        }
    }

    /// Create a mlp layer with predefined weights
    pub fn linear_predef_weights(weight_override: Matrix, bias_override: Matrix) -> Self {
        Self {
            weights: Some(weight_override),
            bias: Some(bias_override),
            act_fn: |x, weights, bias| x.dot(weights.as_ref().unwrap()) + bias.as_ref().unwrap(),
            l_type: LayerType::Linear,
        }
    }

    /// Create a linear layer with random weights (normal distribution) and zero'ed bias
    pub fn linear(dimension: (usize, usize)) -> Self {
        Layer::linear_predef_weights(
            Matrix::randn(dimension),
            Matrix::from(Array2::zeros((1, dimension.1))),
        )
    }

    /// Create a ReLU layer
    pub fn relu() -> Self {
        Self {
            weights: None,
            bias: None,
            act_fn: |x, _, _| x.relu(),
            l_type: LayerType::ReLU,
        }
    }

    /// Calls the function of the layer
    /// Maybe remove this, as it is not particularly useful
    fn act(&self, x: Matrix) -> Matrix {
        (self.act_fn)(x, &self.weights, &self.bias)
    }

    /// Perform an gradient descent step on the layers weights
    /// Does nothing if no weights are present
    fn grad_desc(&mut self, lr: f32) {
        if let Some(weights) = &mut self.weights {
            weights.grad_desc(lr);
        }
        if let Some(bias) = &mut self.bias {
            bias.grad_desc(lr);
        }
    }
}

/// Metadata for the model
/// Currently only used to count the trained epochs
#[derive(PartialEq, Debug)]
pub struct Meta {
    pub num_epochs: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
}

pub struct Model {
    layers: Vec<Layer>,
    last_node: Option<Matrix>,
    loss_fn: Option<Loss::Signature>,
    optimizer: Option<Optimizer::Signature>,
    pub meta: Option<Meta>,
}

impl Model {
    /// Initialize a new model
    pub fn new(
        layers: Vec<Layer>,
        loss_fn: Option<Loss::Signature>,
        optimizer: Option<Optimizer::Signature>,
        meta: Option<Meta>,
    ) -> Self {
        Self {
            layers,
            last_node: None,
            loss_fn,
            optimizer,
            meta,
        }
    }

    /// Get the number of layers in the model
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Perform a forward pass through the model to be used for training
    /// Only used from the train function
    fn forward(&self, x: Array2<f32>) -> Matrix {
        let input = Matrix::from(x);

        let result = self.layers.iter().fold(input, |acc, layer| layer.act(acc));
        result
    }

    /// Calls the optimizer function with the learning rate and the layers
    /// Only used from the train function
    fn optimize(&mut self, _lr: f32) {
        (self.optimizer.expect("No optimizer defined"))(_lr, &mut self.layers);
    }

    /// Resets the gradients of all layers
    /// Only used from the train function
    fn zero_grad(&self) {
        if let Some(node) = &self.last_node {
            node.reset_grad();
        } else {
            panic!("No forward pass has been made yet");
        }
    }

    /// Calculates the loss of the last forward pass
    /// Only used from the train function
    fn loss(&self, target: &Array2<f32>) -> Matrix {
        if let Some(node) = &self.last_node {
            (self.loss_fn.expect("No loss function defined"))(node.clone(), target)
        } else {
            panic!("The last node is not set. Did you forget to set it with the result of the forward pass?");
        }
    }

    /// Does a single forward pass and returns the prediction as a softmaxed f32 array
    pub fn predict(&self, x: Array2<f32>) -> Array2<f32> {
        let result = self.forward(x);
        result.softmax().value() // TODO maybe reconsider returning the cloned data Array2 here instead of Matrix
    }

    /// Trains the model with the given input, target and learning_rate
    /// Performs the following operations:<br>
    /// 1. Forward pass<br>
    /// 2. Loss calculation<br>
    /// 3. Backward pass<br>
    /// 4. Optimize the weights<br>
    /// 5. Reset the gradients<br>
    /// Returns the loss of the current batch
    /// It supports batch training, but does NOT accumulate the gradients
    /// - If you want to benchmark this function, just uncomment the Benchmarker and the break_points
    /// it should print the time it took for each step of a single training iteration
    pub fn train(&mut self, input: Array2<f32>, target: &Array2<f32>, lr: f32) -> Array2<f32> {
        if self.loss_fn.is_none() || self.optimizer.is_none() {
            panic!("Loss function or optimizer not defined");
        }

        // let mut bencher = Benchmarker::new();

        // Do a forward pass and set the result as the last node to keep the following functions tidy
        // This is just personal preference
        self.last_node = Some(self.forward(input));
        // bencher.break_point("Forward pass");

        // Loss calculation
        let loss = self.loss(target);
        // bencher.break_point("Loss calculation");

        // Calculate the gradient
        loss.backward(None);
        // bencher.break_point("Backward pass");

        // Optimize the weights
        self.optimize(lr);
        // bencher.break_point("Optimize");

        // Reset the gradients
        self.zero_grad();
        // bencher.break_point("Zero grad");

        // Return the loss of the current batch
        // bencher.print();
        loss.value()
    }

    /// Save the model to a file in binary format.
    /// This saves the layers, including the weights and bias but not the loss and optimizer! <br>
    /// Saving custom layers is not supported yet
    pub fn save(&self, path: &str) {
        let mut file = std::fs::File::create(path).unwrap();
        let mut buffer = Vec::new();

        // Save metadata if exists
        if let Some(meta) = &self.meta {
            buffer.push(1); // Metadata exists
            buffer.extend_from_slice(&(meta.num_epochs as u16).to_le_bytes());
            buffer.extend_from_slice(&meta.learning_rate.to_le_bytes());
            buffer.extend_from_slice(&(meta.batch_size as u16).to_le_bytes());
        } else {
            buffer.push(0); // No metadata
        }

        // Save the layers
        for layer in &self.layers {
            match layer.l_type {
                LayerType::Custom => {
                    panic!("Doesn't support saving custom layers yet");
                }
                LayerType::Linear => {
                    buffer.push(LayerType::Linear as u8);
                    // Store weights as f32
                    let weights = layer.weights.as_ref().unwrap().value();
                    let (rows, cols) = weights.dim();
                    buffer.extend_from_slice(&(rows as u16).to_le_bytes());
                    buffer.extend_from_slice(&(cols as u16).to_le_bytes());
                    for row in 0..rows {
                        for col in 0..cols {
                            buffer.extend_from_slice(&weights[[row, col]].to_le_bytes());
                        }
                    }
                    // Store bias as f32
                    let bias = layer.bias.as_ref().unwrap().value();
                    let (rows, cols) = bias.dim();
                    buffer.extend_from_slice(&(rows as u16).to_le_bytes());
                    buffer.extend_from_slice(&(cols as u16).to_le_bytes());
                    for row in 0..rows {
                        for col in 0..cols {
                            buffer.extend_from_slice(&bias[[row, col]].to_le_bytes());
                        }
                    }
                }
                LayerType::ReLU => {
                    buffer.push(LayerType::ReLU as u8);
                }
            }
        }
        file.write_all(&buffer).unwrap();
    }

    /// Load a model from a file in binary format.
    /// This loads the layers, including the weights but not the loss and optimizer! <br>
    /// Loading custom layers is not supported yet
    pub fn load(
        path: &str,
        loss_fn: Option<Loss::Signature>,
        optimizer: Option<Optimizer::Signature>,
    ) -> Self {
        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::new();

        file.read_to_end(&mut buffer).unwrap();

        let mut layers = Vec::new();
        let mut index = 0;

        // Load metadata if exists
        let mut meta = None;
        if buffer[index] == 1 {
            index += 1;
            let num_epochs = u16::from_le_bytes([buffer[index], buffer[index + 1]]) as usize;
            index += 2;
            let learning_rate = f32::from_le_bytes([
                buffer[index],
                buffer[index + 1],
                buffer[index + 2],
                buffer[index + 3],
            ]);
            index += 4;
            let batch_size = u16::from_le_bytes([buffer[index], buffer[index + 1]]) as usize;
            index += 2;
            meta = Some(Meta {
                num_epochs,
                learning_rate,
                batch_size,
            });
        } else {
            index += 1;
        }

        while index < buffer.len() {
            match buffer[index] {
                x if x == LayerType::Custom as u8 => {
                    panic!("Why is this here??. This should not have been saved!");
                }
                x if x == LayerType::Linear as u8 => {
                    index += 1; // Read the layer type
                                // Load the weights
                    let rows = u16::from_le_bytes([buffer[index], buffer[index + 1]]) as usize;
                    index += 2;
                    let cols = u16::from_le_bytes([buffer[index], buffer[index + 1]]) as usize;
                    index += 2;
                    let mut weights = Array2::<f32>::zeros((rows, cols));
                    for row in 0..rows {
                        for col in 0..cols {
                            let weight_bytes = [
                                buffer[index],
                                buffer[index + 1],
                                buffer[index + 2],
                                buffer[index + 3],
                            ];
                            weights[[row, col]] = f32::from_le_bytes(weight_bytes);
                            index += 4;
                        }
                    }
                    // Load the bias
                    let rows = u16::from_le_bytes([buffer[index], buffer[index + 1]]) as usize;
                    index += 2;
                    let cols = u16::from_le_bytes([buffer[index], buffer[index + 1]]) as usize;
                    index += 2;
                    let mut bias = Array2::<f32>::zeros((rows, cols));
                    for row in 0..rows {
                        for col in 0..cols {
                            let bias_bytes = [
                                buffer[index],
                                buffer[index + 1],
                                buffer[index + 2],
                                buffer[index + 3],
                            ];
                            bias[[row, col]] = f32::from_le_bytes(bias_bytes);
                            index += 4;
                        }
                    }
                    layers.push(Layer::linear_predef_weights(
                        Matrix::from(weights),
                        Matrix::from(bias),
                    ));
                }
                x if x == LayerType::ReLU as u8 => {
                    index += 1; // Read the layer type
                    layers.push(Layer::relu());
                }
                _ => {
                    panic!(
                        "Unknown layer type: {}, index: {}/{}",
                        buffer[index],
                        index,
                        buffer.len()
                    );
                }
            }
        }

        Self {
            layers,
            last_node: None,
            loss_fn,
            optimizer,
            meta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq, Matrix};
    use ndarray::array;

    fn get_model() -> Model {
        Model::new(
            vec![
                // Layer::linear((3, 3)), but with custom weights for testing
                Layer::linear_predef_weights(
                    Matrix::from(array![[0.1, 0.2, 0.3], [0.2, -0.3, 0.4], [0.3, -0.4, 0.5]]),
                    Matrix::from(array![[0.0, 0.0, 0.0]]),
                ),
                Layer::relu(),
                // Layer::linear((3, 2)), but with custom weights for testing
                Layer::linear_predef_weights(
                    Matrix::from(array![[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]),
                    Matrix::from(array![[0.0, 0.0]]),
                ),
            ],
            Some(Loss::MSE),
            Some(Optimizer::SGD),
            None,
        )
    }

    #[test]
    fn test_model_creation() {
        let model = get_model();
        assert_eq!(model.layers.len(), 3);
    }

    #[test]
    fn test_forward() {
        let model = get_model();
        // Forward pass
        let input = array![[0.0, 1.0, 0.5]];
        let output = model.forward(input);
        assert_approx_eq!(output.value(), array![[0.23, 0.33]], 1e-6, array);
    }

    #[test]
    fn test_optimize() {
        let mut model = get_model();
        // Forward pass
        let input = array![[0.0, 1.0, 0.5]];
        let output = model.forward(input);
        model.last_node = Some(output.clone());
        // Loss & Backward pass
        let loss = output.mse(&array![[0.0, 0.0]]);
        loss.backward(None);
        // Optimize
        model.optimize(0.1);

        assert_approx_eq!(
            model.layers[0].weights.as_ref().unwrap().value(),
            array![
                [0.1, 0.2, 0.3],
                [0.1911, -0.3, 0.3799],
                [0.29555, -0.4, 0.48995]
            ],
            1e-6,
            array
        );
        assert_approx_eq!(
            model.layers[2].weights.as_ref().unwrap().value(),
            array![[0.09195, 0.18845], [0.2, 0.3], [0.28505, 0.37855]],
            1e-6,
            array
        );
    }

    #[test]
    fn test_zero_grad() {
        let mut model = get_model();
        // Forward pass
        let input = array![[0.0, 1.0, 0.5]];
        let output = model.forward(input);
        model.last_node = Some(output.clone());
        // Loss & Backward pass
        let loss = output.mse(&array![[0.0, 0.0]]);
        loss.backward(None);
        // Rest grad
        model.zero_grad();
        // Check if gradients are zero
        assert_eq!(model.layers[0].weights.as_ref().unwrap().grad().sum(), 0.0);
        assert_eq!(model.layers[2].weights.as_ref().unwrap().grad().sum(), 0.0);
    }

    #[test]
    fn test_loss() {
        let mut model = get_model();
        // Forward pass
        let input = array![[0.0, 1.0, 0.5]];
        model.last_node = Some(model.forward(input));
        // Loss calculation
        let loss = model.loss(&array![[0.0, 0.0]]);
        assert_approx_eq!(loss.value(), array![[0.0809]], 1e-6, array);
    }

    #[test]
    fn test_predict() {
        let model = get_model();
        let input = array![[0.0, 1.0, 0.5]];
        let prediction = model.predict(input);
        assert_approx_eq!(prediction, array![[0.475021, 0.524979]], 1e-6, array);
    }

    #[test]
    fn test_save_and_load() {
        const TEST_PATH: &str = "test_model.bin";
        let model = Model::new(
            vec![
                Layer::linear((784, 784)),
                Layer::relu(),
                Layer::linear((784, 784)),
                Layer::relu(),
                Layer::linear((784, 10)),
                Layer::relu(),
            ],
            Some(Loss::MSE),
            Some(Optimizer::SGD),
            Some(Meta {
                num_epochs: 10,
                learning_rate: 1e-3,
                batch_size: 64,
            }),
        );

        model.save(TEST_PATH);
        let loaded_model = Model::load(TEST_PATH, Some(Loss::MSE), Some(Optimizer::SGD));
        std::fs::remove_file(TEST_PATH).unwrap();

        assert_eq!(model.layers.len(), loaded_model.layers.len());
        assert_eq!(model.meta.unwrap(), loaded_model.meta.unwrap());
        for (layer, loaded_layer) in model.layers.iter().zip(loaded_model.layers.iter()) {
            match layer.l_type {
                LayerType::Custom => {
                    panic!("Impossible :(");
                }
                LayerType::Linear => {
                    assert_eq!(layer.l_type, loaded_layer.l_type);
                    assert_approx_eq!(
                        layer.weights.as_ref().unwrap().value(),
                        loaded_layer.weights.as_ref().unwrap().value(),
                        1e-6,
                        array
                    );
                    assert_approx_eq!(
                        layer.bias.as_ref().unwrap().value(),
                        loaded_layer.bias.as_ref().unwrap().value(),
                        1e-6,
                        array
                    );
                }
                LayerType::ReLU => {
                    assert_eq!(layer.l_type, loaded_layer.l_type);
                }
            }
        }
    }
}
