#![allow(dead_code)]
use crate::utils;
use ndarray::{Array2, Zip};
use ndarray_conv::{ConvMode, PaddingMode, ConvFFTExt, ConvExt};
use rand::Rng;
use std::cell::RefCell;
use std::fmt;
use std::ops;
use std::rc::Rc;

/// This type defines an alias for a reference-counted RefCell of a MatrixNode <br>
/// That means, different NodeRefs can point to the same Node, and they can be mutated.
/// Rc therefore counts the number of references to the Node, and RefCell allows for interior mutability
type MatrixNodeRef = Rc<RefCell<MatrixNode>>;

/// This type defines an alias for a dynamic dispatch function that takes the pointer to an 2D Array and returns a Vec<Array2<f32>> (the gradient contributions) <br>
/// The root node is always None, the rest of the nodes have a back_pass function that calculates the gradient contributions
/// Explanation:
/// > - The Box<> is a smart pointer, that allocates the function on the heap, so it can be stored in the Node struct as a callback <br>
/// > - The dyn keyword is used to create a trait object, which allows for dynamic dispatch <br>
/// > - The Fn(&Array2<f32>) -> Vec<Array2<f32>> definies a function signature that takes an &Array2<f32> and returns a Vec<Array2<f32>>. E.g. it takes in the seed
/// of the last node and returns the gradient contributions of the current node, with respect to each parent node (e.g. if 1 parent -> the vec.length() == 1, if 2 parents -> vec.length() == 2) <br>
/// > - The 'static lifetime is used to specify that the function lives for the entire duration of the program
/// It is also rensponsible for allowing the move closure, which is used to move (or copy) any referenced (potential temporary) variables into the closure
///
/// Difference to just defining it as fn(&Array2<f32>) -> Vec<Array2<f32>:
/// It is a function signature for static dispatch, it cannot capture variables from the environment, and is therefore not used here
type BackPass = Option<Box<dyn Fn(&Array2<f32>) -> Vec<Array2<f32>> + 'static>>;

struct MatrixNode {
    // Remember parents for backpropagation
    parents: Vec<MatrixNodeRef>,
    // The value of the current node
    data: Array2<f32>,
    // The gradient of the current node
    grad: Array2<f32>,
    // The closure to calculate the gradient
    back_pass: BackPass,
    // Helper functions for user friendliness
    backward_called: bool,
    is_end_node: bool,
}

impl Drop for MatrixNode {
    /// Callback for when a MatrixNode is dropped.
    /// So that the parents know, that they are end nodes
    fn drop(&mut self) {
        for parent in &self.parents {
            parent.borrow_mut().is_end_node = true;
        }
    }
}

impl MatrixNode {
    /// Create and attach a new MatrixNode
    fn new(data: Array2<f32>, parents: Vec<MatrixNodeRef>, back_pass: BackPass) -> MatrixNodeRef {
        for parent in &parents {
            parent.borrow_mut().is_end_node = false;
        }
        let dim = data.raw_dim();
        Rc::new(RefCell::new(MatrixNode {
            data,
            parents,
            grad: Array2::zeros(dim),
            back_pass,
            backward_called: false,
            is_end_node: true,
        }))
    }

    /// Starts the backpropagation and only stops when the back_pass of a node is None
    fn backward(&mut self, seed: &Array2<f32>) {
        if self.grad.shape() != seed.shape() {
            // Handle the case, that the seed has a higher dimension than the gradient
            // This commonly happens during batch training
            // So we sum the gradients along the batch dimension
            let summed_seed = seed
                .sum_axis(ndarray::Axis(0))
                .insert_axis(ndarray::Axis(0));
            self.grad += &summed_seed;
        } else {
            self.grad += seed;
        }
        self.backward_called = true;

        if let Some(ref back_pass) = &self.back_pass {
            let grad_contributions = back_pass(seed);

            for (i, parent) in self.parents.iter().enumerate() {
                parent.borrow_mut().backward(&grad_contributions[i]);
            }
        }
    }

    /// Resets the gradient of the node and all its parents
    fn reset_grad(&mut self) {
        self.grad.fill(0.0);
        self.backward_called = false;

        for parent in &self.parents {
            parent.borrow_mut().reset_grad();
        }
    }
}

pub struct Matrix {
    node: MatrixNodeRef,
}

impl Matrix {
    /// Create a new matrix with the given data, parents and back pass function
    /// Important: Passing None as the back_pass function will result in the back pass stopping on this node
    /// So if this happens on the last node, the back pass will not even start
    fn new(data: Array2<f32>, parents: Vec<MatrixNodeRef>, back_pass: BackPass) -> Self {
        Self {
            node: MatrixNode::new(data, parents, back_pass),
        }
    }

    /// Convert an Array2 to a Matrix, which supports automatic differentiation
    pub fn from(data: Array2<f32>) -> Self {
        Self {
            node: MatrixNode::new(data, vec![], None),
        }
    }

    /// Creates a new matrix with random values from a normal distribution
    pub fn randn(shape: (usize, usize)) -> Self {
        let mut rng = rand::rng();
        let data: Array2<f32> = Array2::from_shape_fn(shape, |_| {
            let stddev = (2.0 / shape.0 as f32).sqrt();
            rng.sample::<f32, _>(rand_distr::Normal::new(0.0, stddev).unwrap())
        });
        Self::from(data)
    }

    /// Performs a backward pass starting from this node and ending at the input nodes
    /// Panics if called on a non-end node (as this would not make sense, or indicate wrong usage)
    pub fn backward(&self, seed: Option<Array2<f32>>) {
        let mut node = self.node.borrow_mut();

        if !node.is_end_node {
            panic!("Error: backward called on a non-end node");
        }
        if node.backward_called {
            println!("Warning: backward called multiple times on the same node");
        }

        // Get the initial seed or create an identity matrix if not provided.
        // This is not ideal for individual unit-tests, but for our use-case it is acceptable
        // In a normal training scenario, the here generated seed is just [[1.0]] (for the loss node)
        let seed = seed.unwrap_or_else(|| {
            Array2::eye(node.data.nrows())
                .broadcast(node.data.raw_dim())
                .unwrap()
                .to_owned()
        });

        // Ensure the seed has the correct shape
        if seed.raw_dim() != node.data.raw_dim() {
            panic!(
                "Error: Seed shape mismatch. Expected {:?}, got {:?}",
                node.data.raw_dim(),
                seed.raw_dim()
            );
        }

        // The node will take care of the rest, e.g. propagate the seed to the parents
        node.backward(&seed);
    }

    /// Should not be called lightly, only for debugging
    /// Clones and returns the gradient
    pub fn grad(&self) -> Array2<f32> {
        let node = self.node.borrow();
        if !node.backward_called {
            println!(
                "Warning: backward not called on this node, so cannot access the gradient yet"
            );
        }

        node.grad.clone()
    }

    /// Should not be called lightly, only for debugging
    /// Clones and returns the value
    pub fn value(&self) -> Array2<f32> {
        self.node.borrow().data.clone()
    }

    /// Clones itself, by cloning the underlying node Rc
    /// This is cheap, because the new Matrix will still point to the same node
    pub fn clone(&self) -> Matrix {
        Self {
            node: self.node.clone(),
        }
    }

    /// Resets the gradient of the node and all its parents
    /// Currently, doesn't panic if called on a non-end node. (But this could change in the future)
    pub fn reset_grad(&self) {
        let mut node = self.node.borrow_mut();

        // This is a bit tricky, because normally reset_grad should only be called on end nodes
        // to avoid accumulating gradients. However, in the case of my neural network implementation
        // this method is called on the result of the forward pass, which is not the end node (because the loss is the end node)
        // So its removed for the time being
        // if !node.is_end_node {
        //     panic!("Error: reset_grad called on a non-end node");
        // }

        node.reset_grad();
    }

    /// Perform matrix multiplication
    /// The shapes need to match like this: (m, n) * (n, p) = (m, p)
    pub fn dot(&self, other: &Self) -> Self {
        // Perform matrix multiplication
        let data = self.node.borrow().data.dot(&other.node.borrow().data);

        // Create a weak reference to self and other to avoid reference cycles
        // These can then be safely captured in the closure
        let weak_self = Rc::downgrade(&self.node);
        let weak_other = Rc::downgrade(&other.node);

        let back_pass: BackPass = Some(Box::new(move |seed| {
            // Get strong references to self and other
            if let (Some(self_node), Some(other_node)) = (weak_self.upgrade(), weak_other.upgrade())
            {
                // Calculate the gradient to be used as the next seed in the back pass
                let self_grad = seed.dot(&other_node.borrow().data.t());
                // Calculate the gradient to be passed to the weight matrix
                let other_grad = self_node.borrow().data.t().dot(seed);
                vec![self_grad, other_grad]
            } else {
                panic!("Error: weak reference to node in dot back_pass is None");
            }
        }));
        // First parent: input matrix, second parent: weight matrix
        Self::new(data, vec![self.node.clone(), other.node.clone()], back_pass)
    }

    /// Element-wise rectified linear unit (ReLU) activation function
    /// E.g. sets all negative values to zero
    pub fn relu(&self) -> Self {
        let node = self.node.borrow();
        // Create a mask for all values greater than zero, which can be also used for the back pass
        let mask = node.data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        // Apply the mask to the data
        let data = &node.data * &mask;
        // Very important: Drop the borrow here, because it would conflict with the Self::new method which will try to borrow the node mutably
        drop(node);

        let back_pass: BackPass = Some(Box::new(move |seed| {
            // The derivative of the ReLU function is 1 for all values greater than zero and 0 otherwise
            // So we can just use the mask here again
            let grad = seed * &mask;
            vec![grad]
        }));
        Self::new(data, vec![self.node.clone()], back_pass)
    }

    /// Convolves the matrix with a kernel
    /// The kernel is a 2D matrix, which is applied to the input matrix
    pub fn convolute(&self, kernel: &Self) -> Self {
        let node = self.node.borrow();
        let kernel_node = kernel.node.borrow();

        // Convolution parameters
        let conv_mode = ConvMode::Valid;
        let padding_mode = PaddingMode::Zeros;
        let data = if kernel_node.data.shape()[0] > 11 || kernel_node.data.shape()[1] > 11 {
            // Use FFT convolution for large kernels
            node.data.conv_fft(&kernel_node.data, conv_mode, padding_mode)
        } else {
            // Use normal convolution for smaller kernels
            node.data.conv(&kernel_node.data, conv_mode, padding_mode)
        }.expect("Error: Convolution failed");
        drop(node);
        drop(kernel_node);

        let back_pass: BackPass = Some(Box::new(move |_| {
            // TODO
            vec![]
        }));

        Self::new(data, vec![self.node.clone(), kernel.node.clone()], back_pass)
    }

    /// Mean squared error (MSE) loss
    /// Returns a Matrix of shape (1, 1) - So it can be a good starting point for the back pass
    pub fn mse(&self, target: &Array2<f32>) -> Self {
        let node = self.node.borrow();
        // First calculate the difference between the prediction and the target
        let diff = &node.data - target;

        // Then square all elements, sum them up and divide by the number of elements
        // This also works for batch training, because the sum is over all elements
        let data = diff.mapv(|x| x.powi(2)).sum() / (diff.len() as f32);
        drop(node);

        let back_pass: BackPass = Some(Box::new(move |seed| {
            // The derivative of the MSE loss is 2 * (prediction - target) / n
            vec![
                seed.broadcast(diff.raw_dim()).unwrap().to_owned() // Broadcast the seed to the shape of diff, so we can multiply element-wise
                    * (2.0 / (diff.len() as f32))
                    * &diff,
            ]
        }));
        Self::new(
            Array2::from_elem((1, 1), data),
            vec![self.node.clone()],
            back_pass,
        )
    }

    /// Returns the index of the maximum value of each row
    /// Is a bit more efficient than calling utils::argmax(result.value())
    /// But currently useless, because the predict method of the Model struct returns an Array2 instead of a Matrix
    pub fn argmax(&self) -> Vec<usize> {
        utils::argmax(&self.node.borrow().data)
    }

    /// Perform a softmax operation on the matrix
    /// The softmax function is defined as: softmax(x) = exp(x) / sum(exp(x))
    /// In short, it squashes the values to be between 0 and 1 and the sum of all values is 1
    pub fn softmax(&self) -> Self {
        let node = self.node.borrow();
        let data = node.data.mapv(|x| x.exp());
        let sum = data.sum();
        let data = data / sum;
        drop(node);

        // TODO implement back pass if really needed. but it is not needed for the current use-case
        let back_pass: BackPass = None;

        Self::new(data, vec![self.node.clone()], back_pass)
    }

    // TODO implement cross entropy

    /// Perform a gradient descent step on the matrix
    /// E.g. subtracts the gradient times the learning rate from the data
    /// This method is defined here, because it needs access to the gradient and I don't want to expose it
    /// But theoretically it could/should be moved to the nn::Optimizer module
    pub fn grad_desc(&self, learning_rate: f32) {
        let grad = self.node.borrow().grad.clone();
        let mut node = self.node.borrow_mut();

        Zip::from(&mut node.data)
            .and(&grad)
            .for_each(|data, &grad| *data -= grad * learning_rate);
    }
}

/// Implement the Add trait for the Matrix struct
/// This allows for element-wise addition of two matrices, e.g. for the bias
impl ops::Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        let data = &self.node.borrow().data + &other.node.borrow().data;
        let back_pass: BackPass = Some(Box::new(move |seed| vec![seed.clone(), seed.clone()]));
        Matrix::new(data, vec![self.node.clone(), other.node.clone()], back_pass)
    }
}

impl ops::Add<&Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        &self + other
    }
}

impl ops::Add<Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix {
        self + &other
    }
}

impl ops::Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix {
        &self + &other
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let node = self.node.borrow();
        write!(
            f,
            "[value: {}, grad: {}, grad_calculated: {}]",
            node.data, node.grad, node.backward_called
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;

    use super::*;
    use ndarray::array;

    #[test]
    fn test_matrix_dot() {
        let a = Matrix::from(array![[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::from(array![[5.0, 6.0], [7.0, 8.0]]);
        let c = a.dot(&b);
        assert_eq!(c.value(), array![[19.0, 22.0], [43.0, 50.0]]);
        c.backward(None);
        assert_eq!(a.grad(), array![[5.0, 7.0], [6.0, 8.0]]);
        assert_eq!(b.grad(), array![[1.0, 3.0], [2.0, 4.0]]);

        let d = Matrix::from(array![[5.0, 6.0]]);
        let e = Matrix::from(array![[1.0, 2.0], [3.0, 4.0]]);
        let f = d.dot(&e);
        assert_eq!(f.value(), array![[23.0, 34.0]]);
        f.backward(None);
        assert_eq!(d.grad(), array![[3.0, 7.0]]);
        assert_eq!(e.grad(), array![[5.0, 5.0], [6.0, 6.0]]);
    }

    #[test]
    fn test_matrix_relu() {
        let a = Matrix::from(array![[1.0, -2.0], [-3.0, 4.0]]);
        let b = a.relu();
        assert_eq!(b.value(), array![[1.0, 0.0], [0.0, 4.0]]);
        b.backward(None);
        assert_eq!(a.grad(), array![[1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn test_matrix_mse() {
        let a = Matrix::from(array![[1.0, 2.0, 3.0, 2.0]]);
        let b = a.mse(&array![[-1.0, 0.0, 1.0, 2.0]]);
        assert_eq!(b.value(), array![[3.0]]);
        b.backward(None);
        assert_eq!(a.grad(), array![[1.0, 1.0, 1.0, 0.0]]);
    }

    #[test]
    fn test_matrix_complex() {
        let input = Matrix::from(array![[0.0, 1.0, 0.5]]);
        let w1 = Matrix::from(array![[0.1, 0.2, 0.3], [0.2, -0.3, 0.4], [0.3, -0.4, 0.5]]);
        let w2 = Matrix::from(array![[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]);

        let output = input
            .dot(&w1)
            .relu()
            .dot(&w2)
            .relu()
            .mse(&array![[0.0, 0.0]]);
        assert_approx_eq!(output.value(), array![[0.0809]], 1e-6, array);
        output.backward(None);
        assert_approx_eq!(input.grad(), array![[0.0692, 0.0982, 0.1272]], 1e-6, array);
        assert_approx_eq!(
            w1.grad(),
            array![[0.0, 0.0, 0.0], [0.089, 0.0, 0.201], [0.0445, 0.0, 0.1005]],
            1e-6,
            array
        );
        assert_approx_eq!(
            w2.grad(),
            array![[0.0805, 0.1155], [0.0, 0.0], [0.1495, 0.2145]],
            1e-6,
            array
        );
    }

    #[test]
    fn test_matrix_argmax() {
        let a = Matrix::from(array![[1.0, 2.0, 3.0], [4.0, 6.0, 5.0]]);
        let b = a.argmax();
        assert_eq!(b, vec![2, 1]);
    }

    #[test]
    fn test_grad_desc() {
        let a = Matrix::from(array![[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::from(array![[5.0, 6.0], [7.0, 8.0]]);
        let c = a.dot(&b);
        c.backward(None);
        a.grad_desc(0.1);
        b.grad_desc(0.1);
        assert_eq!(a.grad(), array![[5.0, 7.0], [6.0, 8.0]]);
        assert_eq!(b.grad(), array![[1.0, 3.0], [2.0, 4.0]]);
        assert_approx_eq!(a.value(), array![[0.5, 1.3], [2.4, 3.2]], 1e-6, array);
        assert_approx_eq!(b.value(), array![[4.9, 5.7], [6.8, 7.6]], 1e-6, array);
    }

    #[test]
    fn test_drop_hook() {
        let a = Matrix::from(array![[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::from(array![[5.0, 6.0], [7.0, 8.0]]);
        let c = a.dot(&b);
        assert_eq!(c.node.borrow().is_end_node, true);
        assert_eq!(a.node.borrow().is_end_node, false);
        assert_eq!(b.node.borrow().is_end_node, false);
        drop(c);
        assert_eq!(a.node.borrow().is_end_node, true);
        assert_eq!(b.node.borrow().is_end_node, true);
    }

    #[test]
    fn test_softmax() {
        let a = Matrix::from(array![[1.0, 2.0, 3.0]]);
        let b = a.softmax();
        assert_approx_eq!(
            b.value(),
            array![[0.090031, 0.244728, 0.665241]],
            1e-6,
            array
        );
    }

    #[test]
    fn test_addition() {
        let a = Matrix::from(array![[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::from(array![[5.0, 6.0]]);
        let c = &a + &b;
        assert_eq!(c.value(), array![[6.0, 8.0], [8.0, 10.0]]);
        // c.backward(Some(array![[1.0, 1.0], [1.0, 1.0]]));
        // assert_eq!(a.grad(), array![[1.0, 1.0], [1.0, 1.0]]);
        // assert_eq!(b.grad(), array![[1.0, 1.0]]);
    }
}
