#![allow(dead_code)]
use ndarray::{Array1, Array2, ArrayBase, Data, Dimension};
use num_traits::Float;
use std::time::{Duration, Instant};

use crate::machine_learning::neural_network::Model;

/// Checks if two floating-point numbers are approximately equal within a given epsilon.
pub fn approx_eq<T: Float>(a: T, b: T, epsilon: T) -> bool {
    (a - b).abs() < epsilon
}

/// Checks if two ndarrays are approximately equal element-wise within a given epsilon.
pub fn approx_eq_array<T, S, D>(a: &ArrayBase<S, D>, b: &ArrayBase<S, D>, epsilon: T) -> bool
where
    T: Float,
    S: Data<Elem = T>,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| approx_eq(*x, *y, epsilon))
}

#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
        assert_approx_eq!($a, $b, 1e-6);
    };
    ($a:expr, $b:expr, $epsilon:expr) => {
        if !crate::utils::approx_eq($a, $b, $epsilon) {
            panic!(
                "assertion failed: `(left ≈ right)` \
                 (left: `{:?}`, right: `{:?}`, epsilon: `{:?}`)",
                $a, $b, $epsilon
            );
        }
    };
    ($a:expr, $b:expr, $epsilon:expr, array) => {
        if !crate::utils::approx_eq_array(&$a, &$b, $epsilon) {
            panic!(
                "assertion failed: `(left ≈ right)` \
                 (left: `{:?}`, right: `{:?}`, epsilon: `{:?}`)",
                $a, $b, $epsilon
            );
        }
    };
}

/// Returns the index of the maximum value in each row of a 2D array.
pub fn argmax(array: &Array2<f32>) -> Vec<usize> {
    array
        .map_axis(ndarray::Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        })
        .to_vec()
}

/// Prints the top 3 probabilities of a softmax output.
pub fn print_softmax(array: &Array1<f32>) {
    let mut probs: Vec<(usize, &f32)> = array.iter().enumerate().collect();
    probs.sort_by(|item1, item2| item2.1.total_cmp(item1.1));

    for item in probs.iter().take(3) {
        println!("{}, Prob: {:.2}%", item.0, item.1 * 100.0)
    }
}

/// Prints the 1D array as a 28x28 image.
pub fn print_image(image: Array1<f32>) {
    for (i, &pixel) in image.iter().enumerate() {
        if pixel > 0.0 {
            print!("{:.3}", pixel);
        } else {
            print!(".....");
        }
        if (i + 1) % 28 == 0 {
            println!();
        }
    }
}

/// Prints the meta information of a model.
pub fn print_meta(model: &Model) {
    println!("Meta:");
    println!("  Layers: {}", model.num_layers());
    if let Some(meta) = &model.meta {
        println!("  Learning rate: {}", meta.learning_rate);
        println!("  Batch size: {}", meta.batch_size);
        println!("  Epochs: {}", meta.num_epochs);
    }
}

pub struct Break {
    name: String,
    elapsed: Duration,
}

/// A simple benchmarking tool
///
/// This can be used in two ways. The first one is to measure the average time an action takes.
/// For this, after every time the function is ran, call bencher.round().
/// This will store the duration since last time and calculate the average time.
/// ```rust
/// let mut bencher = Benchmarker::new();
/// for _ in 0..100 {
///    // Do something
///   bencher.round();
/// }
/// println!("Average time: {:?}", bencher.avg);
/// ```
///
/// The second way is to measure the time between multiple points in a function.
/// For this, call bencher.break_point("name") at every point you want to measure.
/// After the benchmark is done, call bencher.print() to print the results.
/// Keep in mind that calling bencher.round() will reset the break points.
/// ```rust
/// let mut bencher = Benchmarker::new();
/// // Do First step
/// bencher.break_point("First step");
/// // Do Second step
/// bencher.break_point("Second step");
/// bencher.print();
/// ```
pub struct Benchmarker {
    start: Instant,
    pub avg: Duration,
    durations: Vec<Duration>,
    breaks: Vec<Break>,
}

impl Benchmarker {
    pub fn new() -> Benchmarker {
        Benchmarker {
            start: Instant::now(),
            avg: Duration::new(0, 0),
            breaks: Vec::new(),
            durations: Vec::new(),
        }
    }

    /// Store the time since start or last round() call and calculate the average time of all stored durations.
    ///
    /// This will reset the break points!
    pub fn round(&mut self) {
        self.breaks.clear();
        let elapsed = self.start.elapsed();
        let elapsed_total = self.breaks.iter().map(|v| v.elapsed).sum::<Duration>() + elapsed;
        self.durations.push(elapsed_total);
        self.avg = self.durations.iter().sum::<Duration>() / self.durations.len() as u32;
        self.start = Instant::now();
    }

    /// Store the time since start or last break_point() call.
    /// Calling print() will print all break points by name and the time between them.
    pub fn break_point(&mut self, name: &str) {
        let elapsed = self.start.elapsed();
        self.breaks.push(Break {
            name: name.to_string(),
            elapsed,
        });
        self.start = Instant::now();
    }

    /// Print all break points by name and the time between them.
    pub fn print(&self) {
        println!("[Bencher] ----------------------");
        for b in &self.breaks {
            println!("[Bencher] {}: {}s", b.name, b.elapsed.as_secs_f64());
        }
        println!(
            "[Bencher] Total: {}s",
            (self.breaks.iter().map(|v| v.elapsed).sum::<Duration>() + self.start.elapsed())
                .as_secs_f64()
        );
    }
}
