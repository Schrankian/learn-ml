#![allow(dead_code)]
extern crate blas_src;

mod machine_learning;
mod utils;
use clap::Parser;
use machine_learning::autograd_matrix::Matrix;
use machine_learning::data_loader::{self, DataLoader};
use machine_learning::neural_network::{Layer, Loss, Meta, Model, Optimizer};
use std::error::Error;
use std::io::{self, Write};

fn train(
    path: &str,
    model_path: &str,
    use_pretrained: bool,
    epochs: i32,
) -> Result<(), Box<dyn Error>> {
    // Hyper parameters
    const LEARNING_RATE: f32 = 1e-3; // TODO use model-meta instead of hyperparameter
    const BATCH_SIZE: usize = 64; // TODO use model-meta instead of hyperparameter

    let training_data = DataLoader::from_dir(path, "train", BATCH_SIZE)?;
    let test_data = DataLoader::from_dir(path, "t10k", BATCH_SIZE)?;

    // utils::print_image(training_data.images().row(1).to_owned());
    // println!("{:?}", training_data.labels().row(1));

    // Define (or load) the model
    let mut model = match use_pretrained {
        true => Model::load(model_path, Some(Loss::MSE), Some(Optimizer::SGD)),
        false => Model::new(
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
                num_epochs: 0,
                learning_rate: LEARNING_RATE,
                batch_size: BATCH_SIZE,
            }),
        ),
    };

    for epoch in 0..epochs {
        println!("\nStarting training epoch {}/{} ...", epoch + 1, epochs);

        // For debugging purposes
        let mut bencher = utils::Benchmarker::new();
        let mut handle = io::stdout().lock();
        let size = training_data.images().nrows();
        let progress_width = 50;

        // -----------------------
        // Training
        // -----------------------
        for (i, (images, labels)) in training_data.batches(true).enumerate() {
            // Debugging - size of the current batch
            let batch_size = images.nrows();

            // Actual training step
            let loss = model.train(images, &labels, LEARNING_RATE);

            // Debugging - print the current progress
            if i % 10 == 0 {
                let current = i * BATCH_SIZE + batch_size;
                let progress = (progress_width * current / size) as usize;
                handle.write_all(
                    format!(
                        "\r[{}{}] {:>5}/{:>5} Loss: {:>7.6}",
                        "=".repeat(progress),
                        " ".repeat(progress_width - progress),
                        current,
                        size,
                        loss.get((0, 0)).unwrap()
                    )
                    .as_bytes(),
                )?;
                handle.flush()?;
            }
            bencher.round();
        }
        // Update model meta information
        if model.meta.is_some() {
            model.meta.as_mut().unwrap().num_epochs += 1;
        }
        // Print the current training step
        println!(
            "\nTraining time per batch: {:?}, Training time total: {:?}",
            bencher.avg,
            bencher.avg * training_data.num_batches() as u32
        );
    }

    // -----------------------
    // Testing
    // -----------------------
    println!(" - Evaluating...");
    let mut correct = 0;
    for (images, labels) in test_data.batches(false) {
        let y_hat = model.predict(images);
        let predictions = utils::argmax(&y_hat);
        let correct_labels = utils::argmax(&labels);
        correct += predictions
            .iter()
            .zip(correct_labels.iter())
            .filter(|(a, b)| a == b)
            .count();
    }

    // Print the accuracy achieved, after all epochs
    println!(
        "Accuracy: {}%",
        100.0 * correct as f32 / test_data.images().nrows() as f32
    );

    // Save the model
    println!("Saving model...");
    model.save(model_path);

    Ok(())
}

fn eval(path: &str, model: &str) -> Result<(), Box<dyn Error>> {
    // Load the model without optimizer and loss function
    let model = Model::load(model, None, None);

    // Load the test data with a batch size of 1
    let test_data = DataLoader::from_dir(path, "t10k", 1)?;

    let mut correct = 0;
    // This could also be used with batches
    for (images, labels) in test_data.batches(false) {
        let y_hat = model.predict(images);
        let predictions = utils::argmax(&y_hat);
        let correct_labels = utils::argmax(&labels);
        correct += predictions
            .iter()
            .zip(correct_labels.iter())
            .filter(|(a, b)| a == b)
            .count();
    }

    println!(
        "Accuracy: {:.2}%",
        100.0 * correct as f32 / test_data.images().nrows() as f32
    );

    // Print the meta information of the model
    utils::print_meta(&model);

    Ok(())
}

fn normal(model: &str, image: &str) -> Result<(), Box<dyn Error>> {
    // Load the model without optimizer and loss function
    let model = Model::load(model, None, None);

    // Load and preprocess the image
    let image = data_loader::load_image(image);
    utils::print_image(image.row(0).to_owned());

    // Predict the number
    let prediction = model.predict(image);

    println!("-----Prediction results:-----");
    utils::print_softmax(&prediction.row(0).to_owned());

    Ok(())
}

#[derive(clap::ValueEnum, Clone, Default, Debug)]
#[clap(rename_all = "kebab_case")]
enum Mode {
    Train,
    Eval,
    #[default]
    Normal,
}

/// An Image recognicion Programm based on the MNIST Dataset
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Start evaluation mode (which is based on test data from the mnist dataset)
    mode: Mode,

    /// The path to the mnist data directory
    #[arg(long, default_value_t=String::from("./data/MNIST/raw"))]
    data_path: String,

    /// The path to a pretrained model (or the path it should be stored in)
    #[arg(long, default_value_t=String::from("models/model.bin"))]
    model_path: String,

    /// The path to the image, if mode is set to normal
    #[arg(long, short)]
    image_path: Option<String>,

    /// Continue training from the model at model_path
    #[arg(long, short, default_value_t = false)]
    continue_training: bool,

    /// Number of epochs to train
    #[arg(long, short, default_value_t = 10)]
    epochs: i32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    match args.mode {
        Mode::Train => {
            // Data_path: Data source. Model_path: Model save location. Continue_training: use Model_path as starting point
            train(
                &args.data_path,
                &args.model_path,
                args.continue_training,
                args.epochs,
            )?;
        }
        Mode::Eval => {
            // Data_path: Data source. Model_path: Model source.
            eval(&args.data_path, &args.model_path)?;
        }
        Mode::Normal => {
            // Model_path: Model source.
            normal(&args.model_path, &args.image_path.unwrap())?;
        }
    }

    Ok(())
}
