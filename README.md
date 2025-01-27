# Symmetric Vector Classifier Using Perceptron

This repository contains a simple TensorFlow implementation of a perceptron that learns to classify whether an input vector `X` is symmetric.  
It was written in 2019 and is retained here mostly for historical reasons.  

## Features

- Implements a perceptron using **TensorFlow**.
- Trains the model on randomly generated vectors to classify symmetric and non-symmetric vectors.
- Logs the model's weights and performance metrics during training.

## How It Works

1. **Input Data**:
   - The input vectors (`X`) have a size of `2*hvs` (double the "half input vector size").
   - Symmetric vectors are constructed by concatenating a random vector with itself.
   - Non-symmetric vectors are randomly generated with no symmetry between the two halves.

2. **Model Architecture**:
   - Single hidden layer (`L2`) with ReLU activation.
   - Output layer with a sigmoid activation function to classify symmetry (`1` for non-symmetric, `0` for symmetric).

3. **Loss Function**:
   - Mean Squared Error (MSE) is used to measure the difference between predicted and actual outputs.

4. **Training**:
   - Gradient Descent is used to optimize the model parameters.
   - The learning rate decays over time to stabilize training.


## Requirements

- **TensorFlow** (1.x version, as `tf.Session()` and `tf.placeholder` are used).
- **NumPy** (for random vector generation and array operations).

Install dependencies via pip:
```bash
pip install tensorflow==1.15 numpy
```

## Usage
### Running the Script
1. Clone the repository
```bash
git clone https://github.com/your-username/symmetric-vector-classifier.git
cd symmetric-vector-classifier
```
2. Run the script:
```bash
python symmetric_classifier.py
```
## Output

- **Training Progress**:
  - During training, the script logs:
    - Model weights after each epoch.
    - Input vectors (`x_data`) and their corresponding labels (`y_data`).
    - Training loss (cost) at regular intervals.

- **Final Accuracy**:
  - After training, the model evaluates its classification accuracy using the test data.
  - Predicted outputs (`hy`) and their comparison with actual labels (`Y`) are displayed.


## Notes
  
- **Code Limitations**:
  - This implementation assumes perfect symmetry for symmetric vectors and completely random structure for non-symmetric vectors. It does not handle "almost symmetric" cases effectively.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
