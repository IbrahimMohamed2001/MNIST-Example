# ***Model 1***

Here is the summary of a trained model for the MNIST dataset.

## **1. Model Design**
## This model is consisted of *3* layers:

Layer 1: 
Layer 1 is consisted of *25* neurons.

so the shape of its *Weights and Biases* are: 

 - Weights = (784, 25)

- Biases = (1, 25)

Layer 2: 
Layer 2 is consisted of *15* neurons.

so the shape of its *Weights and Biases* are: 

 - Weights = (25, 15)

- Biases = (1, 15)

Layer 3: 
Layer 3 is consisted of *10* neurons.

so the shape of its *Weights and Biases* are: 

 - Weights = (15, 10)

- Biases = (1, 10)

The total parameters of this model = 20175
## **2. Model's Accuracy**
- ### Model's accuracy on the training examples:                 88.465

- ### Model's accuracy on the testing examples:                 88.26

## **3. Model's Losses**
- ### Model's losses on the training examples:                 0.038948928150391

- ### Model's losses on the testing examples:                 0.04146647030298654
