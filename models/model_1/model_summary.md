# ***Model 1***

Here is the summary of a trained model for the MNIST dataset.

## **1. Model Design**
## This model is consisted of *3* layers

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
## **3. Model's Hyperparametes**
- ### Model's hyperparameters are:

                Batch size (mini batch): 256 training examples

                Learning rate (alpha): 0.01

                Learning rate decay: 0.002

                Regularization term -L2 regularization- (lambda): 64.0

                Gradient descent with momentum hyperparameter (beta 1): 0.9

                RMSprop hyperparameter (beta 2): 0.999

## **3. Model's Accuracy**
- ### Model's accuracy of the training examples:                98.012

![Accuracy of training examples](./accuracy_graph.png)

- ### Model's accuracy of the cross validating examples:                96.67

- ### Model's accuracy of the testing examples:                96.67

## **4. Model's Losses**
- ### Model's losses of the training examples: 0.00847842919266155

![Cost function of training examples](./j_graph.png)

- ### Model's losses of the cross validating examples: 0.012257237308404344

- ### Model's losses of the testing examples: 0.01218768052905945

## **5. Model's Executed time**
- ### The executed time: 8 minutes,,             4 seconds ,,                 973.04 milli seconds, along 150 epochs

