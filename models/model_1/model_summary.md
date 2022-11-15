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
- ### Model's accuracy of the training examples:                 99.70400000000001

![Accuracy of training examples](./accuracy_graph.png)

- ### Model's accuracy of the cross validating examples:                 95.63000000000001

- ### Model's accuracy of the testing examples:                 95.74000000000001

## **3. Model's Losses**
- ### Model's losses of the training examples: 0.001554428475146392

![Cost function of training examples](./j_train_graph.png)

- ### Model's losses of the cross validating examples: 0.02648271543828856

- ### Model's losses of the testing examples: 0.026656296002920184

## **4. Model's Executed time**
- ### The executed time: 6 minutes,,             35 seconds ,,                 294.19 milli seconds, along 200 epochs

