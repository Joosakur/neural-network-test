# Neural network from scratch

This is an attempt to build a simple image recognition neural network from scratch with kotlin.
The network is modelled mostly in an object-oriented fashion and without matrix calculations.
The point is to try to understand its inner workings in a more concrete fashion.
This is NOT an attempt to create a well performing and effective solution.

## Training and test data

By default, the data used for training and testing is a set of images of handwritten digits. 

## Requirements
- kotlin
- Download the four data files from http://yann.lecun.com/exdb/mnist/ an extract to project root.

## Edit parameters
You may change e.g. network topology and step size by editing parameters.kt

## Running
Run main.kt

## Example results

Network without hidden layers works surprisingly well.
![Result 1](./results/example-results-1.PNG)

