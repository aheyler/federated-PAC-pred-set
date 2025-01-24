# Federated PAC Prediction Sets

One challenge in machine learning is determining whether model predictions are trustworthy. One promising approach to build trust is to quantify the uncertainty of model predictions with probably approximately correct (PAC) prediction sets, or sets of values that contain the true label with at most some error with high probability. While prior work addresses how to do this on centralized data, growing applications of ML to decentralized data also motivate a decentralized approach such as federated learning. 

This repository generates PAC prediction sets for model predictions when the model is trained via federated learning. 


## MNIST Demo

To run the algorithm on the MNIST dataset, use the following: 
```
python3 main_cls_mnist.py --exp_name exp_mnist
```

