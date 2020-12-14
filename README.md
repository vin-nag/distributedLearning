# Analyzing Variations in Distributed Deep Neural Network Learning

By Vineel Nagisetty; Husayn Kara; 

## Table of Contents
* Introduction
* Usage
* More Info

## Introduction:
Our project implements Downpour Stochastic Gradient Descent (Downpour-SGD), an asynchronous distributed deep neural network training algorithm using Thrift IDL (for Python) and Pytorch. We also implement several novel variations to parts of the Downpour-SGD - namely in the data splitting as well as parameter aggregation steps.

 ## Usage:
 ### Source Code:
 * The code for the front-end, back-end and client servers is found in the `src/frontEnd/serverFE.py`, `src/backEnd/serverBE.py`, and `src/client/client.py` files respectively
 * The code for the fron-tend and back-end service handlers (that implement a majority of the Downpour-SGD algorithm) is found in the `src/frontEnd/serviceFE.py` and `src/backEnd/serviceBE.py` files respectively
 * The Deep Neural Network model architecture code is found in the `src/utils/model.py` file
 * The code for data splitting and parameter aggregation methods is found in the `src/utils/splitData.py` and `src/utils/aggregateFeedback.py`
 * The code for plotting the results is found in the `src/utils.plotter.py`
 
 ### Reproduce Results:
 * To reproduce the results shown in the report, please install Thrift and all modules from the `requirements.txt` file and run the front-end, back-end and client servers (in that order)
 
