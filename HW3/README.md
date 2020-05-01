# CPSC 881 Deep Learning HW3

## Introduction
In this assignment, two models are trained on CIFAR10 dataset:
* DCGAN
* WGAN-GP

## Instruction
Tensorflow 1.15, numpy, tqdm, and scikit-learn are used for this assignment. Make sure that these libraries are installed before running the code. The individual files included in this project are listed as below.
* DCGAN.py
* WGANGP.py
* dataset.py
* score.py

To visualize the results, run DCGAN.py or WGANGP.py to plot generated images and calculate IS and FID score. To train the network, simply comment the test() function and run the train(epoch) function in DCGAN.py and WGAN.py.
