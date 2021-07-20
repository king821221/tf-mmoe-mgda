# tf-mmoe-mgda

This repo contains implementation of MMOE and MGDA for multi_task learning.

MMOE(Multi-gate Mixture-of-Experts) implementation follows that from https://github.com/drawbridge/keras-mmoe 

MGDA(multiple gradient descent algorithm) implementation follows that from https://github.com/intel-isl/MultiObjectiveOptimization

The task shared variables are composed of the MMOE expert kernels, bias and gate kernels and bias

A 2-task classification demo as well as a 3-task classification demo are also presented.They are also from keras-mmoe
