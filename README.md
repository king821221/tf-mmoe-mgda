# tf-mmoe-mgda
Implement MMOE plus MGDA for multi_task learning.
MMOE implementation follows that from https://github.com/drawbridge/keras-mmoe 
MGDA implementation follows that from https://github.com/intel-isl/MultiObjectiveOptimization
The task shared variables are composed of the MMOE expert kernels, bias and gate kernels and bias
A 2-task classification demo as well as a 3-task classification demo are also presented.They are also from keras-mmoe
