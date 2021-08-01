# tf-mmoe-mgda

This repo is an implementation of MMOE + MGDA multi_task learning algorithm framework which targets to resolve
task dependency and achieve global optimal via approaching Pareto Optimality.

MMOE(Multi-gate Mixture-of-Experts) could be referred to in paper:
  Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts.
Our implementation borrows from https://github.com/drawbridge/keras-mmoe

MGDA(Multiple Gradient Descent Algorithm) could be referred to in paper:
  Multiple-gradient descent algorithm (MGDA) for multiobjective optimization
Our tensorflow implementation borrows from https://github.com/intel-isl/MultiObjectiveOptimization,
which provide pytorch and numpy implementations.

The task shared variables in MGDA are composed of the MMOE expert kernels, bias and gate kernels and bias.

For illustrated purposes, a 2-task classification demo and a 3-task classification demo are provided.
Data and origin implementation are from keras-mmoe

Steps:

pip -r requirements.txt

python census_income_demo.py

python synthetic_demo.py
