# tf-mmoe-mgda

This repo is a tensorflow implementation of MMOE + MGDA multi_task learning algorithm that targets to resolve
task dependency and achieve global optimimum via approaching Pareto Optimality.

1. MMOE(Multi-gate Mixture-of-Experts) could be referred to in paper:
     Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts.
   Our implementation borrows from https://github.com/drawbridge/keras-mmoe

2. MGDA(Multiple Gradient Descent Algorithm) could be referred to in paper:
     Multiple-gradient descent algorithm (MGDA) for multiobjective optimization
   Our implementation borrows from https://github.com/intel-isl/MultiObjectiveOptimization,
   which has provided pytorch and numpy implementations.

3. The task shared variables used in MGDA are composed of 
   1) MMOE expert kernels, bias;
   2) MMOE gate kernels and bias

4. For illustrated purposes, we have provided a 2-task classification demo and a 3-task classification demo.
   Data and origin implementation could be referred to in keras-mmoe

5. Command line:

   1) pip -r requirements.txt

   2) python census_income_demo.py

   3) python synthetic_demo.py

Any feedback and suggestions are greatly appreiciated: 1485840691@qq.com
