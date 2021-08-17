# TF-MMOE-MGDA

This repo is a tensorflow implementation of MMOE + MGDA multi_task learning algorithm that targets to resolve
task dependency and achieve global optimimum via approaching Pareto Optimality.

This repo includes:

1. MMOE(Multi-gate Mixture-of-Experts) 
   - It could be referred to in paper:
     [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](http://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-).
   - Our implementation is built upon the [Keras MMOE implementation](https://github.com/drawbridge/keras-mmoe)

2. MGDA(Multiple Gradient Descent Algorithm)
   - It could be referred to in paper:
     [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://arxiv.org/abs/1810.04650)
   - Our implementation is built upon the [MGDA implementation from intel](https://github.com/intel-isl/MultiObjectiveOptimization),
     which has provided pytorch and numpy versions.

3. The task shared variables used in MGDA are composed of 
   - MMOE expert kernels, bias;
   - MMOE gate kernels and bias

4. For illustrated purposes, we have provided a 2-task classification demo and a 3-task classification demo.
   The training and evaluation data is from keras-mmoe.

# Getting Started

## Requirements
 - Python 3.5
 - Tensorflow 1.9.0 and other libraries listed in `requirements.txt`

## Installation and Run
 1. Clone the repository
 2. Install dependencies
 ```
 pip install -r requirements.txt
 ```
 4. Run demo code
 ```
 python census_income_demo.py
 python synthetic_demo.py
 ```

Any feedback and suggestions are greatly appreiciated: 1485840691@qq.com
