# Distributed Optimization Under a Communication Budget
This repository contains implementations for the numerical evaluations in the paper:

R. Saha, M. Pilanci and A. J. Goldsmith, "Efficient Randomized Subspace Embeddings for Distributed Optimization Under a Communication Budget," in IEEE Journal on Selected Areas in Information Theory, vol. 3, no. 2, pp. 183-196, June 2022, doi: 10.1109/JSAIT.2022.3198412. (https://ieeexplore.ieee.org/abstract/document/9857556)

The following are stand-alone scripts and directly running them should work.

1. **compression_methods_map.m**: Gives the result of Fig. 1(a). Comparison of different compression methods with and without near-democratic embedding.

2. **DGD-DEF_comparison.m**: Gives the result of Fig. 1(b). Variation of empirical convergence rate of **DGD-DEF** with bit-budget per dimension (R).

3. **wall_clock_time_comparsion.m**: Gives the result of Fig. 1(c). Wall clock times for computing near-democratic vs. democratic representations.

For simulations with a Support Vector Machine (SVM) on synthetic data,

1. Firstly, run **generate_SVM_data.m** to generate synthetic data for classification.

2. Then, run **SVM_simulations.m** to train an SVM classifier on the generated synthetic data and obtain plots in Figs. 2(a) and 2(b).

For classification on MNIST dataset using SVMs,

1. Run **SVM_simulations_MNIST.m** as a standalone script.

We would appreciate if you reach out and report any issues.



