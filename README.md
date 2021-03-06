# lsnn (Least Square Neural Networks)
> Language : MATLAB

This project is for non-iterative algorithm development which are mostly based on the least square method. 
All of these following algorithms are based on single-hidden layer feed-forward neural network (SLFN) structure.
List of algorithms:
* ELM (Extreme Learning Machine), PCA-ELM (principal component analysis), I-ELM (incremental), EI-ELM (enhanced incremental), DP-ELM (destructive parsimonious), and CP-ELM (constructive parsimonious).
* AIL (Analitycal Incremental Learning)
* LSM (Local Sigmoid Method)
* BP (backpropagation, LM) matlab wraper.

> How to run the comparison of all algorithms
* Regression case: run `runcvreg.m`.
* Classification case: run `runcvclass.m`.
* To find the hyperparameters: run `hpsearching.m`, you can change the case to regression or classification.
* To summary the metrics (accuracy, #nodes, time): run `perfsummay.m, you need to change the metrics manually --see/read the code.