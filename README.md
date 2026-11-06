# On the Path Dependence of Derivatives via Monte Carlo Methods

## Erdős Institute Project: Quant Finance Boot Camp
 

### Summary

Option prices that obey the Black–Scholes differential equations can be evaluated analytically in only a handful of cases. Monte Carlo simulations are efficient numerical methods that scale well with the number of dimensions and are easy to implement on a computer. In this project, we focus on the Monte Carlo pricing of options for which no analytical solution exists. This is the case for American options and more exotic options that depend on the path taken by the underlying asset. In this project, I will consider American options and an Americanized arithmetic-average Asian option.

### Description of the repository
This project consist of code stored in 
/src and several jupyter notebooks stored
/examples. The notebooks illustrate the code in /src. 

- [random_behavior_of_assets.ipynb](examples/random_behavior_of_assets.ipynb), where I follow the analysis of Wilmott's book on Quantitative Finance
- [brownian_motion.ipynb](examples/brownian_motion.ipynb), where I study the Geometric Brownian Motion starting from the Brownian motion without drift
- [code_dev_path_dependence.ipynb](examples/code_dev_path_dependence.ipynb), where I develop code to implement path-dependent pricing of american 
 options and the americanized version of asian options

- [path_dependence.ipynb](examples/path_dependence.ipynb), where I present the main analysis


The technical report can be found [here](summary_project.pdf).