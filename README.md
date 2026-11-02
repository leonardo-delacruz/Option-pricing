# derivatives_pricing

## Project of the Erdos Institute Project bootcamp on finance

### Executive summary
Options prices that obey the Black-Scholes differential equations can 
be evaluated analitycally only in handful of cases. Monte
Carlo methods are efficient numerical methods that scale well
as the number of dimensions increases and that are easy to
implement in a computer. In this project, we focus on the
Monte Carlo pricing of options where no-analytical solution
exist. This is the case of American options and more exotics
option that depend of the  path taken by the underlying. In 
this repository, I will consider American options and an Americanized 
arithmethic-average  Asian option.

### Description of the repository
This project consist of code stored in 
/src and several jupyter notebooks stored
/examples. The notebooks illustrate the code in /src. 

- random_behavior_of_assets.ipynb, where I follow the analysis of Wilmott's book on Quantitative Finance
- brownian_motion.ipynb, where I study the Geometric Brownian Motion starting from the Brownian motion without drift
- path_dependece.ipynb, where I develop code to implement path-dependent pricing of american 
 options and the americanized version of asian options
