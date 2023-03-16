# ML Regression with numpy

This code was created with the main goal to show how to build an ML regression model with gradient descent for the optimization part.

The definition of the problem was obtained from the portal of https://dataflowr.github.io/website/modules/2b-automatic-differentiation/

The equation model is: 

<p align="center">
$y_{t} = 2x_{t}^{1} - 3x_{t}^{2} + 1, t$ $\epsilon$ $\left( 1,...,30 \right)$
</p>

Our task is given the observation $\left( x_{t},y_{t} \right)_{t\epsilon(1,...,30)}$ to recover the weights $w^{1}=2$, $w^{2}=-3$ and the bias $b=1$. In order to do so, we will solve the following optimization problem. 

<p align="center">
$\underset{w^{1},w^{2},b}{\mathrm{argmin}}$ $\Sigma_{t=1}^{30}$ $(w^{1}x_{t}^{1} + w^{2}x_{t}^{2} + b - y_{t} )^{2}$
</p>

### Computing gradient descent

In vector form, it's define: 

<p align="center">
$y_{pred_{t}} = w^{T}x_{t} + b$
</p>

and we want to minimize the loss given by:

<p align="center">
$loss = \Sigma_{t} (y_{pred_{t}} - y_{t})^{2}$
</p>

To minimize the loss we first compute the **gradient** of each loss t:

<p align="center">
$\frac{\sigma loss_{t}}{\sigma w} = \Sigma_{t} 2*(y_{pred_{t}} - y_{t})*x_{t}$
</p>





