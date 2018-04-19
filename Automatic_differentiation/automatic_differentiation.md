Automatic differentiation is sometimes known as mathemagically finding derivatives! Itseems awesomely magical in the sense having known the 
tideous task of computing the differntials and gradients in most of the machine learning and deep learning projects and concepts,
there is a tool that makes your work very easy! 

Automatic differentiation is not just about computing the derivatives automatically but it also comprises about computing 
generalizations like gradients and jacobians. 

Going with the actual textbook definition of differentiation, it's computationally very difficult to make h very small owing to the 
limitations of the computer itself in computing these functions. As the floating point values are rounded off.

![derivative definiton](https://github.com/SumaDodo/Numerical-Optimization/blob/master/Automatic_differentiation/definition-derivative-function-800x800.jpg)

Automatic differentiation computes the derivatives in their exact values! 

To understand the working of automatic differentiation, let us try working out the below problem.

Let us try to construct simple expression like 
````
y = x1+x2*x2 
````
and try to evaluate their outputs as well as evaluate their gradients by using the computation graph and explicit construction of 
the gradient nodes similar to tensor flow.
