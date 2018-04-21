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

Let us first try understanding the Reverse mode Automatic differentiation.
Automatic differentiation splits the tak of computing the derivatives into two tasks:  
  1. Forward pass  
  2. Reverse pass  
Consider finding the derivatives of the below function
````
z = x1*x2+sin(x1) 
````
and want to find the derivatives *dz/dx1 and dz/dx2* 

## 1. Forward Pass:  
To break down the complex function into primitive ones. With this principle, above function is broken down by Automatic differentiation 
as:  
```
                                                    w1 = x1  
                                                    w2 = x2  
                                                    w3 = sin(w1)  
                                                    w4 = w1 * w2  
                                                    w5 = w4 + w3  
                                                    z = w5  
 ```
Thus, forward pass is all about evaluating these expressions and saving the result values. 
If the input was *x1 = 1 and x2 = 2*
```
                                                    w1 = x1 = 1
                                                    w2 = x2 = 2
                                                    w3 = sin(w1) = 0.84
                                                    w4 = w1 * w2 = 1 *2 = 2
                                                    w5 = w4 + w3 = 2 * 0.84 = 1.68
                                                    z = 1.68
 ```
 ## 2. Reverse Pass:  
  The chain rule is core to finding derivatives for most of the applications.
  In its basic form, chain rule states that if you have variable t which depends on u which, in its turn, depends on v, then:  
  dt/dv = dt/du * du/dv
 
and try to evaluate their outputs as well as evaluate their gradients by using the computation graph and explicit construction of 
the gradient nodes similar to tensor flow.
