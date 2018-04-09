**Program the dogleg method, and choose Bk to be the exact Hessian. Use the method to minimize the Rosenbrock function**

  1] With initial radius = 1 and ը = 0.15 and x0 = [1.2, 1.2]  
  No of iterations to converge : 8  
  Point of convergence: [1.00000023 , 1.00000047]  
  
  2] With initial radius = 3 and ը = 0.15 and x0 = [1.2, 1.2]  
  No of iterations to converge : 10  
  Point of convergence: [1.00e+00, 1.00e+00]  
  
 **Observations:**  
  1] The size of trust region is very important. As it plays an important role in the rate of convergence and effectiveness.  
  2] If the region is very small then the algorithm take substantial step. If it’s too large, there will be reduction in step
  and search again.  
  3] Trust region radius plays a very important role.  
  4] If rho value is negative then that step is rejected.  
  
  
**Reference:**  
  1] Jorge Nocedal and Stephen J. Wright, Numerical Optimization, 2nd edition,2006, Springer.  
  2] Python Scipy libraries and implementation guide and documentation.  [https://docs.scipy.org/doc/]  
