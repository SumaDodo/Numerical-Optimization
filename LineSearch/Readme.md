**Program the steepest descent and Newton algorithms using the backtracking line search, Use them to minimize the Rosenbrock function. 
Set the initial step length alpha = 1 and print the step length used by each method at each iteration. First try the initial point 
x0 = (1:2; 1:2)T and then the more difficult starting point x0 = (1:2; 1:0)T**

The Rosenbrock equation is seen to have the global minimum at approximately (1,1) where the function would be 0.
Here, to find the global minimum we are using the backtracking line search method where each iteration we need to compute the
search direction and the step length.

**Discussion:**  
  1] The initial step length is set to 1.  
  2] ρ is set to 0.9  
  3] The constant c is set to 10-4  

**a) Steepest Descent with initial point [1.2, 1.2]:**  
With the above said parameters, it reaches the minimum point **[1.00e+00, 1.00e+00] after 19679** iterations. However we observe 
that changing the value of constant c makes a great difference as in this case the program reaches the minimum at a faster pace 
with just 490 iterations for c= 0.5.  

**b) Newton method with initial point [1.2, 1.2]:**  
With the above said parameters, it reaches the minimum point [1.00e+00, 1.00e+00] after 140 iterations. However we observe that 
changing the value of constant c makes a great difference as in this case the program reaches the minimum at a faster pace with 
just 9 iterations at c = 0.5.  

**c) Steepest Descent with initial point [1.2, 1]:**  
With the above said parameters, it reaches the minimum point [1.00e+00, 1.00e+00] after 18,103 iterations. However we observe that
changing the value of constant c makes a great difference as in this case the program reaches the minimum at a faster pace with just 
502 iterations at c = 0.5.

**d) Newton’s method with initial point [1.2, 1]:**
With the above said parameters, it reaches the minimum point [1.00e+00, 1.00e+00] after 151 iterations. However we observe that
changing the value of constant c makes a great difference as in this case the program reaches the minimum at a faster pace with 
just 10 iterations for c =0.5.
