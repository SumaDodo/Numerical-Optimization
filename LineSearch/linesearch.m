function line_search()
step_len_max = 1;
rho = 0.9;
c = 10^-4;
%c = 0.5;

%x_axis = linspace(-1.2,1.2);
%y_axis = linspace(-1,3);
%[x,y] = meshgrid(x_axis,y_axis);
%x_1 = [x,y];
%fu = func(x_1);
%levels = 10:10:300;
%L = 'linewidth';
%Fs = 'fontsize';
%M = 'markersize';
%title('Contour plot of Rosenbrock function',Fs,16)
%figure, contour(x_axis,y_axis,fu,levels,L,1.2), colorbar
%axis([-1 1 -1 3]), axis square, hold on 


x_0 = [1.2; 1.2];
[x_p, f_obj, step_len, iter] =  backtracking(x_0, 'steepest descent', ...
                                              step_len_max, rho, c);%Steepest Descent with [1.2, 1.2]
plot_results(x_p, f_obj, step_len, iter, 'steepest descent', x_0);
fprintf('Steepest descent with [1.2, 1.2]:\n');
fprintf('The minimum value is at: %i\n',x_p);
fprintf('Step length: %i\n',step_len);
fprintf('No of iterations: %i\n',iter);
[x_p, f_obj, step_len, iter] =  backtracking(x_0, 'newton', step_len_max, rho, c);%Newton's method with [1.2, 1.2]
plot_results(x_p, f_obj, step_len, iter, 'newton', x_0);
fprintf('Newtons method with [1.2, 1.2]:\n');
fprintf('The minimum value is at: %i\n',x_p);
fprintf('Step length: %i\n',step_len);
fprintf('No of iterations: %i\n',iter);

x_0 = [1.2; 1];
[x_p, f_obj, step_len, iter] =  backtracking(x_0, 'steepest descent', ...
                                              step_len_max, rho, c);%Steepest descent with [1.2, 1]
plot_results(x_p, f_obj, step_len, iter, 'steepest descent', x_0);
fprintf('Steepest descent method with [1.2, 1]:\n');
fprintf('Step length: %i\n',step_len);
fprintf('No of iterations: %i\n',iter);
fprintf('The minimum value is at: %i\n',x_p);
[x_p, f_obj, step_len, iter] =  backtracking(x_0, 'newton', step_len_max, rho, c);%Newtons method at [1.2, 1]
plot_results(x_p, f_obj, step_len, iter, 'newton', x_0);
fprintf('Newtons method with [1.2, 1]:\n');
fprintf('The minimum value is at: %i\n',x_p);
fprintf('Step length: %i\n',step_len);
fprintf('No of iterations: %i\n',iter);
end

function [x, f_obj, step_len, iter] = backtracking(x_0, method, ...
                                                      step_len_max, rho, c)
iter = 1;%initializing the iteration
tol = 1e-6;%tolerance
max_iter = 50000;%maximum number of iterations
f_obj = zeros(max_iter, 1);
step_len = step_len_max * ones(max_iter, 1);
step_len(1) = 0;

x = x_0;
f_obj(iter) = func(x);
while norm(gradi(x)) > tol && iter < max_iter%while gradient is greater than tolerance level and iterations less than maximum
  iter = iter + 1;
  p = step_direction(x, method);%get direction based on method
  [step_len(iter), f_obj(iter)] = step_length(x, p, step_len(iter), rho, c);%calculate step length
  x = x + step_len(iter) * p;
end
f_obj = f_obj(1:iter);
step_len = step_len(1:iter);
end

function p = step_direction(x_k, method)
% Return a unit direction of search
if strcmp(method, 'newton')%If its newtons method then the direction is calculated using hessain matric and gradient both
  p = - hessi(x_k)^-1 * gradi(x_k);
else
  p = - gradi(x_k);%Else for steepest descent just consider the negative gradient as the direction
end
p = p / norm(p);%norm
end

function [step_len, f_x_k] = step_length(x_k, p_k, step_len_max, rho, c)%Calculate step length
step_len = step_len_max;%initial step length
f_x_k = func(x_k);
while (func(x_k + step_len * p_k) > ...
      f_x_k + c * step_len * gradi(x_k)' * p_k)%checking first Wolfe condition
  step_len = rho * step_len;%update the step length by multiplying with rho is the condition satisfies
end
end

% Visualize iteration
function plot_results(x_p, f_obj, alpha, iter, method, x_0)
figure;
subplot(1, 2, 1);
plot(1:iter, f_obj);
title(['Rosenbrock function with ', method])
ylabel('f(x)');
xlabel('Iterations');
subplot(1, 2, 2);
plot(1:iter, alpha)
title(['x_0 = ', mat2str(x_0', 3), ', x^* = ', mat2str(x_p', 3)]);
ylabel('Step length');
xlabel('Iterations');
end

% Function to minimize, its gradient and hessian

function f = func(x)%Rosenbrock function
f = 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
end

function gf = gradi(x)%Gradient of Rosenbrock function
gf = [2 * x(1) - 400 * x(1) * (- x(1)^2 + x(2)) - 2;
      200 * (x(2) - x(1)^2)];
end

function hf = hessi(x)%hessian matrix
hf = [2 + 1200 * x(1)^2 - 400 * x(2), -400*x(1);
      -400 * x(1), 200];
end
