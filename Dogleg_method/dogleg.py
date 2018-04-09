import numpy as np
from scipy import optimize as opt
import matplotlib.pylab as plt
import inspect
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plot
import numpy as np
import scipy.linalg
import math

import numpy as np
import scipy.linalg
import warnings
from scipy import optimize as opt
from scipy.optimize._trustregion_dogleg import DoglegSubproblem

Nfeval = 1
__all__ = []
fun = opt.rosen
jac = opt.rosen_der
hess = opt.rosen_hess
hessp = opt.rosen_hess_prod
x0 = np.array([1.2, 1.2])# initial values

def rosenbrock(x, y): #defining the function
    return (1 - x) ** 2 + 100 * ((y - x ** 2)) ** 2


fig = plot.figure()#plotting the rosenbrock function
ax = fig.gca(projection='3d')

s = 0.05
X = np.arange(-2, 2. + s, s)  # setting values for the function
Y = np.arange(-2, 3. + s, s)  # setting values for the function

# Create the mesh grid
X, Y = np.meshgrid(X, Y)

# Rosenbrock function w/ two parameters using numpy Arrays
Z = (1. - X) ** 2 + 100. * (Y - X * X) ** 2

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

# Display the figure
plot.show()

def rosen(X): #Rosenbrock function
    return (1.0 - X[0])**2 + 100.0 * (X[1] - X[0]**2)**2 + \
           (1.0 - X[1])**2 + 100.0 * (X[2] - X[1]**2)**2

class OptimizeWarning(UserWarning): #setting up warning
    pass

def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)

def wrap_function(function, args):#wrapper function
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper

def callback_on_crack(x):#callback to trace the values of each iteration and print
    print(inspect.currentframe().f_back.f_locals)
    print(x)

class Result(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

# standard status messages of optimizers in scipy
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}

class BaseQuadraticSubproblem(object):
    def __init__(self, x, fun, jac, hess=opt.rosen_hess, hessp=opt.rosen_hess_prod):
        self._x = x
        self._f = None
        self._g = None
        self._h = None
        self._g_mag = None
        self._cauchy_point = None
        self._newton_point = None
        self._fun = opt.rosen
        self._jac = opt.rosen_der
        self._hess = opt.rosen_hess
        self._hessp = opt.rosen_hess_prod

    def __call__(self, p):
        return self.fun + np.dot(self.jac, p) + 0.5 * np.dot(p, self.hessp(p))

    @property
    def fun(self):
        #Value of objective function at current iteration.If the function is not set
        if self._f is None:
            self._f = self._fun(self._x)
        return self._f

    @property
    def jac(self):
        #Value of jacobian of objective function at current iteration.If there is no jacobian
        if self._g is None:
            self._g = self._jac(self._x)
        return self._g

    @property
    def hess(self):
        #Value of hessian of objective function at current iteration.If there is no Hessian
        if self._h is None:
            self._h = self._hess(self._x)
        return self._h

    def hessp(self, p):
        # Value of hessian of objective function at current iteration.If there is no Hessian
        if self._hessp is not None:
            return self._hessp(self._x, p)
        else:
            return np.dot(self.hess, p)

    @property
    def jac_mag(self):
        #Magniture of jacobian of objective function at current iteration.If it's not set
        if self._g_mag is None:
            self._g_mag = scipy.linalg.norm(self.jac)
        return self._g_mag

    def get_boundaries_intersections(self, z, d, trust_radius):#getting the boundary intersection based on trust radius
        a = np.dot(d, d)
        b = 2 * np.dot(z, d)
        c = np.dot(z, z) - trust_radius**2
        sqrt_discriminant = math.sqrt(b*b - 4*a*c)
        ta = (-b - sqrt_discriminant) / (2*a)
        tb = (-b + sqrt_discriminant) / (2*a)
        return ta, tb

    def solve(self, trust_radius):
        raise NotImplementedError('The solve method should be implemented by '
                                  'the child class')

def _minimize_trust_region(fun, x0, args=(), jac=opt.rosen_der, hess=opt.rosen_hess, hessp=opt.rosen_hess_prod,
                           subproblem=DoglegSubproblem, initial_trust_radius=3.0,
                           max_trust_radius=1000.0, eta=0.15, gtol=1e-5,
                           maxiter=None, disp=True, return_all=True,
                           callback=callback_on_crack, **unknown_options):
    #the trust region method to solve rosenbrock function with Dogleg sub problem implementation with trust radius
    #and call back enabled
    x0 = np.array([1.2,1.2]) # initial values
    #wrap fucntion
    nfun, fun = wrap_function(fun, args)
    njac, jac = wrap_function(jac, args)
    nhess, hess = wrap_function(hess, args)
    nhessp, hessp = wrap_function(hessp, args)

    # limiting the number of iterations
    if maxiter is None:
        maxiter = len(x0)*200

    # init the search status
    warnflag = 0

    # initializing the search
    trust_radius = initial_trust_radius
    x = x0
    if return_all:
        allvecs = [x]
    m = subproblem(x, fun, jac, hess, hessp)#call to the dogleg sub problem
    k = 0

    # search for the function min
    while True:
        try:
            p, hits_boundary = m.solve(trust_radius)
        except np.linalg.linalg.LinAlgError as e:
            warnflag = 3
            break

        predicted_value = m(p)#predicted value at the proposed point

        #proposed point
        x_proposed = x + p
        m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)

        #calculating the ratio based on actual reduction and predicted reduction
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction

        # updating the trust radius based on rho value
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2*trust_radius, max_trust_radius)

        # if the ratio is high accept the proposed step
        if rho > eta:
            x = x_proposed
            m = m_proposed

        if return_all:
            allvecs.append(x)
        if callback is not None:
            callback(x)#callback to trace the values at the current iteration

        k += 1

        # Stopping condition based on tolerance value
        if m.jac_mag < gtol:
            warnflag = 0
            break

        # Stopping conditions based on maximum number of iterations
        if k >= maxiter:
            warnflag = 1
            break

            # Default status messages
    status_messages = (
        _status_message['success'],
        _status_message['maxiter'],
        'A bad approximation caused failure to predict improvement.',
        'A linalg error occurred, such as a non-psd Hessian.',
    )

    #Scipy display options if no warning
    if disp:
        if warnflag == 0:
            print(status_messages[warnflag])
        else:
            print('Warning: ' + status_messages[warnflag])
        # print("         Current function value: %f")
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % nfun[0])
        print("         Gradient evaluations: %d" % njac[0])
        print("         Hessian evaluations: %d" % nhess[0])

    result = Result(x=x, success=(warnflag == 0), status=warnflag, fun= opt.rosen,
                    jac=m.jac, nfev=nfun[0], njev=njac[0], nhev=nhess[0],
                    nit=k, message=status_messages[warnflag])

def _minimize_dogleg(fun, x0, args=(), jac=opt.rosen_der, hess=opt.rosen_hess,
                     **trust_region_options):
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
                                  subproblem=DoglegSubproblem,
                                  **trust_region_options)

#Dogleg Sub problem
class DoglegSubproblem(BaseQuadraticSubproblem):
    def cauchy_point(self):
        if self._cauchy_point is None:
            g = self.jac
            Bg = self.hessp(g)
            self._cauchy_point = -(np.dot(g, g) / np.dot(g, Bg)) * g #defining the cauchypoint
        return self._cauchy_point

    def newton_point(self):
        if self._newton_point is None:
            g = self.jac
            B = self.hess
            cho_info = scipy.linalg.cho_factor(B)
            self._newton_point = -scipy.linalg.cho_solve(cho_info, g)
        return self._newton_point

    def solve(self, trust_radius):
        p_best = self.newton_point()
        if scipy.linalg.norm(p_best) < trust_radius:
            hits_boundary = False
            return p_best, hits_boundary

        # Compute the Cauchy point.
        p_u = self.cauchy_point()

        # If the Cauchy point is outside the trust region
        # return the point where the path intersects the boundary.
        p_u_norm = scipy.linalg.norm(p_u)
        if p_u_norm >= trust_radius:
            p_boundary = p_u * (trust_radius / p_u_norm)
            hits_boundary = True
            return p_boundary, hits_boundary

        _, tb = self.get_boundaries_intersections(p_u, p_best - p_u,
                                                  trust_radius)
        p_boundary = p_u + tb * (p_best - p_u)
        hits_boundary = True
        return p_boundary, hits_boundary

# def callbackF(Xi):
#     global Nfeval
#     print ('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], rosen(Xi)))
#     Nfeval += 1


# x0 = np.array([4., -2.5])
x0 = np.array([1.2,1.2])
objective = np.poly1d([1.0, -2.0, 0.0])
# print(objective)
# print  ('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))
# result = opt.minimize(opt.rosen, x0, method='dogleg', hess=opt.rosen_hess, jac=opt.rosen_der,callback=callback_on_crack,options={'gtol':1e-6, 'disp': True})
# print (result.x)
# print(result.success)
# print(result.status)
# print(result.message)
# print(result.fun)
# print(result.jac)
# print(result.hess)
#
# x = np.linspace(-3,5,100)
# plt.plot(x,objective(x))
# plt.plot(result.x, objective(result.x),'ro')
# plt.show()
# print(result.message)

_minimize_trust_region(fun,x0,callback=callback_on_crack)
