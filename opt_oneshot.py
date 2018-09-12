#! /usr/bin/python3
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
import sys
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad  # for functions that vectorize over inpu

contourp = 999
g_conv = 1000
x_conv = 1001
w_conv = 1002
a_conv = 1003

# Design variables x
# State variables w
# Minimize cost function I = (w,x) = I(w(x),x) = I(x) w.r.t design variables
# Constraint on state variable (e.g. Flow solution)
#        w = sqrt(x)
# Update through the fixed point  (e.g. Flow solver)
#        w = 0.5(w + x/w)

# Optimization solution for quadratic cost function
# I = (x-1)^2 - x  - w^2
#   = x^2 - 3x + 1 - w^2  --> w^2 = x
#   = x^2 - 3x + 1 - x
#   = x^2 - 4x + 1
# Has a global minimum at x = 2, w = sqrt(2), such that I = -3

# Stepsize when using gradient descent
#step = 1e-1 if quadratic_cost else 1e-2
step = 0.05
n_iterations = 200
# Initial guesses
x0 = 7 # Design variable
w0 = 7 # State variable
a0 = 7 # Adjoint variable

# Quadratic cost has a unique global minimum
# Else sines and cosines are used and have multiple local minima
quadratic_cost = False

# Using Jacobi uses past iteration information for everything
# Else Gauss-Seidel uses the updated state variable information to evaluate the search
# direction of the adjoint and the design variables
jacobi = False

# Scales the state vector update by some amount. Equivalent to lowering the CFL.
# Else just apply the given fixed-point update
scale_dw = True # Use step defined above to rescale the state variable update
scale = step if scale_dw else 1.0


if quadratic_cost:
	# Cost function I = (x-1)^2 - w^2
	def cost(w, x): return (x-1)**2 - w**2 - x
else:
	# Alternate, more non-linear cost function
	# I = sin(x/period) - cos(x/period)
	period = 1
	def cost(w, x): return np.sin(x / (period)) - np.cos(w / (2*period))

# Partial sensitivities of cost function pIpW and pIpX
# Automatic differentiation
pCostpW_ad = grad(cost, 0)
pCostpX_ad = grad(cost, 1)
# Analytical
def pCostpW(w, x): return -2*w
def pCostpX(w, x): return 2*(x-1)

# Constraint used is sqrt(x)
def constraint(w_in, x): return np.sqrt(x);
# Constraint fixed-point w(i+1) = G(w(i),x)
# Babylonian method of evaluating sqrt(x)
def fixedpoint_G(w_in, x): return 0.5*(w_in + x/w_in);
# Sensitivity of the fixed point pGpX and pGpW
# Automatic differentiation
pGpW_ad = grad(fixedpoint_G, 0)
pGpX_ad = grad(fixedpoint_G, 1)

# Analytical
def fp_pGpX(w_in, x): return 0.5*(1.0/w_in);
def fp_pGpW(w_in, x): return 0.5*(1.0 - x/(w_in**2));

def plot_cost_constraint(xmin, xmax, wmin, wmax):
	xplot = np.linspace(xmin, xmax, 100)
	wplot = np.linspace(wmin, wmax, 100)
	Xplot, Wplot = np.meshgrid(xplot, wplot)
	Zplot = cost(Wplot, Xplot)
	plt.figure(contourp); plt.title('Contour plot')
	CS = plt.contour(Xplot, Wplot, Zplot, 24)
	plt.clabel(CS, inline=True, fontsize=10)
	xplot = np.linspace(xmin, xmax, 100)
	wplot = constraint(0, xplot)
	plt.plot(xplot, wplot, '-r', label='Constraint');
	plt.xlabel('x'); plt.ylabel('w')

def oneshot_dwdx(n_iterations, x0, w0, jacobi):
	g_norm = np.zeros(n_iterations)
	x = np.zeros(n_iterations)
	w = np.zeros(n_iterations)
	g_norm[0] = 1;
	x[0] = x0;
	w[0] = w0;

	i = 0
	I = cost(w[i], x[i])
	print("Iteration: %d Cost = %e, w = %e, x = %e" %(i, I, w[i], x[i]))
	for i in range(1,n_iterations):
		# Update constraint through fixed-point
		#w[i] = fixedpoint_G(w[i-1], x[i-1])
		dw = fixedpoint_G(w[i-1], x[i-1]) - w[i-1]
		w[i] = w[i-1] + scale*dw
			
		# If Jacobi, then use previous w[i-1], else use Gauss-Seidel,
		# which uses the updated w[i]
		w_used = w[i-1] if jacobi else w[i]
		x_used = x[i-1]
		# Sensitivities of fixed-point
		pGpX = pGpX_ad(w_used, x_used)
		pGpW = pGpW_ad(w_used, x_used)
		
		# Sensitivities of cost function
		pIpX = pCostpX_ad(w_used, x_used)
		pIpW = pCostpW_ad(w_used, x_used)
		
		# dW(i+1)/dX = pW(i+1)/pX(i) + pW(i+1)/pW(i) * pW(i)/pX(i)
		#            = pW(i+1)/pX(i)
		if jacobi:
			# dW(i+1)/dX = pW(i+1)/pX(i) + pW(i+1)/pW(i) * pW(i)/pX(i)
			#            = pW(i+1)/pX(i)
			#            = pG/pX
			dWdX = pGpX
		else:
			# dW(i+1)/dX = pW(i+1)/pX(i) + pW(i+1)/pW(i+1) * pW(i+1)/pX(i)
			#            = pG/pX(i) + pG/pW(i+1) * pW(i+1)/pX(i)
			#            = pG/pX + pGpW*pGpX
			dWdX = pGpX + pGpW*pGpX

		# Total derivative
		dIdX = pIpX + pIpW * dWdX

		dX = -dIdX
		x[i] = x[i-1] + step*dX

		g_norm[i] = (w[i] - w[i-1])**2 + dX**2
		I = cost(w[i], x[i])
		print("Iteration: %d Cost = %e, w = %e, x = %e" %(i, I, w[i], x[i]))


	return g_norm, x, w


def oneshot_adjoint(n_iterations, x0, w0, a0, jacobi):
	g_norm = np.zeros(n_iterations)
	x = np.zeros(n_iterations)
	w = np.zeros(n_iterations)
	a = np.zeros(n_iterations)
	g_norm[0] = 1;
	x[0] = x0;
	w[0] = w0;
	a[0] = a0;

	i = 0
	I = cost(w[i], x[i])
	print("Iteration: %d Cost = %e, w = %e, x = %e" %(i, I, w[i], x[i]))
	for i in range(1,n_iterations):
		# Update constraint through fixed-point
		#w[i] = fixedpoint_G(w[i-1], x[i-1])
		dw = fixedpoint_G(w[i-1], x[i-1]) - w[i-1]
		w[i] = w[i-1] + scale*dw
			
		# If Jacobi, then use previous w[i-1], else use Gauss-Seidel,
		# which uses the updated w[i]
		w_used = w[i-1] if jacobi else w[i]
		x_used = x[i-1]

		# Sensitivities of fixed-point
		pGpX = pGpX_ad(w_used, x_used)
		pGpW = pGpW_ad(w_used, x_used)

		# Sensitivities of cost function
		pIpX = pCostpX_ad(w_used, x_used)
		pIpW = pCostpW_ad(w_used, x_used)

		# Adjoint 
		a[i] = pGpW*a[i-1] + pIpW;

		# If Jacobi, then use previous w[i-1], else use Gauss-Seidel,
		# which uses the updated w[i]
		a_used = a[i-1] if jacobi else a[i]
		
		# Total derivative
		dIdX = pIpX + pGpX * a_used;

		# Design update
		dX = -dIdX
		x[i] = x[i-1] + step*dX

		I = cost(w[i], x[i])

		g_norm[i] = (w[i] - w[i-1])**2 + dX**2
		print("Iteration: %d Cost = %e, w = %e, x = %e" %(i, I, w[i], x[i]))


	return g_norm, x, w, a

def constraint_zero(w_in, x): return w_in-fixedpoint_G(w_in,x)
pRpW_ad = grad(constraint_zero, 0)
pRpX_ad = grad(constraint_zero, 1)

def lagrangian(w, x, a):
	return cost(w,x) + a*constraint_zero(w, x)
pLpW_ad = grad(lagrangian, 0)
pLpX_ad = grad(lagrangian, 1)
pLpA_ad = grad(lagrangian, 2)

def oneshot_everythingisdesign(n_iterations, x0, w0, a0, jacobi):
	g_norm = np.zeros(n_iterations)
	x = np.zeros(n_iterations)
	w = np.zeros(n_iterations)
	a = np.zeros(n_iterations)
	g_norm[0] = 1;
	x[0] = x0;
	w[0] = w0;
	a[0] = a0;

	i = 0
	dw = constraint_zero(w[i], x[i])
	# L = I(w,x) + a * R(w,x)  --> R(w,x) = w - G(w,x) = 0
	# min L wrt w,x
	# such that R(w,x) = 0
	L = lagrangian(w[i], x[i], a[i])
	# Gradient of Lagrangian
	# dLdW = pIpW + a * (pGpW - 1)
	# dLdX = pIpX + a * (-pGpX)
	# dLda = (w - G) = dw = 0
	print("Iteration: %d Cost = %e, w = %e, x = %e" %(i, L, w[i], x[i]))
	for i in range(1,n_iterations):
		# Update constraint through fixed-point
		#w[i] = fixedpoint_G(w[i-1], x[i-1])

		x_used = x[i-1]
		w_used = w[i-1]
		a_used = a[i-1]
		pIpW = pCostpW_ad(w_used, x_used)
		pIpX = pCostpX_ad(w_used, x_used)
		pGpW = pGpW_ad(w_used, x_used)
		pGpX = pGpX_ad(w_used, x_used)

		# Constraint jacobian
		pRpW = pRpW_ad(w_used, x_used)
		pRpX = pRpX_ad(w_used, x_used)

		# Lagrangian gradient
		dLdW = pIpW + a_used * pRpW
		dLdX = pIpX + a_used * pRpX
		dLdA = constraint_zero(w_used, x_used)

		dLdW = pLpW_ad(w_used, x_used, a_used)
		dLdX = pLpX_ad(w_used, x_used, a_used)
		dLdA = pLpA_ad(w_used, x_used, a_used)

		# KKT Conditions
		kkt_rhs = np.zeros((3,1))
		kkt_rhs[0] = -dLdW
		kkt_rhs[1] = -dLdX
		kkt_rhs[2] = -dLdA

		# Gradient descent by setting the Hessian of the Lagragian to identity
		lagrangian_hessian = np.zeros((3,3))
		lagrangian_hessian[0,0] = 1
		lagrangian_hessian[1,1] = 1

		lagrangian_hessian[2,0] = pRpW
		lagrangian_hessian[2,1] = pRpX

		lagrangian_hessian[0,2] = pRpW
		lagrangian_hessian[1,2] = pRpX

		search_direction = np.linalg.solve(lagrangian_hessian, kkt_rhs)

		print('Matrix ')
		matprint(lagrangian_hessian)
		print('RHS ')
		matprint(kkt_rhs)
		print('Search Direction ')
		matprint(search_direction)
		print('Residual is %e ' %dLdA)

		w[i] = w[i-1] + step*search_direction[0]
		x[i] = x[i-1] + step*search_direction[1]
		a[i] = a[i-1] + search_direction[2]

		L = lagrangian(w[i], x[i], a[i])

		g_norm[i] = dLdW**2+dLdX**2+dLdA**2
		print("Iteration: %d Cost = %e, w = %e, x = %e" %(i, L, w[i], x[i]))


	return g_norm, x, w, a


it = range(n_iterations);

xmin, xmax, ymin, ymax = 0,10,0,10
plot_cost_constraint(xmin, xmax, ymin, ymax)

plt.figure(g_conv); plt.title('Gradient Convergence')
plt.figure(x_conv); plt.title('X Convergence')
plt.figure(w_conv); plt.title('W Convergence')
plt.figure(a_conv); plt.title('A Convergence')

g, x, w = oneshot_dwdx(n_iterations, x0, w0, jacobi)
plt.figure(contourp); plt.plot(x,w,'-bo', label='dwdx method')
plt.figure(g_conv); plt.semilogy(it,g,'-bs', label='dwdx method')
plt.figure(x_conv); plt.plot(it,x,'-bo', label='dwdx method')
plt.figure(w_conv); plt.plot(it,w,'-b^', label='dwdx method')

g, x, w, a = oneshot_adjoint(n_iterations, x0, w0, a0, jacobi)
plt.figure(contourp); plt.plot(x,w,'-go', label='adjoint method')
plt.figure(g_conv); plt.semilogy(it,g,'-gs', label='adjoint method')
plt.figure(x_conv); plt.plot(it,x,'-go', label='adjoint method')
plt.figure(w_conv); plt.plot(it,w,'-g^', label='adjoint method')
plt.figure(a_conv); plt.plot(it,a,'-g*', label='adjoint method')

g, x, w, a = oneshot_everythingisdesign(n_iterations, x0, w0, a0, jacobi)
plt.figure(contourp); plt.plot(x,w,'-mo', label='alldesign method')
plt.figure(g_conv); plt.semilogy(it,g,'-ms', label='alldesign method')
plt.figure(x_conv); plt.plot(it,x,'-mo', label='alldesign method')
plt.figure(w_conv); plt.plot(it,w,'-m^', label='alldesign method')
plt.figure(a_conv); plt.plot(it,a,'-m*', label='alldesign method')


plt.figure(contourp); plt.legend(); plt.xlim([xmin, xmax]); plt.ylim([ymin, ymax]);
plt.figure(g_conv); plt.legend()
plt.figure(x_conv); plt.legend()
plt.figure(w_conv); plt.legend()
plt.figure(a_conv); plt.legend()



plt.draw()
plt.pause(1)
input("<Hit Enter To Close>")
plt.close('all')
