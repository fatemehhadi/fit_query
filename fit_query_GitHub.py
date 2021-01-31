import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv    

def Gauss(x, mu, sigma, intensity = 1):
	# x is an array
	# mu is the expected value
	# sigma is the square root of the variance
	# intensity is a multiplication factor
	# This def returns the Gaussian function of x
	gauss_distribution = intensity/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)
	return gauss_distribution

def least_sq(y, X):
	# sample_spectrum (unknown spectrum): array of w values.
	# components (known spectra): array of n (number of components) columns with w values.
	# This def returns an array of n values. Each value is the similarity score for the sample_spectrum and a component spectrum.
	theta = np.dot(inv(np.dot(X.T, X)) , np.dot(X.T, y))
	return theta

def gradientDescent_intercept(X, y, theta, alpha, iterations, lambda_gd):
	# number of training examples
	y_size = np.shape(y)
	m = y_size[0]
	J_history = np.zeros(iterations)
	for iter in range(iterations):
		h = np.dot(X,theta)
		temp0 = theta
		temp0[0] = 0
		J = np.sum(np.dot((h-y).T,(h-y)))/(2*m)+lambda_gd*np.sum(np.dot(temp0.T,temp0))/(2*m)
		temp = np.dot(lambda_gd,theta)/m
		temp[0] = 0
		grad = np.dot(X.T,(h-y))/m+temp
		theta = theta-np.dot(alpha,grad)
		# Save the cost J in every iteration    
		J_history[iter] = J;
	return (theta, J_history)

def costFunctionReg(theta, X, y, lambda_gd):
	X_size = np.shape(X)
	m = X_size[0]
	n = X_size[1]
	J = 0
	grad = np.zeros(n)
	theta = np.reshape(theta, (n, 1))
	h = np.dot(X,theta)
	temp0 = theta
	temp0[0] = 0
	J = np.sum(np.dot((h-y).T,(h-y)))/(2*m)+lambda_gd*np.sum(np.dot(temp0.T,temp0))/(2*m)
	temp = np.dot(lambda_gd,theta)/m
	temp[0] = 0
	grad = np.dot(X.T,(h-y))/m+temp
	return (J, grad)

# X-axis (Wavelengths)
wavelength_range =  np.linspace(100, 200, 1000)

# Four different components
# Component A
mu_a1 = 135
sigma_a1 = 2
intensity_a1 = 1
mu_a2 = 185
sigma_a2 = 2
intensity_a2 = 0.4
gauss_a =  Gauss(wavelength_range, mu_a1, sigma_a1, intensity_a1) + Gauss(wavelength_range, mu_a2, sigma_a2, intensity_a2)

# Component B
mu_b = 150
sigma_b = 15
intensity_b = 1
gauss_b = Gauss(wavelength_range, mu_b, sigma_b, intensity_b)

# Component C
mu_c1 = 120
sigma_c1 = 2
intensity_c1 = 0.15
mu_c2 = 165
sigma_c2 = 8
intensity_c2 = 1
gauss_c = Gauss(wavelength_range, mu_c1, sigma_c1, intensity_c1) + Gauss(wavelength_range, mu_c2, sigma_c2, intensity_c2)

# Component D
mu_d1 = 115
sigma_d1 = 5
intensity_d1 = 1
mu_d2 = 140
sigma_d2 = 5
intensity_d2 = 0.85
gauss_d = Gauss(wavelength_range, mu_d1, sigma_d1, intensity_d1) + Gauss(wavelength_range, mu_d2, sigma_d2, intensity_d2)

# Spectra normalization:
component_a = gauss_a/np.max(gauss_a)
component_b = gauss_b/np.max(gauss_b)
component_c = gauss_c/np.max(gauss_c)
component_d = gauss_d/np.max(gauss_d)

# Generate the components matrix
components = np.array([component_a, component_b, component_c, component_d])

# Rename the library spectra
X = components
X = X.T
X_size = np.shape(X)
m = X_size[0]
X = np.c_[np.ones(m), X] # Add intercept term to X
X_size = np.shape(X)
n = X_size[1]
print("np.shape(X)",np.shape(X))

# What concentrations we want them to have in our query spectrum:
c_a = 0.25
c_b = 0.7
c_c = -0.1
c_d = 0.35

# Let's build the spectrum to be studied: The query spectrum
query_spectrum = c_a * component_a + c_b * component_b + c_c *component_c + c_d *component_d

# Let's add it some noise for a bit of realism:
query_spectrum = query_spectrum +  np.random.normal(0, 0.02, len(wavelength_range))
query_spectrum = np.reshape(query_spectrum, (m, 1))

# Rename the query spectrum
y = query_spectrum
y = np.reshape(y, (m, 1))
print("np.shape(y)",np.shape(y))

# Apply Least squares
# It will not constraint the coefficients
theta = least_sq(y, X)

# Set gradient descent parameters
initial_theta = np.zeros((n, 1))
lambda_gd = 0;
iterations = 10000;
alpha = 0.5;

# Run gradient descent
# It will not constraint the coefficients
(theta_gradientDescent, J_history) = gradientDescent_intercept(X, y, initial_theta, alpha, iterations, lambda_gd)

# Run fmin_tnc
# It will constraint the coefficients to be only positive
initial_theta[2] = 1
bounds = [[0, 1]] * n
(theta_fmin_tnc,nfeval,rc) = scipy.optimize.fmin_tnc(costFunctionReg, initial_theta, fprime=None, args=(X, y, lambda_gd), approx_grad=0, bounds=bounds)
# Alternatively fmin_l_bfgs_b can be used
#(theta_fmin_tnc,nfeval,rc) = scipy.optimize.fmin_l_bfgs_b(costFunctionReg, initial_theta, fprime=None, args=(X, y, lambda_gd), approx_grad=0, bounds=bounds)
print("theta_fmin_tnc",theta_fmin_tnc)
np.savetxt('theta_fmin_tnc.txt',theta_fmin_tnc)

# Plot the library spectra
plt.plot(wavelength_range, component_a, label = 'Component 1')
plt.plot(wavelength_range, component_b, label = 'Component 2')
plt.plot(wavelength_range, component_c, label = 'Component 3')
plt.plot(wavelength_range, component_d, label = 'Component 4')
plt.title('Library Spectra', fontsize = 15)
plt.xlabel('wavenumber', fontsize = 15)
plt.ylabel('normalized amplititude', fontsize = 15)
plt.legend()
plt.show()

# Plot the query spectrum
plt.plot(wavelength_range, query_spectrum, color = 'black', label = 'Mixture spectrum with noise')
plt.title('Query Spectrum', fontsize = 15)
plt.xlabel('wavenumber', fontsize = 15)
plt.ylabel('normalized amplititude',  fontsize = 15)
plt.show()

# And plot the results
plt.plot(wavelength_range, query_spectrum, color = 'black', label = 'Mix spectrum' ) # Plot the unknown spectrum
print("np.arange(len(theta)-2)",np.arange(len(theta)-2))
for i in range(1,n):
    plt.plot(wavelength_range, theta_fmin_tnc[i]*components[i-1], label = 'c' + str(i)+ ' =               ' + str(np.round(theta_fmin_tnc[i], 3)))
plt.plot(wavelength_range, np.dot(X,theta), color = 'green', linewidth = 2, label = 'normal equation') # Plot the calculated spectrum
plt.plot(wavelength_range, np.dot(X,theta_gradientDescent), color = 'blue', linewidth = 2, label = 'gradient descent') # Plot the calculated spectrum
plt.plot(wavelength_range, np.dot(X,theta_fmin_tnc), color = 'cyan', linewidth = 2, label = 'fmin_tnc') # Plot the calculated spectrum
plt.title('Query Spectrum and Calculated Components', fontsize = 15)
plt.xlabel('wavenumber', fontsize = 15)
plt.ylabel('normalized amplititude', fontsize = 15)
plt.legend()
plt.show()