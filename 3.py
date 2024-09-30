#Question 3
# Sahil, 23B0943
# Koushik, 23B1000
# Kishan, 23B1061
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import binom
from scipy.stats import gamma 
from scipy.stats import norm


#---------------------------------------------------->

# Task A
# Loading the data
D = np.loadtxt('3.data')

# Computing the values
moment1 = np.mean(D)
moment2 = np.mean(D**2)

# Print the values
print(f"First_moment = {moment1}")
print(f"Second_moment = {moment2}")

#---------------------------------------------------->

# Task B

# Parameters for finding the mode
counts, bin_edges = np.histogram(D, bins=100, density=True)

# Mode is the midpoint of the highest bin 
mode_index = np.argmax(counts)
mode_value = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2

# Print the value of the mode
print(f"Graphical guess for the mode: {mode_value}")

# Plot the histogram
plt.figure(figsize = (8,6))
plt.hist(D, bins = 100, density = True) #density = True make the y axis represent the probability  
                                                            # we can adjust the no.of bins as we want 
# Add labels
plt.title('Dataset Histogram', fontsize=16)
plt.xlabel('Data Values', fontsize = 14)
plt.ylabel('Probability', fontsize = 14)

# Save the plot
plt.savefig('3b.png')


#------------------------------------------------>

#Task C
#first and second moments of the dataset
moment1_sample = moment1
moment2_sample = moment2

# Define the equations for the first two moments of Bin(n, p)
def equations(params):
    n, p = params
    moment1_bin = n * p  # First moment of Bin(n, p)
    moment2_bin = n * p * (1 - p) + (n * p)**2  # Second moment of Bin(n, p)
    return [moment1_bin - moment1_sample, moment2_bin - moment2_sample]

# Use fsolve to find n and p
n_guess, p_guess = 10, 0.5  # Initial guesses for n and p
n_opt, p_opt = fsolve(equations, [n_guess, p_guess])

# Round n_opt to nearest integer
n_opt = int(round(n_opt))

# Print the values of n* and p*
print(f"Optimized values: n* = {n_opt}, p* = {p_opt}")

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(D, bins=100,  density=True, alpha=0.6, label="Dataset Histogram")

# Plot the binomial distribution Bin(n_opt, p_opt) on top of the histogram
x = np.linspace(np.min(D), np.max(D), 1000)  # Generate values to plot the binomial distribution
binom_pmf = binom.pmf(np.arange(n_opt+1), n_opt, p_opt)
plt.plot(np.arange(n_opt+1), binom_pmf, 'o-', label=f"Bin({n_opt}, {p_opt:.2f})", color='red')

# Add labels
plt.title('Histogram with Best-Fit Binomial Distribution', fontsize=16)
plt.xlabel('Data Values', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.legend()

# Save the plot
plt.savefig('3b_binomial_fit.png')

#---------------------------------------------------------->

# Task D
# First and second moments of the dataset
moment1_sample = moment1
moment2_sample = moment2

# Define equations for the first two moments of Gamma(k, theta)
def equations(params):
    k , theta = params
    moment1_gamma = k * theta  # First moment of Gamma(k, theta)
    moment2_gamma = k * theta**2 + (k * theta)**2  # Second moment of Gamma(k, theta)
    return [moment1_gamma - moment1_sample, moment2_gamma - moment2_sample]

# Use fsolve to find k and theta
k_guess, theta_guess = 2, 2  # Initial guesses for α and β
k_opt, theta_opt = fsolve(equations, [k_guess, theta_guess])

# Print the values of k* and theta*
print(f"Optimized values: k* = {k_opt}, theta* = {theta_opt}")

# Plot the histogram of the dataset
plt.figure(figsize=(8, 6))
plt.hist(D, bins=100,  density=True, alpha=0.6, label="Dataset Histogram")

# Plot the Gamma distribution Gamma(k*, theta*) on top of the histogram
x = np.linspace(np.min(D), np.max(D), 1000)  # Generate values to plot the Gamma distribution
gamma_pdf = gamma.pdf(x, a=k_opt, scale=theta_opt) 
plt.plot(x, gamma_pdf, label=f"Gamma(k = {k_opt:.2f}, theta = {theta_opt:.2f})", color='red')

# Add labels and title
plt.title('Histogram with Best-Fit Gamma Distribution', fontsize=16)
plt.xlabel('Data Values', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.legend()

# Save the plot
plt.savefig('3d.png')


#-------------------------------------------------------------->

#Task E

#from scipy.special import gammaln

# Best-fit parameters for Binomial and Gamma distributions
# for the given dataset found in the previous tasks are n_opt , p_opt , k_opt , theta_opt
# we are going to use them in this task

# Number of samples
n_samples = len(D)

# Compute the average log-likelihood for the Binomial distribution
D_binom = np.round(D) # Rounding
log_likelihood_binom = binom.logpmf(D_binom, n=n_opt, p=p_opt) # Log value for each datapoint
avg_log_likelihood_binom = np.mean(log_likelihood_binom) # Average likelihood value


# Compute the average log-likelihood for the Gamma distribution
log_likelihood_gamma = gamma.logpdf(D, a=k_opt, scale=theta_opt) # Logvalue for each datapoint
avg_log_likelihood_gamma = np.mean(log_likelihood_gamma) # Average likelihood value


# Print the results
print(f"Average Log-Likelihood (Binomial): {avg_log_likelihood_binom}")
print(f"Average Log-Likelihood (Gamma): {avg_log_likelihood_gamma}")

# Determine which distribution is a better fit
if avg_log_likelihood_binom > avg_log_likelihood_gamma:
    print("The Binomial distribution is a better fit.")
else:
    print("The Gamma distribution is a better fit.")


#--------------------------------------------------------------------->

# Task F

# Number of samples
n = len(D)

# Compute sample moments (1st, 2nd, 3rd, 4th)
moment1 = np.mean(D)
moment2 = np.mean(D**2)
moment3 = np.mean(D**3)
moment4 = np.mean(D**4)

# Define the GMM moments in terms of the parameters mu1, mu2, p1 (with p2 = 1 - p1)
def gmm_moments(params):
    mu1, mu2, p1  = params
    p2 = 1 - p1
    # Compute the first four moments of the GMM
    gmm_1 = p1 * mu1 + p2 * mu2
    gmm_2 = p1 * (mu1**2 + 1) + p2 * (mu2**2 + 1)  # Variance is 1 for each component
    gmm_3 = p1 * (mu1**3 + 3 * mu1) + p2 * (mu2**3 + 3 * mu2)
    # gmm_4 = p1 * (mu1**4 + 6 * mu1**2 + 3) + p2 * (mu2**4 + 6 * mu2**2 + 3)  
    # we don't need four moments as p1 and p2 constrained as their sum is 1 , so essentially we have only three parameters
    return [gmm_1 - moment1, gmm_2 - moment2, gmm_3 - moment3]

# Initial guess for mu1, mu2, p1
initial_guess = [np.min(D), np.max(D), 0.5]

# Use fsolve to find the best-fit parameters
mu1_opt, mu2_opt, p1_opt = fsolve(gmm_moments, initial_guess)
p2_opt = 1 - p1_opt
print(f"Optimal parameters: mu1 = {mu1_opt}, mu2 = {mu2_opt}, p1 = {p1_opt}, p2 = {p2_opt}")

# Plot the histogram of the data
plt.hist(D, bins=100, density=True, alpha=0.6, color='blue', label = 'Dataset')

# Generate the GMM PDF
x_values = np.linspace(np.min(D), np.max(D), 1000)
gmm_pdf = p1_opt * norm.pdf(x_values, loc=mu1_opt, scale=1) + p2_opt * norm.pdf(x_values, loc=mu2_opt, scale=1)

# Plot and label the GMM PDF 
plt.plot(x_values, gmm_pdf, label='GMM Fit', color='green')
plt.xlabel('Data Values')
plt.ylabel('Probability')
plt.legend()
plt.title('GMM Fit to the Data')
plt.savefig('3f.png')

# Compute the average negative log-likelihood for GMM
log_likelihood_gmm = np.log(p1_opt * norm.pdf(D, loc=mu1_opt, scale=1) + p2_opt * norm.pdf(D, loc=mu2_opt, scale=1)) # Log value
avg_neg_log_likelihood_gmm = -np.mean(log_likelihood_gmm) # Average negative likelihood
print(f"Average Negative Log-Likelihood (GMM): {avg_neg_log_likelihood_gmm}")


#------------------------------------------------------------------------->