import numpy as np
import matplotlib.pyplot as plt

# Constants
r0 = 2.8179403227e-15  # classical electron radius in meters
epsilon_0 = 0.4  # Initial photon energy (in units of electron rest energy)
num_samples = 100000  # Number of samples to generate

# Functions for Klein-Nishina sampling
def differential_cross_section(theta, phi, epsilon_0):
    epsilon = epsilon_0 / (1 + epsilon_0 * (1 - np.cos(theta)))
    return 0.5 * r0**2 * (epsilon / epsilon_0)**2 * (
        epsilon_0 / epsilon + epsilon / epsilon_0 - 2 * np.sin(theta)**2 * np.cos(phi)**2
    )* (np.sin(theta))

def total_cross_section(epsilon_0):
    term1 = (1 + epsilon_0) / epsilon_0**3
    term2 = (2 * epsilon_0 * (1 + epsilon_0)) / (1 + 2 * epsilon_0)
    term3 = np.log(1 + 2 * epsilon_0)
    term4 = (1 + 3 * epsilon_0) / (1 + 2 * epsilon_0)**2
    sigma_kn = 2 * np.pi * r0**2 * (
        term1 * (term2 - term3) + term3 / (2 * epsilon_0) - term4
    )
    return sigma_kn

def klein_nishina_probability(theta, phi, epsilon_0):
    dsigma_domega = differential_cross_section(theta, phi, epsilon_0)
    sigma_kn = total_cross_section(epsilon_0)
    return dsigma_domega / sigma_kn

def rejection_sampling(epsilon_0, num_samples):
    samples = []
    max_prob = 1  # Set a suitable max probability; adjust if necessary for efficiency
    
    while len(samples) < num_samples:
        theta = np.random.uniform(0, np.pi)     # Theta from 0 to π
        phi = np.random.uniform(0, 2 * np.pi)   # Phi from 0 to 2π
        p_theta_phi = klein_nishina_probability(theta, phi, epsilon_0)
        u = np.random.uniform(0, max_prob)
        
        if u < p_theta_phi:
            samples.append((theta, phi))
    
    return samples

# Generate samples
samples = rejection_sampling(epsilon_0, num_samples)
thetas, phis = zip(*samples)

# Plot theta distribution
plt.figure(figsize=(12, 5))

# Theta histogram
plt.subplot(1, 2, 1)
plt.hist(thetas, bins=50, density=True, color='skyblue', edgecolor='black')
plt.xlabel(r"$\theta$ (radians)")
plt.ylabel("Probability Density")
plt.title(r"Distribution of $\theta$")

# Phi histogram
plt.subplot(1, 2, 2)
plt.hist(phis, bins=50, density=True, color='salmon', edgecolor='black')
plt.xlabel(r"$\phi$ (radians)")
plt.ylabel("Probability Density")
plt.title(r"Distribution of $\phi$")

plt.tight_layout()
plt.show()

plt.savefig('Klein-Nishina-Sampling_theta_and_phi_histograms.png')


# Convert spherical coordinates to Mollweide coordinates
x = np.array(phis) - np.pi  # Shift phi to range [-pi, pi] for Mollweide projection
y = np.pi/2 - np.array(thetas)  # Convert theta to "latitude" for Mollweide projection

# Plot heatmap on spherical (Mollweide) plot
plt.figure(figsize=(10, 5))
plt.subplot(111, projection="mollweide")
plt.hexbin(x, y, gridsize=100, cmap='plasma', mincnt=1)
plt.colorbar(label='Sample Density')
plt.xlabel(r"$\phi$ (radians)")
plt.ylabel(r"$\theta$ (radians)")
plt.title("Heatmap of Sampled $(\theta, \phi)$ on a Spherical (Mollweide) Plot")
plt.grid(True)
plt.show()

plt.savefig('Klein-Nishina-Sampling_spherical distribution.png')

#analytically plot the KN distribution using a molleweide projection
# Generate grid of theta and phi values
theta_vals = np.linspace(0, np.pi, 300)    # theta from 0 to π
phi_vals = np.linspace(0, 2 * np.pi, 300)  # phi from 0 to 2π
theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)

# Calculate probability density over the grid
P_grid = klein_nishina_probability(theta_grid, phi_grid, epsilon_0)

# Convert to Mollweide projection coordinates
x = phi_grid - np.pi
y = np.pi / 2 - theta_grid

# Plot the theoretical probability density
plt.figure(figsize=(10, 5))
plt.subplot(111, projection="mollweide")
plt.pcolormesh(x, y, P_grid, cmap='plasma', shading='auto')
plt.colorbar(label='Analytical Probability Density')
plt.xlabel(r"$\phi$ (radians)")
plt.ylabel(r"$\theta$ (radians)")
plt.title("Analytical Probability Density $P(\\theta, \\phi)$ on a Spherical (Mollweide) Plot")
plt.grid(True)
plt.show()

plt.savefig('Klein-Nishina-Sampling_analytical distribution.png')

