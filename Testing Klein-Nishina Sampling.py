import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import electron_mass, c
from scipy import integrate

def klein_nishina_polarized(cos_theta, energy_kev, phi):
    """
    Calculate the Klein-Nishina differential cross section for polarized photons.
    
    Parameters:
    cos_theta: cosine of scattering angle theta
    energy_kev: incident photon energy in keV
    phi: azimuthal angle in radians
    """
    # Convert energy to dimensionless parameter
    epsilon = energy_kev / 511.0  # Energy in units of electron rest mass
    
    # Calculate sin theta
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Calculate ratio of scattered to incident photon energy
    epsilon_prime = 1 / (1 + epsilon * (1 - cos_theta))
    
    # Calculate the unpolarized part
    term1 = epsilon_prime**2 * (epsilon/epsilon_prime + epsilon_prime/epsilon - sin_theta**2 * np.cos(phi)**2)
    
    return term1

def rejection_sampling(n_samples, energy_kev):
    """
    Perform rejection sampling to generate scattered angles from Klein-Nishina distribution.
    
    Parameters:
    n_samples: number of samples to generate
    energy_kev: incident photon energy in keV
    
    Returns:
    theta_accepted: array of accepted theta values
    phi_accepted: array of accepted phi values
    """
    # Initialize arrays to store accepted values
    theta_accepted = []
    phi_accepted = []
    
    # Find approximate maximum of distribution for given energy
    cos_theta_grid = np.linspace(-1, 1, 1000)
    phi_grid = np.linspace(0, 2*np.pi, 1000)
    max_val = 0
    
    for cos_theta in cos_theta_grid:
        for phi in phi_grid:
            val = klein_nishina_polarized(cos_theta, energy_kev, phi)
            if val > max_val:
                max_val = val
    
    # Add safety factor to maximum
    max_val *= 1.1
    
    # Perform rejection sampling
    accepted = 0
    iterations = 0
    max_iterations = n_samples * 1000  # Safety limit
    
    while accepted < n_samples and iterations < max_iterations:
        # Generate random angles
        cos_theta = np.random.uniform(-1, 1)
        phi = np.random.uniform(0, 2*np.pi)
        
        # Calculate probability
        prob = klein_nishina_polarized(cos_theta, energy_kev, phi)
        
        # Accept or reject
        if np.random.uniform(0, max_val) < prob:
            theta_accepted.append(np.arccos(cos_theta))
            phi_accepted.append(phi)
            accepted += 1
        
        iterations += 1
    
    return np.array(theta_accepted), np.array(phi_accepted)

# Test the sampling
energy_kev = 100  # Test with 100 keV photons
n_samples = 10000

# Perform sampling
theta_samples, phi_samples = rejection_sampling(n_samples, energy_kev)

# Create visualization
plt.figure(figsize=(15, 5))

# Plot theta distribution
plt.subplot(131)
plt.hist(theta_samples, bins=50, density=True, alpha=0.7)
plt.xlabel('Theta (radians)')
plt.ylabel('Density')
plt.title('Theta Distribution')

# Plot phi distribution
plt.subplot(132)
plt.hist(phi_samples, bins=50, density=True, alpha=0.7)
plt.xlabel('Phi (radians)')
plt.ylabel('Density')
plt.title('Phi Distribution')

# Create 2D histogram
plt.subplot(133)
plt.hist2d(theta_samples, phi_samples, bins=50, cmap='viridis')
plt.xlabel('Theta (radians)')
plt.ylabel('Phi (radians)')
plt.title('2D Distribution')
plt.colorbar(label='Counts')

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Number of accepted samples: {len(theta_samples)}")
print(f"Mean theta: {np.mean(theta_samples):.3f} radians")
print(f"Mean phi: {np.mean(phi_samples):.3f} radians")

#make a plot of the Klein-Nishina differential cross section
theta = np.linspace(0, np.pi, 1000)
energy_kev = 100
phi = 0
differential_cross_section = klein_nishina_polarized(np.cos(theta), energy_kev, phi)

plt.figure()
plt.plot(theta, differential_cross_section)
plt.xlabel('Theta (radians)')
plt.ylabel('Differential Cross Section')
plt.title('Klein-Nishina Differential Cross Section')
plt.show()


