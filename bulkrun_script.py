
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.optimize import curve_fit
import time



# Constants
ELECTRON_MASS_ENERGY = 511  # keV

def rotate_vector(vector,polarisation, theta, phi):
    """
    Rotate a 3D vector by the polar angle (theta) and azimuthal angle (phi).
    
    :param vector: The original vector to rotate.
    :param polarisation: The polarisation direction, considered the x axis
    :param theta: The polar angle (rotation around y-axis, the direction perp to both propagation and polarisation).
    :param phi: The azimuthal angle (rotation around z-axis, the propagation direction itself).
    :return: Rotated vector.
    """

    # Define rotation matrices
    def rotation_matrix_around_axis(axis, angle):
        """
        Creates a rotation matrix to rotate 'angle' radians around 'axis'.
        """
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle
        x, y, z = axis
        
        # Rotation matrix based on Rodrigues' rotation formula
        R = np.array([
            [cos_angle + x*x*one_minus_cos, x*y*one_minus_cos - z*sin_angle, x*z*one_minus_cos + y*sin_angle],
            [y*x*one_minus_cos + z*sin_angle, cos_angle + y*y*one_minus_cos, y*z*one_minus_cos - x*sin_angle],
            [z*x*one_minus_cos - y*sin_angle, z*y*one_minus_cos + x*sin_angle, cos_angle + z*z*one_minus_cos]
        ])
        return R
    
    #generate y axis (preferred direction) of photon
    y_photon = np.cross(vector, polarisation)
    y_photon/=np.linalg.norm(y_photon)  # Ensure right-handed system for the photon frame
    
    #generate rotation matrices
    R_phi = rotation_matrix_around_axis(vector, phi)
    R_theta = rotation_matrix_around_axis(y_photon, theta)

    #apply rotations
    rotated_vector = R_phi@R_theta@vector
    
    return rotated_vector

def generate_photon(energy, polarisation_generator, incidence_angle, incidence_point):
    incidence_direction= np.array([np.sin(incidence_angle[0])*np.cos(incidence_angle[1]), np.sin(incidence_angle[0])*np.sin(incidence_angle[1]), np.cos(incidence_angle[0])])
    incident_polarisation = np.cross(incidence_direction, polarisation_generator)/np.linalg.norm(np.cross(incidence_direction, polarisation_generator))
    return {
        'energy': energy,
        'polarisation': incident_polarisation,
        'direction': incidence_direction,
        'position': np.array(incidence_point)
    }

def simulate_interaction_position(photon, detector_size,interaction_number=1):
    attenuation_coeff = 2.64  # cm^-1, example value
    distance = expon.rvs(scale=1/attenuation_coeff)
    new_position = photon['position'] + distance * photon['direction']

    if(interaction_number == 1):
        if (-detector_size[0]/6 <= new_position[0] <= detector_size[0]/6 and
            -detector_size[0]/6 <= new_position[1] <= detector_size[1]/6 and
            0 <= new_position[2] <= detector_size[2]):
            return new_position
        return None
   # print(f"direction: {photon['direction']}")
    #print(f"New position: {new_position}")
    if(interaction_number==2):
        if (-detector_size[0]/2 <= new_position[0] <= detector_size[0]/2 and
            -detector_size[0]/2 <= new_position[1] <= detector_size[1]/2 and
            0 <= new_position[2] <= detector_size[2]):
            return new_position
        return None

# Constants
r0 = 2.8179403227e-15  # classical electron radius in meters

def differential_cross_section(theta, phi, epsilon_0):
    """
    Calculate the differential cross-section dσ/dΩ for a polarized photon.
    
    Parameters:
    theta (float): Scattering angle in radians.
    phi (float): Azimuthal angle in radians.
    epsilon_0 (float): Initial photon energy (dimensionless, as a multiple of electron rest energy).
    
    Returns:
    float: Differential cross-section dσ/dΩ.
    """
    epsilon = epsilon_0 / (1 + epsilon_0 * (1 - np.cos(theta)))
    return 0.5 * r0**2 * (epsilon / epsilon_0)**2 * ( epsilon_0 / epsilon + epsilon / epsilon_0 - 2 * np.sin(theta)**2 * np.cos(phi)**2)*(np.sin(theta))

def total_cross_section(epsilon_0):
    """
    Calculate the total Klein-Nishina cross-section σ_KN.
    
    Parameters:
    epsilon_0 (float): Initial photon energy (dimensionless, as a multiple of electron rest energy).
    
    Returns:
    float: Total cross-section σ_KN.
    """
    term1 = (1 + epsilon_0) / epsilon_0**3
    term2 = (2 * epsilon_0 * (1 + epsilon_0)) / (1 + 2 * epsilon_0)
    term3 = np.log(1 + 2 * epsilon_0)
    term4 = (1 + 3 * epsilon_0) / (1 + 2 * epsilon_0)**2
    
    sigma_kn = 2 * np.pi * r0**2 * (term1 * (term2 - term3) + term3 / (2 * epsilon_0) - term4)
    
    return sigma_kn

def klein_nishina_probability(theta, phi, epsilon_0):
    """
    Calculate the normalized Klein-Nishina probability density P(θ, φ).
    
    Parameters:
    theta (float): Scattering angle in radians.
    phi (float): Azimuthal angle in radians.
    epsilon_0 (float): Initial photon energy (dimensionless, as a multiple of electron rest energy).
    
    Returns:
    float: Normalized Klein-Nishina probability density P(θ, φ).
    """
    dsigma_domega = differential_cross_section(theta, phi, epsilon_0)
    sigma_kn = total_cross_section(epsilon_0)
    return dsigma_domega / sigma_kn


def compton_scattering(photon, max_iterations=10000):
    start_time = time.time()
    energy = photon['energy']
    
    for i in range(max_iterations):
        # Randomly sample theta and phi
        theta = np.random.uniform(0, np.pi)     # Theta from 0 to π
        phi = np.random.uniform(0, 2 * np.pi)   # Phi from 0 to 2π

        # Calculate P(theta, phi)
        p_theta_phi = klein_nishina_probability(theta, phi, energy / ELECTRON_MASS_ENERGY)

        # Random threshold for rejection
        u = np.random.uniform(0,1)
        
        # Accept the sample if u < P(theta, phi)
        if u < p_theta_phi:
            break
        else:
           # print(f"Warning: Compton scattering did not converge after {max_iterations} iterations")
            return None, None, None
    
    # Calculate new energy
    energy_ratio = 1 / (1 + (energy / ELECTRON_MASS_ENERGY) * (1 - np.cos(theta)))
    new_energy = energy * energy_ratio
    #print(photon)
    new_direction = rotate_vector(photon['direction'],photon['polarisation'], theta, phi)
    
    scattered_photon = photon.copy()
    scattered_photon['energy'] = new_energy
    scattered_photon['direction'] = new_direction
    
    end_time = time.time()
    #print(f"Compton scattering time: {end_time - start_time:.4f} seconds")
 #   print(f"Rejection sampling iterations: {i+1}")
  #  print(f"Scattering angle (theta): {theta:.4f} radians")
   # print(f"Azimuthal angle (phi): {phi:.4f} radians")
    #print(f"Energy before: {energy:.2f} keV, Energy after: {new_energy:.2f} keV")
    
    return scattered_photon, theta, phi

def get_pixel_number(interaction_postion,detector_size):
    x = interaction_postion[0]
    y = interaction_postion[1]
    z = interaction_postion[2]
    pixel_number = 0

    if z < detector_size[2]:
        if x < -detector_size[0]/6 :
            if y< -detector_size[1]/6:
                pixel_number = 5
            elif y < detector_size[1]/6:
                pixel_number = 4
            else :
                pixel_number= 3
        elif x < detector_size[0]/6:
            if y< -detector_size[1]/6:
                pixel_number = 6
            elif y < detector_size[1]/6:
                pixel_number = -1
            else :
                pixel_number= 2
        else :
            if y< -detector_size[1]/6:
                pixel_number = 7
            elif y < detector_size[1]/6:
                pixel_number = 0
            else :
                pixel_number= 1
    else :
        pixel_number = -2
    
    return pixel_number

def get_phi_value(pixel_number):
    if pixel_number == 0:
        return 0
    elif pixel_number == 1:
        return np.pi/4
    elif pixel_number == 2:
        return 2*np.pi/4
    elif pixel_number == 3:
        return 3*np.pi/4
    elif pixel_number == 4:
        return 4*np.pi/4
    elif pixel_number == 5:
        return 5*np.pi/4
    elif pixel_number == 6:
        return 6*np.pi/4
    elif pixel_number == 7:
        return 7*np.pi/4
    else:
        return None
    

def simulate_detector(num_photons, detector_size, initial_energy, incidence_angle,randomise_incidence_point=False,PF=1,preferred_angle= np.pi/2):
    '''''''''

    For this simulator, I will implement polarisation by generating a perpendicular vector to the direction of the photon via crossing it with a vector in the plane of the detector
    By default I consider the 'polarisation generator' to be along the Y axis of the detector. Thus the polarisation will be along the X axis for normal incidence
    This may seem like a convoluted way of achieving polarisation, but it is a general way to achieve polarisation for any incidence angle.
    Coincidentally, by the nature of KN Cross section this will end up giving us the preferred scattering angle. Hence I will call it preferred angle in the params

    '''''''''
    polarisation_generator=[np.cos(preferred_angle),np.sin(preferred_angle),0]
    phi_values=[]
    theta_values=[]
    for i in range(num_photons):
        start_time = time.time()

        if i % 100000==0:
            time.sleep(20)

        print(f"\nSimulating photon {i+1}/{num_photons}")
        
        # Generate photon at the face of the detector
        if randomise_incidence_point==False:
            initial_position = np.array([0,0,0])
        else :
            initial_position = np.array([np.random.uniform(-detector_size[0]/6, detector_size[0]/6),
                                         np.random.uniform(-detector_size[1]/6, detector_size[1]/6),0])

        #rejection sampling for polarisation fraction
        if (np.random.uniform(0,1) < PF):
            photon = generate_photon(initial_energy, polarisation_generator, incidence_angle, initial_position)
        else:
            #for less than full PA we have to pick a random angle on the x-y plane to generate a random polarisation vector perp to propagation direction
            random_angle = np.random.uniform(0,2*np.pi)
            polarisation_generator=[np.cos(random_angle),np.sin(random_angle),0]
            photon = generate_photon(initial_energy, polarisation_generator, incidence_angle, initial_position)
        
        print(f"Initial position: {photon['position']}")
        
        # First interaction
        interaction_pos = simulate_interaction_position(photon, detector_size,interaction_number=1)
        if interaction_pos is None:
            print(f"Photon escaped without interacting. Final position: {photon['position']}")
            continue
        
        photon['position'] = interaction_pos
        scattered_photon, theta, phi = compton_scattering(photon)
        
        if scattered_photon is None:
            print("Compton scattering failed. Skipping this photon.")
            continue
        
     #   print(f"First interaction:")
      #  print(f"  Position: {interaction_pos}")
       # print(f"  Scattering angle (θ): {np.degrees(theta):.2f}°")
        #print(f"  Azimuthal angle (φ): {np.degrees(phi):.2f}°")
        #print(f"  Energy after scattering: {scattered_photon['energy']:.2f} keV")
        
        # Second interaction
        second_interaction_pos = simulate_interaction_position(scattered_photon, detector_size,interaction_number=2)
        if second_interaction_pos is not None:
            pixel_number=get_pixel_number(second_interaction_pos,detector_size)

            if pixel_number == -1:
                print("Single Pixel Event, Second Interaction is in the same pixel")
            elif pixel_number == -2 :
                print("Photon escaped into deeper layer")
            else:
                phi_detected=get_phi_value(pixel_number)
                phi_values.append(phi_detected)
#                print(f"Second interaction:")
 #               print(f"Position: {second_interaction_pos}")
  #              print(f"Actual photon azimuthal angle: {phi}")
   #             print(f"Pixel number: {pixel_number}")
    #            print(f"Phi_Assigned : {phi_detected}")
        else:
            print("Photon escaped after first scattering")
        
        end_time = time.time()
        print(f"Total time for this photon: {end_time - start_time:.4f} seconds")
        print("--------------------")
    return phi_values


# Main simulation
num_photons = 300000
detector_size = (0.738, 0.738,0.5)  # cm
initial_energy = 200  # keV
incidence_angles = [[np.pi/3,0],[np.pi/3,np.pi/3],[0,0]]  # radians
pref_angles = [np.pi/4,np.pi/2]
injected_PFs=[1,0.5,0]
i=0    
for incidence_angle in incidence_angles:
    for pref_angle in pref_angles:
        for injected_PF in injected_PFs:
                print("Starting simulation...")
                start_time = time.time()
                phi_unpol = simulate_detector(num_photons, detector_size, initial_energy, incidence_angle,randomise_incidence_point=False,PF=0,preferred_angle=pref_angle)
                end_time = time.time()
                print(f"Total simulation time: {end_time - start_time:.4f} seconds")
                print(phi_unpol)
                # Plotting code (if enough data is collected)
                if len(phi_unpol) > 0:
                    hist, bin_edges = np.histogram(phi_unpol, bins=8, range=(-np.pi/4, 8*np.pi/4))
                    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
                    plt.figure(figsize=(10, 6))
                    plt.bar(bin_centres, hist, width=np.pi/4, align='center', edgecolor='k')
                    plt.xlabel('Azimuthal angle (phi)')
                    plt.ylabel('Counts')
                    plt.xticks(bin_centres, [f'{i}π/4' for i in range(8)])
                    plt.savefig(f'AzimuthalUnpolE{initial_energy}Incidence{np.degrees(incidence_angle)}Pref{pref_angle}.png')
                else:
                    print("No phi values collected. Unable to create histogram")


                print("Starting simulation...")
                start_time = time.time()
                phi_fullpol = simulate_detector(num_photons, detector_size, initial_energy, incidence_angle,randomise_incidence_point=False, PF=1,preferred_angle=pref_angle)
                end_time = time.time()
                #print(f"Total simulation time: {end_time - start_time:.4f} seconds")
                #print(phi_fullpol)
                # Plotting code (if enough data is collected)
                if len(phi_fullpol) > 0:
                    hist, bin_edges = np.histogram(phi_fullpol, bins=8, range=(-np.pi/4, 8*np.pi/4))
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    plt.figure(figsize=(10, 6))
                    plt.bar(bin_centers, hist, width=np.pi/4, align='center', edgecolor='k')
                    plt.xlabel('Azimuthal angle (phi)')
                    plt.ylabel('Counts')
                    plt.xticks(bin_centers, [f'{i}π/4' for i in range(8)])
                    plt.savefig(f'AzimuthalfullpolE{initial_energy}Incidence{np.degrees(incidence_angle)}Pref{pref_angle}.png')
                    
                else:
                    print("No phi values collected. Unable to create histogram")


                # Ensure both histograms have been generated
                # Calculate histograms for phi_unpol and phi_fullpol

                hist_unpol, _ = np.histogram(phi_unpol, bins=bin_edges)
                hist_fullpol, _ = np.histogram(phi_fullpol, bins=bin_edges)
                    
                # Avoid division by zero by adding a small value to hist_unpol
                hist_unpol = hist_unpol
                    
                # Divide the histograms
                hist_ratio = hist_fullpol / hist_unpol
                    
                #multiply by average of hist_unpol
                phi_corrected = hist_ratio * np.mean(hist_unpol)

                # Plot the corrected phi values
                plt.figure(figsize=(10, 6))

                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                plt.figure(figsize=(10, 6))
                plt.bar(bin_centers, phi_corrected, width=np.pi/4, align='center', edgecolor='k')
                plt.xlabel('Azimuthal angle (phi)')
                plt.ylabel('Counts')
                plt.xticks(bin_centers, [f'{i}π/4' for i in range(8)])
                plt.savefig(f'AzimuthalgeometrycorrectedfullpolE{initial_energy}Incidence{np.angle(incidence_angle)}Pref{pref_angle}.png')

                #perform a curve fit to the corrected phi values with the function y = A * cos(2(x-phi) ) + B
                def func(x, A, phi, B):
                    return A * np.cos(2*(x-phi)) + B
                bounds = ([1e-6, 0, -np.inf], [np.inf, np.pi, np.inf])  # A > 0, phi between 0 and π, B unbounded
                popt_full, pcov_full = curve_fit(func, bin_centers, phi_corrected,bounds=bounds)
                print(f"Optimal parameters: A={popt_full[0]:.2f}, phi={popt_full[1]:.2f}, B={popt_full[2]:.2f}")

                # Plot the corrected phi values with the fitted curve
                plt.figure(figsize=(10, 6))
                plt.bar(bin_centers, phi_corrected, width=np.pi/4, align='center', edgecolor='k', label='Corrected phi values')
                plt.plot(bin_centers, func(bin_centers, *popt_full), 'r-', label='Fitted curve')
                plt.xlabel('Azimuthal Angle (radians)')
                plt.ylabel('Geometry corrected phi_values')
                plt.legend()
                plt.grid(True)
                plt.xticks(bin_centers, [f'{i}π/4' for i in range(8)])


                #overplot the obtained smoothed curve with the actual data
                plt.figure(figsize=(10, 6))
                plt.bar(bin_centers, phi_corrected, width=np.pi/4, align='center', edgecolor='k', label='Corrected phi values')
                plt.plot(np.linspace(-1,7,10000), func(np.linspace(-1,7,10000), *popt_full), 'r-', label='Fitted curve')
                plt.xlabel('Azimuthal Angle (radians)')
                plt.ylabel('Geometry corrected phi_values')
                plt.legend()
                plt.grid(True)
                plt.xticks(bin_centers, [f'{i}π/4' for i in range(8)])


                # Vary the obtained optimal parameters within one standard deviation
                num_variations = 100
                alpha_value = 0.1  # Reduced opacity

                plt.figure(figsize=(10, 6))
                plt.bar(bin_centers, phi_corrected, width=np.pi/4, align='center', edgecolor='k', label='Corrected phi values')

                # Plot the original fitted curve
                plt.plot(np.linspace(-1, 7, 10000), func(np.linspace(-1, 7, 10000), *popt_full), 'r', label='Fitted curve')

                # Generate 100 sets of parameter variations
                for _ in range(100):
                    sample_params = np.random.multivariate_normal(popt_full, pcov_full)
                    y_sample = func(np.linspace(-1, 7, 10000), *sample_params)
                    plt.plot(np.linspace(-1, 7, 10000), y_sample, 'r', alpha=0.1)

                plt.xticks(bin_centers, [f'{i}π/4' for i in range(8)])

                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend(["Best Fit Curve", "Parameter Variations"], loc="upper right")
                plt.savefig(f'AzimuthalgeometrycorrectedfullwithmanycurvepolE{initial_energy}Incidence{np.angle(incidence_angle)}Pref{pref_angle}.png')


                print("Starting simulation...")
                start_time = time.time()
                phi_somepol = simulate_detector(num_photons, detector_size, initial_energy, incidence_angle,randomise_incidence_point=False, PF=injected_PF,preferred_angle=pref_angle)
                end_time = time.time()
                #print(f"Total simulation time: {end_time - start_time:.4f} seconds")
                #print(phi_fullpol)
                # Plotting code (if enough data is collected)
                if len(phi_somepol) > 0:
                    hist, bin_edges = np.histogram(phi_fullpol, bins=8, range=(-np.pi/4, 8*np.pi/4))
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    plt.figure(figsize=(10, 6))
                    plt.bar(bin_centers, hist, width=np.pi/4, align='center', edgecolor='k')
                    plt.xlabel('Azimuthal angle (phi)')
                    plt.ylabel('Counts')
                    plt.xticks(bin_centers, [f'{i}π/4' for i in range(8)])
                    
                    plt.savefig(f'AzimuthalsomepolE{initial_energy}Incidence{np.angle(incidence_angle)}Pref{pref_angle}.png')
                    
                else:
                    print("No phi values collected. Unable to create histogram")

                # Ensure both histograms have been generated
                # Calculate histograms for phi_unpol and phi_fullpol

                hist_unpol, _ = np.histogram(phi_unpol, bins=bin_edges)
                hist_somepol, _ = np.histogram(phi_somepol, bins=bin_edges)
                    
                # Avoid division by zero by adding a small value to hist_unpol
                hist_unpol = hist_unpol
                    
                # Divide the histograms
                hist_ratio = hist_somepol / hist_unpol
                    
                #multiply by average of hist_unpol
                phi_corrected = hist_ratio * np.mean(hist_unpol)

                # Plot the corrected phi values
                plt.figure(figsize=(10, 6))

                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                plt.figure(figsize=(10, 6))
                plt.bar(bin_centers, phi_corrected, width=np.pi/4, align='center', edgecolor='k')
                plt.xlabel('Azimuthal angle (phi)')
                plt.ylabel('Counts')
                plt.xticks(bin_centers, [f'{i}π/4' for i in range(8)])

                plt.savefig(f'AzimuthalgeometrycorrectedsomepolE{initial_energy}Incidence{np.angle(incidence_angle)}Pref{pref_angle}.png')

                # %%
                #perform a curve fit to the corrected phi values with the function y = A * cos(2(x-phi) ) + B
                def func(x, A, phi, B):
                    return A * np.cos(2*(x-phi)) + B
                bounds = ([1e-10, 0, -np.inf], [np.inf, np.pi, np.inf])  # A > 0, phi between 0 and π, B unbounded
                popt_some, pcov_some = curve_fit(func, bin_centers, phi_corrected, bounds=bounds)
                print(f"Optimal parameters: A={popt_some[0]:.2f}, phi={popt_some[1]:.2f}, B={popt_some[2]:.2f}")

                # Plot the corrected phi values with the fitted curve
                plt.figure(figsize=(10, 6))
                plt.bar(bin_centers, phi_corrected, width=np.pi/4, align='center', edgecolor='k', label='Corrected phi values')
                plt.plot(bin_centers, func(bin_centers, *popt_some), 'r-', label='Fitted curve')
                plt.xlabel('Azimuthal Angle (radians)')
                plt.ylabel('Geometry corrected phi_values')
                plt.legend()
                plt.grid(True)
                plt.xticks(bin_centers, [f'{i}π/4' for i in range(8)])
                plt.savefig(f'AzimuthalgeometrycorrectedsomepolE{initial_energy}Incidence{np.angle(incidence_angle)}Pref{pref_angle}.png')

                # Vary the obtained optimal parameters within one standard deviation
                num_variations = 100
                alpha_value = 0.1  # Reduced opacity

                plt.figure(figsize=(10, 6))
                plt.bar(bin_centers, phi_corrected, width=np.pi/4, align='center', edgecolor='k', label='Corrected phi values')

                # Plot the original fitted curve
                plt.plot(np.linspace(-1, 7, 10000), func(np.linspace(-1, 7, 10000), *popt_some), 'r', label='Fitted curve')

                # Generate 100 sets of parameter variations
                for _ in range(100):
                    sample_params = np.random.multivariate_normal(popt_some, pcov_some)
                    y_sample = func(np.linspace(-1, 7, 10000), *sample_params)
                    plt.plot(np.linspace(-1, 7, 10000), y_sample, 'r', alpha=0.1)

                plt.xticks(bin_centers, [f'{i}π/4' for i in range(8)])

                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend(["Best Fit Curve", "Parameter Variations"], loc="upper right")
                plt.savefig(f'AzimuthalgeometrycorrectedsomewithmanycurvepolE{initial_energy}Incidence{np.angle(incidence_angle)}Pref{pref_angle}.png')


                # Parameter errors from covariance matrices
                A_some_err, phi_some_err, B_some_err = np.sqrt(np.diag(pcov_some))
                A_full_err, _, B_full_err = np.sqrt(np.diag(pcov_full))

                # Calculate recovered_PF and recovered_PA
                recovered_PF = (popt_some[0] / popt_some[2]) / (popt_full[0] / popt_full[2])
                recovered_PA = popt_some[1]

                # Error propagation for recovered_PF
                PF_some = popt_some[0] / popt_some[2]
                PF_full = popt_full[0] / popt_full[2]

                # Error on PF_some and PF_full
                PF_some_err = abs(PF_some) * np.sqrt((A_some_err / popt_some[0])**2 + (B_some_err / popt_some[2])**2)
                PF_full_err = abs(PF_full) * np.sqrt((A_full_err / popt_full[0])**2 + (B_full_err / popt_full[2])**2)

                # Error on recovered_PF using the propagated errors
                recovered_PF_err = abs(recovered_PF) * np.sqrt((PF_some_err / PF_some)**2 + (PF_full_err / PF_full)**2)

                # Error for recovered_PA (just the error in phi from popt_some)
                recovered_PA_err = phi_some_err

                # Print the results
                print(f"Recovered PF = {recovered_PF:.3f} ± {recovered_PF_err:.3f}")
                print(f"Recovered PA = {recovered_PA:.3f} ± {recovered_PA_err:.3f}")

                import csv
                import numpy as np

                # Assuming recovered_PF, recovered_PF_err, recovered_PA, and recovered_PA_err are already computed

                # File path for the output CSV file
                
                output_file = f'simulation_results{i}.csv'
                i=i+1
                # Data dictionary for easier writing to CSV
                data_row = {
                    'Number of photons':num_photons,
                    'Energy': initial_energy,
                    'Incidence[0]': incidence_angle[0],
                    'Incidence[1]': incidence_angle[1],
                    'Pref_Angle': pref_angle,
                    'Injected_PF': injected_PF,
                    'Recovered_PF': recovered_PF,
                    'Recovered_PF_Error': recovered_PF_err,
                    'Recovered_PA': recovered_PA,
                    'Recovered_PA_Error': recovered_PA_err
                }
                # Write to CSV file
                with open(output_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=data_row.keys())
                    writer.writeheader()  # Write headers once
                    writer.writerow(data_row)  # Write data row

                print(f"Results written to {output_file}")

                print(f"completed{incidence_angle}{pref_angle}{injected_PF}")



