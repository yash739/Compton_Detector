import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Define the detector as a 3x3 grid in the xy-plane
x = np.linspace(-1, 1, 3)
y = np.linspace(-1, 1, 3)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)  # z = 0 for detector in the xy-plane

# Plot the detector grid
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, color="cyan", edgecolor="k")

# Define the photon's propagation vector (incident photon)
photon_direction = np.array([0.5, -0.7, 1.2])
photon_direction /= np.linalg.norm(photon_direction)  # Normalize

# Plot the photon's direction vector
origin = np.array([0, 0, 0])
ax.quiver(*origin, *photon_direction, color="red", arrow_length_ratio=0.2, label="Photon Direction")

# Choose a random "preferred angle" on the detector (xy-plane)
preferred_angle = np.array([0.7, 0.5, 0])  # example preferred angle in xy-plane
preferred_angle /= np.linalg.norm(preferred_angle)  # Normalize

# Plot the preferred angle vector
ax.quiver(*origin, *preferred_angle, color="blue", arrow_length_ratio=0.2, label="Preferred Angle")

# Calculate the polarization direction as the cross product
polarization_direction = np.cross(preferred_angle, photon_direction)
polarization_direction /= np.linalg.norm(polarization_direction)  # Normalize

# Plot the polarization direction
ax.quiver(*origin, *polarization_direction, color="green", arrow_length_ratio=0.2, label="Polarization Direction")

# Define the plane perpendicular to the polarization direction
# Using point-normal form: (x - x0) * nx + (y - y0) * ny + (z - z0) * nz = 0
# Here (x0, y0, z0) = origin, and (nx, ny, nz) = polarization_direction

# Create a meshgrid to represent the perpendicular plane
plane_size = 2
d = -np.dot(polarization_direction, origin)
xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10), np.linspace(-plane_size, plane_size, 10))
zz = (-polarization_direction[0] * xx - polarization_direction[1] * yy - d) * 1.0 / polarization_direction[2]

# Plot the plane
ax.plot_surface(xx, yy, zz, color="yellow", alpha=0.3)

# Set labels and legend
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()

# Set plot limits for better visualization
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

plt.show()


# Adjust the detector size and add some depth

# Define the detector dimensions
detector_depth = 0.7  # depth along z-axis
detector_size = 2     # larger size for the detector in xy-plane

# Create a new 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Define the detector surface with depth
x = np.linspace(-detector_size/2, detector_size/2, 4)  # Larger grid in xy-plane
y = np.linspace(-detector_size/2, detector_size/2, 4)
X, Y = np.meshgrid(x, y)

# Plot front face of detector at z = 0
Z_front = np.zeros_like(X)  # front surface of the detector
ax.plot_surface(X, Y, Z_front, color="cyan", alpha=0.5, edgecolor="k", rstride=1, cstride=1)

# Plot back face of detector at z = detector_depth
Z_back = Z_front + detector_depth  # back surface of the detector
ax.plot_surface(X, Y, Z_back, color="cyan", alpha=0.5, edgecolor="k", rstride=1, cstride=1)

# Draw lines to create the "depth" of the detector
for i in range(len(x)):
    for j in range(len(y)):
        ax.plot([X[i, j], X[i, j]], [Y[i, j], Y[i, j]], [Z_front[i, j], Z_back[i, j]], color="black")

# Plot photon's direction vector
origin = np.array([0, 0, 0])
ax.quiver(*origin, *photon_direction, color="red", arrow_length_ratio=0.2, label="Photon Direction")

# Plot the preferred angle vector in the xy-plane
ax.quiver(*origin, *preferred_angle, color="blue", arrow_length_ratio=0.2, label="Preferred Angle")

# Plot the polarization direction
ax.quiver(*origin, *polarization_direction, color="green", arrow_length_ratio=0.2, label="Polarization Direction")

# Define and plot the plane perpendicular to the polarization direction
d = -np.dot(polarization_direction, origin)
xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10), np.linspace(-plane_size, plane_size, 10))
zz = (-polarization_direction[0] * xx - polarization_direction[1] * yy - d) * 1.0 / polarization_direction[2]

# Plot the perpendicular plane
ax.plot_surface(xx, yy, zz, color="yellow", alpha=0.3)

# Set labels and legend
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()

# Set plot limits for better visualization
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

plt.show()


# Adjust the visualization to make it clearer that the preferred angle lies in the intersection plane.

# Create a new 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the detector as a 3x3 grid with depth
x = np.linspace(-detector_size/2, detector_size/2, 4)
y = np.linspace(-detector_size/2, detector_size/2, 4)
X, Y = np.meshgrid(x, y)

# Front and back surfaces of the detector
Z_front = np.zeros_like(X)
Z_back = Z_front + detector_depth

# Plot front face at z = 0
ax.plot_surface(X, Y, Z_front, color="cyan", alpha=0.5, edgecolor="k", rstride=1, cstride=1)
# Plot back face at z = detector_depth
ax.plot_surface(X, Y, Z_back, color="cyan", alpha=0.5, edgecolor="k", rstride=1, cstride=1)

# Add lines to show the detector depth
for i in range(len(x)):
    for j in range(len(y)):
        ax.plot([X[i, j], X[i, j]], [Y[i, j], Y[i, j]], [Z_front[i, j], Z_back[i, j]], color="black")

# Plot the photon's direction vector
ax.quiver(*origin, *photon_direction, color="red", arrow_length_ratio=0.2, label="Photon Direction")

# Plot the preferred angle vector within the xy-plane (detector surface)
ax.quiver(*origin, *preferred_angle, color="blue", arrow_length_ratio=0.2, label="Preferred Angle")

# Plot the polarization direction
ax.quiver(*origin, *polarization_direction, color="green", arrow_length_ratio=0.2, label="Polarization Direction")

# Define and plot the plane perpendicular to the polarization direction (with transparency for better clarity)
d = -np.dot(polarization_direction, origin)
xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10), np.linspace(-plane_size, plane_size, 10))
zz = (-polarization_direction[0] * xx - polarization_direction[1] * yy - d) / polarization_direction[2]

# Plot the perpendicular plane in yellow
ax.plot_surface(xx, yy, zz, color="yellow", alpha=0.3)

# Overlay the preferred angle within the plane
# Redraw the preferred angle vector in the same plane to emphasize its location
preferred_angle_end = preferred_angle * detector_size / 2  # scaled endpoint
ax.plot([-preferred_angle_end[0], preferred_angle_end[0]], [-preferred_angle_end[1], preferred_angle_end[1]], [-preferred_angle_end[2], preferred_angle_end[2]], 
        color="orange", linestyle="--", linewidth=2, label="intersection of plane with detector")

# Set labels and legend
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend(loc="upper left")

# Set plot limits for better visualization
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

plt.show()
