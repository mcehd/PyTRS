import matplotlib.pyplot as plt
import numpy as np

# A map can be represented by a matrix of integers. Here, the convention is the following:
#   -  0 for accessible cells
#   -  1 for obstacles
# The type of the elements is chosen to be logical (as in Fortran this type is more commonly called 
# bool or boolean). This allows to use as little memory as possible for the same map size (and 
# thus to have faster operations on the map). To store larger elements, you could use 'uint8' 
# (unsigned integer, 8 bits) or 'int8' (signed integer, 8 bits).
world_map = np.ones((64, 64), np.bool)  # At first, no cell is accessible.
world_map[19:40, 19:40] = 0 # Dig a rectangle in it.
world_map[29, 19:40] = 1    # Draw three walls.
world_map[24:35, [25, 35]] = 1

# With this representation, contour can be used to draw the map.
plt.subplot(121).set_aspect('equal')
plt.contour(1 - world_map)  # Invert the booleans to draw the inner contour
plt.title('Contour')

# We can also draw the map
plt.subplot(122).set_aspect('equal')
plt.imshow(world_map, origin='lower')  # Invert the booleans to draw the inner contour
plt.title('Map')

plt.show()