import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [], lw=5)

# Set the limits of the plot
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# Animation time setup
start_time = time.time()
max_duration = 10
fps = 30

def update_line(line, delta_theta):
    fixed_point = [0, 0, 5]  # Base point
    length = 1  # Length of the line

    # Calculate the end point of the line using trigonometry (rotation)
    # delta_theta is the angle in degrees, we convert to radians for np.cos and np.sin
    angle_rad = np.deg2rad(delta_theta)
    
    end_point = [
        fixed_point[0] + (length * np.cos(angle_rad)),  # X changes with cos(theta)
        fixed_point[1],  # Y stays the same
        fixed_point[2] + (length * np.sin(angle_rad))   # Z changes with sin(theta)
    ]
    
    # Update the line with new data: X, Y, and Z
    line.set_data([fixed_point[0], end_point[0]], [fixed_point[1], end_point[1]])  # X, Y
    line.set_3d_properties([fixed_point[2], end_point[2]])  # Z
    
    return line,

def update(frame):
    # Stop the animation after max_duration
    if time.time() - start_time > max_duration:
        ani.event_source.stop()

    # Calculate the delta_theta based on the frame
    # This will rotate the line continuously over the frames
    delta_theta = frame  # Degrees per frame
    
    # Call the update_line function to update the line's position
    return update_line(line, delta_theta)

# Create the animation
ani = FuncAnimation(fig, update, frames=300, interval=50, repeat=False)

# Display the animation
plt.show()
