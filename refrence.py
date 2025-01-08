import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


line, = ax.plot([], [], [], lw=5)
position = [0.5, 0.5, 0]


# # Set the limits of the plot
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

start_time = time.time()
max_duration = 10
fps = 30

def update(frame):
    if(time.time() - start_time > max_duration):
        ani.event_source.stop()
    render_time = frame / fps
    position_delta = [0.1, 0.1, 0.1]
    global position
    position = [p + delta for p, delta in zip(position, position_delta)]
    print(position)
    line.set_data([0, position[0]], [0, position[1]])  # X, Y
    line.set_3d_properties([0, position[2]])  # Z

    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=300, interval=50, repeat=False)

# Display the animation
plt.show()
