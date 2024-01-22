import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

IMG_FILE = "pyrcareworld/pyrcareworld/utils/human_outline.jpg"

class SkeletonVisualizer():
    # Initializes and shows the visualizer.
    def show(self):
        img = plt.imread(IMG_FILE)
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(img, extent=[0, 400, 0, 800], cmap="gray")

        plt.ion()
    
    # Updates the visualizer with new forces.
    def update(self, forces):
        # Add logic to show forces via a heatmap.
        norm = Normalize(vmin=0, vmax=1)

        cmap = 'RdYlGn'
        widths = {0: 100, 1: 140, 2: 140, 3: 50, 4: 50, 5: 50, 6: 50}
        heights = {0: 120, 1: 120, 2: 170, 3: 300, 4: 300, 5: 300, 6: 300}
        angles = {0: 0, 1: 0, 2: 0, 3: -15, 4: 15, 5: -4, 6: 4}
        points = {
            0: (190, 730),
            1: (190, 600),
            2: (190, 450),
            3: (90, 500),
            4: (300, 500),
            5: (150, 225),
            6: (250, 225)}
        self.ax.clear()
        img = plt.imread(IMG_FILE)
        self.ax.imshow(img, extent=[0, 400, 0, 800], cmap="gray")
        for i, force in forces.items():
            if force == 0:
                continue

            color = plt.cm.get_cmap(cmap)(1-norm(force))
            color_alpha = color[:-1] + (0.7,)

            ellipse = patches.Ellipse(points[i], widths[i], heights[i], angle=angles[i], linewidth=2, edgecolor=color_alpha, facecolor=color_alpha)

            self.ax.add_patch(ellipse)


        # Update fig to show the new heatmap.
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Test code. Run with main.
if __name__ == "__main__":
    env = SkeletonVisualizer()
    env.show_visualizer()
    for i in range(100):
        # Sleep 100 millis
        plt.pause(.5)
        # Initialize forces to random values.
        forces = {i: np.random.rand() for i in range(7)}
        print(forces)
        env.update_visualizer(forces)
