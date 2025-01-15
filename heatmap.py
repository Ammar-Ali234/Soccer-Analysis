import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

class FootballHeatmap:
    def __init__(self, field_length=105, field_width=68):
        self.field_length = field_length
        self.field_width = field_width
        
    def draw_field(self, ax):
        """Draw a football field with proper markings"""
        # Set background color
        ax.set_facecolor('#1a1a1a')
        
        # Main field outline
        ax.add_patch(Rectangle((0, 0), self.field_length, self.field_width, 
                             fill=False, color='white', linewidth=1))
        
        # Halfway line
        ax.plot([self.field_length/2, self.field_length/2], 
                [0, self.field_width], color='white', linewidth=1)
        
        # Center circle
        center_circle = Circle((self.field_length/2, self.field_width/2), 
                             9.15, fill=False, color='white', linewidth=1)
        ax.add_patch(center_circle)
        
        # Penalty areas
        pen_area_length = 16.5
        pen_area_width = 40.3
        pen_area_start = (self.field_width - pen_area_width)/2
        
        # Left penalty area
        ax.add_patch(Rectangle((0, pen_area_start), pen_area_length,
                             pen_area_width, fill=False, color='white', linewidth=1))
        
        # Right penalty area
        ax.add_patch(Rectangle((self.field_length-pen_area_length, pen_area_start),
                             pen_area_length, pen_area_width, fill=False, color='white', linewidth=1))
        
        # Goal areas
        goal_area_length = 5.5
        goal_area_width = 18.32
        goal_area_start = (self.field_width - goal_area_width)/2
        
        ax.add_patch(Rectangle((0, goal_area_start), goal_area_length,
                             goal_area_width, fill=False, color='white', linewidth=1))
        ax.add_patch(Rectangle((self.field_length-goal_area_length, goal_area_start),
                             goal_area_length, goal_area_width, fill=False, color='white', linewidth=1))

    def generate_heatmap(self, tracking_data, output_file='heatmap.png', 
                        player_id=None, sigma=1):
        """Generate heatmap from tracking data"""
        plt.figure(figsize=(15, 10))
        ax = plt.gca()
        
        # Draw the football field
        self.draw_field(ax)
        
        # Filter data for specific player if requested
        if player_id is not None:
            tracking_data = tracking_data[tracking_data['player_id'] == player_id]
        
        # Normalize coordinates to field dimensions
        x_normalized = tracking_data['x'] * self.field_length / tracking_data['x'].max()
        y_normalized = tracking_data['y'] * self.field_width / tracking_data['y'].max()
        
        # Create heatmap
        heatmap, xedges, yedges = np.histogram2d(
            x_normalized,
            y_normalized,
            bins=50,
            range=[[0, self.field_length], [0, self.field_width]]
        )
        
        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Define a custom gamma colormap
        gamma_cmap = LinearSegmentedColormap.from_list(
            'gamma_cmap', 
            [(0, 'black'), (0.2, 'blue'), (0.4, 'green'), (0.6, 'yellow'), (0.8, 'orange'), (1, 'red')]
        )
        
        # Plot heatmap using the custom gamma colormap
        plt.imshow(
            heatmap.T,
            extent=[0, self.field_length, 0, self.field_width],
            origin='lower',
            cmap=gamma_cmap,  # Use the custom gamma colormap
            alpha=0.7
        )
        
        # Customize plot
        title = 'Team Movement Heatmap' if player_id is None else f'Player {player_id} Movement Heatmap'
        plt.title(title, color='white', pad=20)
        plt.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Movement Intensity', color='white')
        
        # Save plot
        plt.savefig(output_file,
                   facecolor='#1a1a1a',
                   edgecolor='none',
                   bbox_inches='tight',
                   dpi=300)
        plt.close()
        
        return output_file
