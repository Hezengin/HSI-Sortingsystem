import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-GUI backend so it doesnt crash becuz dearpygui and matplotlib both must run on the main thread and i couldnt figure that out
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from log_levels import LogLevel
import threading
import time

def extract_image(ui_context, band, cmap):
    
        cube_data = np.load("Pictures/cube_output_appel.npy", allow_pickle=True)
        
        # Data validation
        if cube_data.ndim != 3:
            ui_context["log_func"](LogLevel.ERROR, "Loaded datacube is not a 3D array.")
            return
            
        num_bands = cube_data.shape[1]
        band_index = int(band)
        
        if band_index < 0 or band_index >= num_bands:
            ui_context["log_func"](LogLevel.ERROR, f"Band index {band_index} is out of bounds (0-{num_bands-1}).")
            return

        # Process image data
        band_data = cube_data[:, band_index, :]
        normalized_band = (band_data - band_data.min()) / (band_data.max() - band_data.min())
        ui_context["log_func"](LogLevel.INFO, f"Processing band {band_index} with {cmap} colormap")

        # Create and save plot
        fig = Figure()
        ax = fig.subplots()
        img = ax.imshow(normalized_band, cmap=cmap, origin='lower')
        fig.colorbar(img, label='Intensity')
        ax.set_title(f'Band {band_index}')
        
        temp_image_path = "temp_plot.png"
        fig.savefig(temp_image_path)
        plt.close(fig)