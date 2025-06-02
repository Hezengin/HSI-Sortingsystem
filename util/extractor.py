import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-GUI backend so it doesnt crash becuz dearpygui and matplotlib both must run on the main thread and i couldnt figure that out
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from util.log_levels import LogLevel
import os
import cv2 as cv

def extract_image(ui_context, datacube_path, band, cmap):
    try:  
        cube_data = np.load(datacube_path, allow_pickle=True)
        ui_context["log_func"](LogLevel.INFO,f"Datacube loaded: {datacube_path}")

        # Data validation
        if cube_data.ndim != 3:
            ui_context["log_func"](LogLevel.ERROR, "Corrupted Datacube file. Loaded datacube is not a 3D array.")
            return
        
        num_bands = cube_data.shape[1]
        strings = band.split(' - ')
        band_index = int(strings[1])
        
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
        
        ax.axis('off') # Turn axes back off for clean image
        fig.savefig("temp_plot_clean.png", dpi=300, bbox_inches='tight', pad_inches=0) # empty with only picture for cv development
        
        ax.axis('on')  # Turn axes back on for detailed image
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Lines scanned')
        ax.set_title(f'Band {band_index} - ColorMap {cmap}')
        fig.colorbar(img, label='Intensity')

        # Save the full plot with info
        fig.savefig("temp_plot_info.png")
        plt.close(fig)
        
    except FileNotFoundError as e:
        ui_context["log_func"](LogLevel.ERROR,f"Datacube file not found.")
        
def save_image(ui_context, band , cmap):
    temp_image_path = "temp_plot_clean.png"
    img = plt.imread(temp_image_path)
    os.makedirs('Pictures', exist_ok=True)
    
    save_path = f'Pictures/{band}_{cmap}.png'
    plt.imsave(save_path, img, cmap=cmap)
    
    ui_context["log_func"](LogLevel.INFO, f"Image saved band: {band}, cmap: {cmap}")
    
def cube_shrinker(ui_context):
    temp_image_path = "temp_plot_clean.png"