import os
import re
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import gui.gui as gui

def extractor(ui_context, path):
    data_cube = np.load(path, allow_pickle=True)

    bands = np.genfromtxt('bands/bands.csv', delimiter=',')  # Load spectral bands
    rgb_composite = generate_rgb_composite(data_cube, bands)

    # Convert to uint8 RGB, then to HSV
    rgb_uint8 = (rgb_composite * 255).astype(np.uint8)
    hsv_img = cv.cvtColor(rgb_uint8, cv.COLOR_RGB2HSV)

    # Fixed HSV range
    lower_bound = np.array([0, 20, 0])
    upper_bound = np.array([120, 255, 157])
    mask = cv.inRange(hsv_img, lower_bound, upper_bound)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Extract timestamp from filename
    base_dir = os.path.dirname(path)
    filename = os.path.basename(path)
    match = re.search(r"dc_(\d{8}_\d{6})", filename)
    if match:
        timestamp = match.group(1)
        folder_name = f"Cropped_{timestamp}"
        output_dir = os.path.join(base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise ValueError(f"Geen geldige timestamp gevonden in bestandsnaam: {filename}")


    crop_count = 0
    for i in range(1, len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)

        # Skip small contours
        if w < 30 or h < 15 or area < 3000:
            # ui_context.log(f"[SKIP] Contour {i} too small: w={w}, h={h}, area={area}")
            continue

        cropped_cube = data_cube[y:y+h, :, x:x+w]
        crop_filename = os.path.join(output_dir, f'crop_{crop_count:03d}.npy')
        np.save(crop_filename, cropped_cube)
        # ui_context.log(f"[SAVED] {os.path.basename(crop_filename)} - shape: {cropped_cube.shape}, area: {area}")
        crop_count += 1
        
    gui.refresh_comboboxes()

def generate_rgb_composite(data_cube, bands, red_range=(620, 750), green_range=(495, 570), blue_range=(450, 495)):
    red_band = np.argmin(np.abs(bands - np.mean(red_range)))
    green_band = np.argmin(np.abs(bands - np.mean(green_range)))
    blue_band = np.argmin(np.abs(bands - np.mean(blue_range)))

    rgb = np.stack([
        data_cube[:, red_band, :],
        data_cube[:, green_band, :],
        data_cube[:, blue_band, :]
    ], axis=-1)

    rgb_norm = rgb - rgb.min(axis=(0, 1), keepdims=True)
    rgb_norm = rgb_norm / rgb_norm.max(axis=(0, 1), keepdims=True)

    gamma = 1.5
    rgb_composite = np.power(rgb_norm, 1 / gamma)
    return rgb_composite

