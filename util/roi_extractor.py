import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob

def detect_strawberry_contours(crop_path, band_index=130, visualize=True, roi_size=5):
    crop = np.load(crop_path)
    band_img = crop[:, band_index, :]

    threshold = 0.73 * band_img.max()
    mask = (band_img > threshold).astype(np.uint8)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    roi_idx = 0
    if visualize:
        plt.imshow(band_img, cmap='plasma')
        plt.title(f"Band {band_index} met contouren")
        ax = plt.gca()
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
        if visualize:
            ax.add_patch(rect)
        if w >= 10 and h >= 5:
            cx = x + w // 2
            cy = y + h // 2
            # Grotere spacing voor extra ROIs (meer variatie)
            for dx in [-6, -3, 0, 3, 6]:
                roi_cx = cx + dx
                roi_x = int(np.clip(roi_cx - roi_size // 2, 0, band_img.shape[1] - roi_size))
                roi_y = int(np.clip(cy - roi_size // 2, 0, band_img.shape[0] - roi_size))
                roi = crop[roi_y:roi_y+roi_size, :, roi_x:roi_x+roi_size]
                crop_dir = os.path.dirname(crop_path)
                crop_base = os.path.splitext(os.path.basename(crop_path))[0]
                roi_path = os.path.join(crop_dir, f"{crop_base}_roi_{roi_idx:02d}.npy")
                np.save(roi_path, roi)
                roi_idx += 1
                if visualize:
                    roi_rect = plt.Rectangle((roi_x, roi_y), roi_size, roi_size, linewidth=2, edgecolor='blue', facecolor='none')
                    ax.add_patch(roi_rect)
    if visualize:
        plt.show()

def create_rois(crop_path):
    crop_files = sorted(glob.glob(os.path.join(crop_path, 'crop_*.npy')))

    print(f"Gevonden {len(crop_files)} crops in: {crop_path}")

    for crop_file in crop_files:
        detect_strawberry_contours(crop_file,130,False)

def load_roi(crop_path):
    create_rois(crop_path)

    for filename in os.listdir(crop_path):
            if "roi" in filename and filename.endswith(".npy"):
                filepath = os.path.join(crop_path, filename)
                return filepath  

    raise FileNotFoundError("Geen ROI .npy-bestanden gevonden in de map.")

# TESTING
# array = load_roi_arrays(r"DataCubes\05_06_2025\Cropped_05062025_120856")
# print(array)

# for i in range(17):
#     crop_path = f'DataCubes/03_06_2025/Cropped_03062025_152045/crop_{i:03d}.npy'
#     print(f"Processing {crop_path}")
#     contours = detect_strawberry_contours(crop_path, band_index=130)
    
# for i in range(17):
#     crop_path = f'DataCubes/04_06_2025/Cropped_04062025_133054/crop_{i:03d}.npy'
#     print(f"Processing {crop_path}")
#     contours = detect_strawberry_contours(crop_path, band_index=130)
    
# for i in range(17):
#     crop_path = f'DataCubes/04_06_2025/Cropped_04062025_133942/crop_{i:03d}.npy'
#     print(f"Processing {crop_path}")
#     contours = detect_strawberry_contours(crop_path, band_index=130)

# for i in range(17):
#     crop_path = f'DataCubes/04_06_2025/Cropped_04062025_134921/crop_{i:03d}.npy'
#     print(f"Processing {crop_path}")
#     contours = detect_strawberry_contours(crop_path, band_index=130)

# for i in range(17):
#     crop_path = f'DataCubes/05_06_2025/Cropped_05062025_120810/crop_{i:03d}.npy'
#     print(f"Processing {crop_path}")
#     contours = detect_strawberry_contours(crop_path ,band_index=130)

# for i in range(17):
#     crop_path = f'DataCubes/05_06_2025/Cropped_05062025_121305/crop_{i:03d}.npy'
#     print(f"Processing {crop_path}")
#     contours = detect_strawberry_contours(crop_path ,band_index=130)