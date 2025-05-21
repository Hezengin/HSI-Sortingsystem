import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def generate_rgb_composite(data_cube, bands, red_range=(620, 750), green_range=(495, 570), blue_range=(450, 495)):
    """
    Generate an RGB composite image from a data cube using specified wavelength ranges.

    Parameters:
    - data_cube (numpy.ndarray): The input 3D data cube.
    - bands (numpy.ndarray): Array of band wavelengths to map to.
    - red_range (tuple): Wavelength range for the red channel (default: (620, 750)).
    - green_range (tuple): Wavelength range for the green channel (default: (495, 570)).
    - blue_range (tuple): Wavelength range for the blue channel (default: (450, 495)).

    Returns:
    - rgb_composite (numpy.ndarray): The generated RGB composite image.
    """
    # Find the indices of the bands that correspond to the wavelength ranges
    red_band = np.argmin(np.abs(bands - np.mean(red_range)))
    green_band = np.argmin(np.abs(bands - np.mean(green_range)))
    blue_band = np.argmin(np.abs(bands - np.mean(blue_range)))

    # Stack the selected bands into an RGB image
    rgb = np.stack([data_cube[:, red_band, :], data_cube[:, green_band, :], data_cube[:, blue_band, :]], axis=-1)

    # Normalize each channel separately
    rgb_norm = rgb - rgb.min(axis=(0, 1), keepdims=True)
    rgb_norm = rgb_norm / rgb_norm.max(axis=(0, 1), keepdims=True)

    # Apply gamma correction
    gamma = 1.5
    rgb_composite = np.power(rgb_norm, 1 / gamma)

    return rgb_composite

# Example usage
data_cube = np.load('Datacubes/09_05_2025/data_cube_20250509_151914.npy', allow_pickle=True)
bands = np.genfromtxt('bands/bands.csv', delimiter=',')  # Load the bands
red_range = (620, 750)  # Define the wavelength range for red

rgb_composite = generate_rgb_composite(data_cube, bands)

# Stap 1: Van jouw genormaliseerde rgb_composite naar uint8 (0-255)
rgb_uint8 = (rgb_composite * 255).astype(np.uint8)

# Stap 2: RGB naar BGR want OpenCV verwacht BGR
bgr_img = cv.cvtColor(rgb_uint8, cv.COLOR_RGB2BGR)

# Stap 3: BGR naar HSV
hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)

# Stap 4: Rood heeft twee ranges in HSV (OpenCV hue 0-179)
lower_red1 = np.array([0, 10, 10])      # lagere saturatie en value drempel
upper_red1 = np.array([10, 255, 255])   # Eerste range hoog

lower_red2 = np.array([170, 10, 10])    # Tweede range laag
upper_red2 = np.array([179, 255, 255])  # Tweede range hoog

# Stap 5: Maak twee maskers en combineer
mask1 = cv.inRange(hsv_img, lower_red1, upper_red1)
mask2 = cv.inRange(hsv_img, lower_red2, upper_red2)
red_mask = cv.bitwise_or(mask1, mask2)

# Stap 6: Masker toepassen op originele afbeelding
red_segment = cv.bitwise_and(bgr_img, bgr_img, mask=red_mask)

# Stap 7: Terug naar RGB voor matplotlib
red_segment_rgb = cv.cvtColor(red_segment, cv.COLOR_BGR2RGB)

# Tonen
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Originele RGB composite")
plt.imshow(rgb_uint8)
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Rood segment uit aardbeien")
plt.imshow(red_segment_rgb)
plt.axis('off')

plt.show()

def nothing(x):
    pass

cv.namedWindow('Trackbars')
cv.createTrackbar('LowH', 'Trackbars', 0, 179, nothing)
cv.createTrackbar('HighH', 'Trackbars', 10, 179, nothing)
cv.createTrackbar('LowS', 'Trackbars', 20, 255, nothing)
cv.createTrackbar('HighS', 'Trackbars', 255, 255, nothing)
cv.createTrackbar('LowV', 'Trackbars', 20, 255, nothing)
cv.createTrackbar('HighV', 'Trackbars', 255, 255, nothing)

while True:
    low_h = cv.getTrackbarPos('LowH', 'Trackbars')
    high_h = cv.getTrackbarPos('HighH', 'Trackbars')
    low_s = cv.getTrackbarPos('LowS', 'Trackbars')
    high_s = cv.getTrackbarPos('HighS', 'Trackbars')
    low_v = cv.getTrackbarPos('LowV', 'Trackbars')
    high_v = cv.getTrackbarPos('HighV', 'Trackbars')

    lower_red = np.array([low_h, low_s, low_v])
    upper_red = np.array([high_h, high_s, high_v])

    mask = cv.inRange(hsv_img, lower_red, upper_red)

    cv.imshow('Mask', mask)

    k = cv.waitKey(1) & 0xFF
    if k == 27:  # ESC om te stoppen
        break

cv.destroyAllWindows()

