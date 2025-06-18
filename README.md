# HSI Sorting System

A hyperspectral imaging (HSI) based sorting system for strawberries. This project classifies strawberries into three categories based on their freshness: **Fresh**, **Old**, and **Spoiled**. It uses a hyperspectral camera (Specim FX10), a conveyor belt system (Desktop Conveyor - Delta X Robot), and a machine learning model to make real-time predictions through a graphical user interface.

---

## 🔧 System Components

The system is designed to work with the following physical setup:

- Specim FX10 Hyperspectral Camera
- Desktop Conveyor - Delta X Robot
- Controlled lighting (light box or dark room)
- A Windows PC with GPU (CUDA support recommended)
- Python 3.10 environment (used python version: 3.10.16)

---

## 🚀 Quick Start

> ⚠️ Update paths in the code if your setup uses different folders or directories.

### 1. Clone the repository
  ```bash
  git clone https://github.com/Hezengin/HSI-Sortingsystem.git
  cd HSI-Sortingsystem
```
### 2. Create and activate a conda environment (recommended)
  ```bash
  conda create -n hsi_sorting python=3.10
  conda activate hsi_sorting
  ```

### 3. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
### 4. Run the system
  ```bash
  python main.py
  ```
Or, with a specific full path on Windows:
  ```bash
 & C:/.conda/envs/env/python.exe c:/HSI-Sortingsystem/main.py
  ```
## 📁 Project Structure
```bash
HSI-Sortingsystem/
├── main.py               # Entry point for the UI and logic
├── camera/               # Camera Interface
├── gui/                  # UI components
├── util/                 # Helper scripts
├── lib/                  # Used Libraries
  ├── spectralcam/                 # Lbrary to use Specim Camera's
├── miscellaneous/        # Miscelanneous/ test files 
├── DataCubes/            # Datacube files and results
├── Resources/            # AI model, fonts
├── bands/                # Files with wavelengths and according bands. 
├── requirements.txt      # Python dependencies
├── temp_plot_clean.png   # Image Viewer
├── temp_plot_info.png    # Image Viewer
└── README.md             # Documentation
```

## 🔍 Using the Classification Software
Follow these steps to scan and classify strawberries using the graphical user interface:

### 1. Prediction Categories
The system classifies each strawberry into one of:
- Fresh
- Old
- Spoiled

### 2. Preparing the System
- Go to the HSI Camera tab and press "Connect to FX10" to connect the hyperspectral camera.
- Go to the Conveyor Belt tab, select the correct device from the dropdown, and press "Connect to Conveyor Belt".

### 3. Placing the Strawberry
- Place a strawberry at the start of the conveyor belt, directly under the camera.

### 4. Environment Setup
- Close all curtains or blinds.
- Avoid direct light or strong reflections that may affect scanning quality.

### 5. Start the Scan
- Press the "Start Scan" button to begin capturing hyperspectral data.

### 6. Stop the Scan
- Press "Stop Scan" once the strawberry has passed through and been fully scanned.

### 7. Selecting the Datacube
- After scanning, press the "Refresh" button to update the dropdown list of datacubes.
- Select the relevant datacube from the dropdown.

### 8. Classify the Result
- Press "Classificate" to make a prediction.
- The predicted category will appear in the Result window.
- A certainty value will also be shown, indicating the confidence of the AI model.

## 📦 Dependencies
This project relies on several libraries for machine learning, image processing, and the UI:
- torch, torchvision, torchaudio
- opencv-python
- scikit-learn, scikit-image
- matplotlib, seaborn
- dearpygui, PyQt5
- numpy, pandas
See the full list in requirements.txt.

## 📌 Notes
- Make sure the FX10 camera driver/SDK is installed and correctly configured.
- Depending on your PC or setup, paths may need to be modified manually.
- Results may vary based on lighting conditions and camera calibration.
- This system is built for research or educational use.

## 🔗 External Documentation & Libraries
- [Conveyor X2 Documentation – Delta Robot](https://docs.deltaxrobot.com/reference/specifications/accessories/delta_x2_accessories/conveyor_x/)
- [SpectralCam Library – Linear Scanner Controller GitLab](https://gitlab.jyu.fi/jpasonen/linear-scanner-controller/-/blob/main/spectralcam/README.md)
