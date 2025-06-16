import os
import time
import threading
import numpy as np
from datetime import datetime
from util.log_levels import LogLevel
from util.gcode_sender import ConveyorBeltConnection

# === Camera Operation Functions ===

conveyor = ConveyorBeltConnection(port="COM5", baudrate=115200)

def init_cam_parameters(ui_context, frame_rate=30.0, exposure_time=30000.0, r=100, g=120, b=140, binning_type="BinningHorizontal", binning=2):
    """Initializes camera parameters including preview bands and acquisition settings."""
    cam = ui_context["camera_data"].get("cam")
    if not cam:
        ui_context["log_func"](LogLevel.ERROR,"No camera to initialize.")
        return

    cam.preview_bands(r, g, b)
    cam.set_defaults(frame_rate, exposure_time)
    cam.set(binning_type, binning)
    ui_context["log_func"](LogLevel.INFO,f"Camera initialized with parameters: FPS: {frame_rate} , EXP time: {exposure_time}, RGB: {r}, {g}, {b}")

def start_datacube(ui_context, preview=True):
    global conveyor
    init_cam_parameters(ui_context, 15.0, 60000.0, 100, 120, 140, "BinningHorizontal", 2)
    cam = ui_context["camera_data"].get("cam")
    if not cam:
        ui_context["log_func"](LogLevel.ERROR,"No camera found in UI context.")
        return

    if preview is True:
        cam.show_preview()
    
    cam.start_acquire(record=True)
    conveyor.send_gcode("M310 1\n")
    conveyor.send_gcode("M311 -10\n")
    ui_context["log_func"](LogLevel.INFO,"Stream opened and data acquisition started.")

def stop_datacube(ui_context):
    global conveyor
    cam = ui_context["camera_data"].get("cam")
    if not cam:
        ui_context["log_func"](LogLevel.ERROR,"No camera found in UI context.")
        return

    cube = cam.stop_acquire()
    if conveyor is not None:
        conveyor.send_gcode("M311 0\n")

    # Prepare directory paths
    folder = "Datacubes"
    daystamp = datetime.now().strftime("%d_%m_%Y")
    full_dir = os.path.join(folder, daystamp)
    os.makedirs(full_dir, exist_ok=True) 

    # Prepare save path
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    save_path = os.path.join(full_dir, f"dc_{timestamp}.npy")

    if cube is not None:
        try:
            np.save(save_path, cube)
            ui_context["log_func"](LogLevel.INFO,f"Datacube saved to {save_path}. Shape: {cube.shape}")
        except Exception as e:
            ui_context["log_func"](LogLevel.ERROR,f"Failed to save datacube: {e}")
    else:
        ui_context["log_func"](LogLevel.WARNING,"No data was captured.")
    ui_context["log_func"](LogLevel.INFO,"Data acquisition stopped. DataCube created.")

def create_datacube_with_duration(ui_context, preview=True, duration=2):
    """Captures a datacube from the camera for a given duration and saves it asynchronously."""
    def worker():
        cam = ui_context["camera_data"].get("cam")
        if not cam:
            ui_context["log_func"](LogLevel.ERROR,"Cannot start creating datacube, no camera connected.")
            return
        time_to_sleep = float(duration)
        start_datacube(ui_context, preview)
        ui_context["log_func"](LogLevel.INFO, f"Starting datacube acquisition for {time_to_sleep:.1f} seconds...")
        time.sleep(time_to_sleep)
        stop_datacube(ui_context)

    threading.Thread(target=worker, daemon=True).start()
def close():
    os._exit(0)