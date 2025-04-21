import os
import time
import threading
import numpy as np

# === Camera Operation Functions ===

def quick_init_camera(ui_context):
    """Quickly initializes the camera using default parameters."""
    cam = ui_context["camera_data"].get("cam")
    if not cam:
        ui_context["message_box"]("[ERROR] No camera to quick init.")
        return

    cam.quick_init()
    ui_context["log_func"]("[INFO] Camera quick initialized.")

def init_cam_parameters(ui_context, frame_rate=15.0, exposure_time=30000.0, r=100, g=120, b=140):
    """Initializes camera parameters including preview bands and acquisition settings."""
    cam = ui_context["camera_data"].get("cam")
    if not cam:
        ui_context["message_box"]("[ERROR] No camera to initialize.")
        return

    cam.preview_bands(r, g, b)
    cam.set_defaults(frame_rate, exposure_time)
    cam.set("BinningHorizontal", 2)
    cam.open_stream()
    cam.show_preview()
    cam.start_acquire(True)

    ui_context["log_func"]("[INFO] Camera initialized with parameters.")

def save_picture(ui_context, duration=2.0, filename="cube_output.npy"):
    """Captures a datacube and saves it after a given duration without blocking the UI."""
    def worker():
        cam = ui_context["camera_data"].get("cam")
        if not cam:
            ui_context["message_box"]("[ERROR] No camera connected.")
            return

        folder = "Pictures"
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)

        ui_context["log_func"](f"[INFO] Starting acquisition for {duration} seconds...")
        cam.start_acquire(True)
        time.sleep(duration)
        cube = cam.stop_acquire()

        if cube is not None:
            np.save(save_path, cube)
            ui_context["log_func"](f"[INFO] Cube captured and saved to {save_path}. Shape: {cube.shape}")
        else:
            ui_context["log_func"]("[WARNING] No data captured.")

    threading.Thread(target=worker).start()

def get_info(ui_context):
    """Retrieves and displays information about the connected camera."""
    cam = ui_context["camera_data"].get("cam")
    if cam:
        cam.get_info()
        ui_context["log_func"]("[INFO] Camera info retrieved.")
    else:
        ui_context["message_box"]("[ERROR] No camera connected.")