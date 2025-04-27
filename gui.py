import dearpygui.dearpygui as dpg
import camera_connection
import camera_helper
from log_levels import LogLevel
import matplotlib.pyplot as plt
import extractor

def run_gui():
    
    camera_data = {
        "system": None
    }

    ui_context = {
        "dpg": dpg,
        "camera_data": camera_data,
        "set_connection_status": set_connection_status,
        "log_func": log_message
    }

    # GUI Window
    dpg.create_context()
    with dpg.window(label="HSI Sorting System", tag="main_window", width=550, height=400, no_resize=True):
        # UI Element IDs
        
        with dpg.group(horizontal=True, tag="connection_group"):
            dpg.add_text(tag="connection_bullet",bullet=True, color=(255,0,0))
            dpg.add_text("Connection status: Disconnected", tag="status_label")
            dpg.add_spacer(height=10)
            dpg.add_button(label="Connect FX10", tag="connect_button",callback=lambda: camera_connection.connect(ui_context, dpg))

        dpg.add_spacer(height=10)

        dpg.add_text("Operations")
        dpg.add_child_window(tag="operations_window", autosize_x=True, height=83)
        with dpg.group(horizontal=False, parent="operations_window"):
            dpg.add_button(label="Make DataCube with Preview", callback=lambda: camera_helper.start_datacube(ui_context, True))
            dpg.add_button(label="Make DataCube without Preview", callback=lambda: camera_helper.start_datacube(ui_context, False))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Make DataCube with duration", callback=lambda: camera_helper.create_datacube_with_duration(ui_context, True, dpg.get_value("duration_input")))
                dpg.add_input_text(label="duration (in seconds)",tag="duration_input", default_value= 2.0, width=50)

        dpg.add_text("Image Viewer")
        dpg.add_child_window(tag="image_extraction_window", autosize_x=True, height=105)
        with dpg.group(horizontal=False, parent="image_extraction_window"):
            dpg.add_text("Default is band 50 and colormap magma.")
            with dpg.group(horizontal=True):
                dpg.add_text("Band")
                dpg.add_input_text(tag="band_input", width=50, label="(0-223)", default_value=50, payload_type=int)
            with dpg.group(horizontal=True):
                cmaps = plt.colormaps()
                dpg.add_text("Select a colormap")
                dpg.add_combo(cmaps, tag="cmap_dropdown", width=100, default_value="magma")
            dpg.add_button(label="Show Image", callback=lambda: extractor.extract_image(ui_context, dpg.get_value("band_input"), dpg.get_value("cmap_dropdown")))
        
        dpg.add_spacer(height=10)
        dpg.add_button(label="Close App", callback=lambda: camera_helper.close(ui_context))
        dpg.add_spacer(height=10)
        
        dpg.add_text("Log:")   
        dpg.add_child_window(tag="log_window", autosize_x=True, height=100)
        dpg.add_button(label="Clear Log", callback=lambda: clear_log())

    dpg.create_viewport(title='HSI GUI', width=570, height=550)
    dpg.setup_dearpygui()
    dpg.set_primary_window("main_window", True)
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

# Logger
def log_message(message_level, text):
    if isinstance(message_level, LogLevel):
        message_level_txt = message_level.name
    else:
        message_level_txt = "UNKNOWN"

    dpg.add_text(f"[{message_level_txt}] {text}", parent="log_window")
    dpg.set_y_scroll("log_window", 9999)

def clear_log():
    dpg.delete_item("log_window", children_only=True)  

# Status update
def set_connection_status(connected: bool):
    status = "Connected" if connected else "Disconnected"
    dpg.set_value("status_label", f"Connection status: {status}")