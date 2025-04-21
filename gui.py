import dearpygui.dearpygui as dpg
import camera_connection
import camera

def run_gui():
    # UI-context opslaan
    ui_context = {
        "camera_data": {"cam": None, "intf": None},
        "message_box": message_box,
        "set_connection_status": set_connection_status,
        "log_func": log_message
    }

    # GUI Window
    dpg.create_context()
    with dpg.window(label="HSI Sorting System", width=700, height=400):
        
        # UI Element IDs
        dpg.add_text("Connection status: Disconnected", tag="status_label")
        dpg.add_spacer(height=10)
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Connect FX10", callback=lambda: camera_connection.connect(ui_context))
            dpg.add_button(label="Quick Init", callback=lambda: camera.quick_init_camera(ui_context))
            dpg.add_button(label="Init Parameters", callback=lambda: camera.init_cam_parameters(ui_context))
            dpg.add_button(label="Make DataCube", callback=lambda: camera.save_picture(ui_context))
            # dpg.add_button(label="Extract Picture", callback=lambda: camera.save_picture(ui_context))

        dpg.add_spacer(height=10)
        dpg.add_text("Log:", bullet=True)
        dpg.add_child_window(tag="log_window", autosize_x=True, height=100)

    dpg.create_viewport(title='HSI GUI', width=720, height=450)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

# Logging functie
def log_message(text):
    dpg.add_text(text, parent="log_window")

# Berichtvenster simulatie
def message_box(text):
    log_message(f"[MESSAGE] {text}")

# Status bijwerken
def set_connection_status(connected: bool):
    status = "Connected" if connected else "Disconnected"
    dpg.set_value("status_label", f"Connection status: {status}")
