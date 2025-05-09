from lib.spectralcam.gentl.gentl import GCSystem
from lib.spectralcam.specim.fx10 import FX10
from log_levels import LogLevel

def connect(ui_context, dpg):
    # Check if there's no camera system yet, and if so, create a new GCSystem
    if ui_context["camera_data"].get("system") is None:
        ui_context["camera_data"]["system"] = GCSystem()
    
    system = ui_context["camera_data"]["system"]
    system.__init__()

    if ui_context["camera_data"].get("cam") is None or ui_context["camera_data"].get("intf") is None:
        # Discover the camera and interface
        cam, intf = system.discover(FX10)

        # update the log
        if not cam:
            ui_context["log_func"](LogLevel.ERROR,"No cam detected")
        elif not intf:
            ui_context["log_func"](LogLevel.ERROR,"No interface detected")
        else:
            ui_context["set_connection_status"](True)  # Update the connection status
            ui_context["log_func"](LogLevel.INFO,"Camera found")
            ui_context["camera_data"]["cam"] = cam
            ui_context["camera_data"]["intf"] = intf
            cam.open_stream()
            ui_context["log_func"](LogLevel.INFO,"Stream opened")

            # When the camera connects init the button for it too
            dpg = ui_context["dpg"]
            dpg.configure_item("connect_button",show=False)
            # dpg.configure_item("disconnect_button",show=True)
            dpg.configure_item("connection_bullet", color=(0,255,0))
    else:
        ui_context["log_func"](LogLevel.WARNING,"Already connected to the camera")

def disconnect(ui_context, dpg):
    dpg = ui_context["dpg"]
    system = ui_context["camera_data"]["system"]
    cam = ui_context["camera_data"].get("cam")

    if cam:
        try:
            cam.close_stream()  # Close the stream if open
        except Exception as e:
            ui_context["log_func"](LogLevel.WARNING, f"Error while closing stream: {e}")

        try:
            cam.close()  # Close the camera connection
        except Exception as e:
            ui_context["log_func"](LogLevel.WARNING, f"Error while closing camera: {e}")

        try:
            system.close()  # Close the system (optional, depending on implementation)
        except Exception as e:
            ui_context["log_func"](LogLevel.WARNING, f"Error while closing system: {e}")

        # Reset the context
        ui_context["camera_data"]["cam"] = None
        ui_context["camera_data"]["intf"] = None
        ui_context["set_connection_status"](False)
        ui_context["log_func"](LogLevel.INFO, "Camera is Disconnected")
    else:
        ui_context["log_func"](LogLevel.WARNING, "No camera was connected")

    dpg.configure_item("disconnect_button", show=False)
    dpg.configure_item("connect_button", show=True)
    dpg.configure_item("connection_bullet", color=(255, 0, 0))
