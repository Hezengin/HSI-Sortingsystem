from lib.spectralcam.gentl.gentl import GCSystem
from lib.spectralcam.specim.fx10 import FX10

def connect(app):
    # Check if there's no camera system yet, and if so, create a new GCSystem
    if app.get("camera_data") is None:
        app["camera_data"] = {"cam": None, "intf": None}

    # Create the GCSystem object
    system = GCSystem()

    # Discover the camera and interface
    cam, intf = system.discover(FX10)

    # Update the UI based on whether the camera and interface are found
    if not cam:
        app["message_box"]("No cam detected")
    elif not intf:
        app["message_box"]("No interface detected")
    else:
        app["set_connection_status"](True)  # Update the connection status
        app["message_box"]("Cam found")  # Show the message box
        app["camera_data"]["cam"] = cam
        app["camera_data"]["intf"] = intf
