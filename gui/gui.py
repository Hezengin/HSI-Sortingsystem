import dearpygui.dearpygui as dpg
import camera.camera_connection as camera_connection
import camera.camera_helper as camera_helper
from util.log_levels import LogLevel
import matplotlib.pyplot as plt
import util.image_extractor as image_extractor
import util.datacube_extractor as datacube_extractor
import util.roi_extractor as detect_roi
import util.strawberry_classifier as strawberry_classifier
import util.conveyorbelt_helper as conveyorbelt_helper

def run_gui():
    
    camera_data = {
        "system": None
    }

    ui_context = {
        "dpg": dpg,
        "camera_data": camera_data,
        "conveyor": None,
        "set_connection_status": set_connection_status,
        "log_func": log_message
    }

    # GUI Window
    dpg.create_context()
    with dpg.window(label="HSI Sorting System", tag="main_window", width=550, height=400, no_resize=True):
        
        # FONTS
        with dpg.font_registry():
            large_font = dpg.add_font("Resources/OpenSans-Bold.ttf", 32)
        
        with dpg.tab_bar():
            # HSI CAMERA TAB
            with dpg.tab(label="HSI Camera"):
                hsi_camera_header = dpg.add_text("Hyperspectral Camera")
                dpg.bind_item_font(hsi_camera_header, large_font)
                
                # CONNECTION BUTTON AND STATUS
                dpg.add_text("Connection")
                dpg.add_child_window(tag="connection_window", autosize_x=True, height=35)
                with dpg.group(horizontal=True, tag="connection_group", parent="connection_window"):
                    dpg.add_text(tag="connection_bullet",bullet=True, color=(255,0,0))
                    dpg.add_text("Connection status: Disconnected", tag="status_label")
                    dpg.add_button(label="Connect FX10", tag="connect_button",callback=lambda: camera_connection.connect(ui_context, dpg))
                    # In case we want to disconnect#dpg.add_button(label="Disconnect FX10", tag="disconnect_button", callback=lambda: camera_connection.disconnect(ui_context, dpg), show=False)

                # CAMERA OPERATIONS WINDOW
                dpg.add_text("Operations")
                dpg.add_child_window(tag="operations_window", autosize_x=True, height=105)
                with dpg.group(horizontal=False, parent="operations_window"):
                    dpg.add_button(label="Make DataCube with Preview", callback=lambda: camera_helper.start_datacube(ui_context, True))
                    dpg.add_button(label="Make DataCube without Preview", callback=lambda: camera_helper.start_datacube(ui_context, False))
                    
                    # MAKE DATACUBE WITH DURATION FIELD
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Make DataCube with duration", callback=lambda: camera_helper.create_datacube_with_duration(ui_context, True, dpg.get_value("duration_input")))
                        dpg.add_input_text(label="duration (in seconds)",tag="duration_input", default_value= 2, width=50)
                    dpg.add_button(label="Save the Datacube", callback=lambda: camera_helper.stop_datacube(ui_context))
                    
                # DATACUBE EXTRACTOR OPERATIONS WINDOW
                dpg.add_text("Datacube extractor")
                dpg.add_child_window(tag="datacube_extractor_window", autosize_x=True, height=60)
                with dpg.group(horizontal=False, parent="datacube_extractor_window"):
                    
                    # DATACUBE STRAWBERRY EXTRACTOR 
                    with dpg.group(horizontal=True, ):
                        dpg.add_text("Datacube to extract strawberries from: ")
                        dpg.add_combo(datacube_getter_for_extractor(), tag="datacubes_extractor_combobox", width=275, default_value=datacube_getter_for_extractor()[0])
                        dpg.add_button(label="Refresh",tag="refresh_extractor" , callback=lambda: refresh_comboboxes())
                    
                    dpg.add_button(label="Extract", callback=lambda: datacube_extractor.extractor(ui_context, dpg.get_value("datacubes_extractor_combobox")))

                # IMAGE VIEWER WINDOW
                dpg.add_text("Image Viewer")
                dpg.add_child_window(tag="image_extraction_window", autosize_x=True, height=130)
                with dpg.group(horizontal=False, parent="image_extraction_window"):
                    
                    # DATACUBE COMBOBOX AND TEXT
                    with dpg.group(horizontal=True):
                        dpg.add_text("Select a datacube to show: ")
                        dpg.add_combo(datacube_getter(), tag="datacubes_combobox", width=300, default_value=datacube_getter()[0])
                        dpg.add_button(label="Refresh",tag="refresh_imageviewer", callback=lambda: refresh_comboboxes())
                    
                    # NANOMETER RANGE SELECTION COMBOBOX
                    dpg.add_text("Default is band 50 and colormap magma.")
                    with dpg.group(horizontal=True):
                        dpg.add_text("Select a wavelength range - band: ")

                        #reading the bands.csv file
                        bands_file = open('bands/banden.csv','r', encoding='utf-8-sig')
                        bands = bands_file.read()
                        dropdown = bands.split(",")

                        dpg.add_combo(dropdown, tag="nm_dropdown", width=110, default_value='529.59 - 50')
                    
                    # COLORMAP SELECTION 
                    with dpg.group(horizontal=True):
                        cmaps = plt.colormaps()
                        dpg.add_text("Select a colormap")
                        dpg.add_combo(cmaps, tag="cmap_dropdown", width=100, default_value="magma")
                        
                    # FUNCTION BUTTONS
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Show Image", callback=lambda: image_extractor.extract_image(ui_context, dpg.get_value("datacubes_combobox"),dpg.get_value("nm_dropdown"), dpg.get_value("cmap_dropdown")))
                        dpg.add_button(label="Save Image", callback=lambda: image_extractor.save_image(ui_context, dpg.get_value("nm_dropdown"), dpg.get_value("cmap_dropdown")))
            
            # CONVEYOR BELT TAB
            with dpg.tab(label="Conveyor Belt"):
                conveyorbelt_header = dpg.add_text("Conveyor Belt")
                dpg.bind_item_font(conveyorbelt_header, large_font)
                dpg.add_text("Setup Conveyor belt")
                dpg.add_child_window(tag="conveyorbelt_window", autosize_x=True, height=80)
                with dpg.group(horizontal=False, parent="conveyorbelt_window"):
                    dpg.add_text("Select the COM port where the conveyor belt is connected")
                    with dpg.group(horizontal=True, parent="conveyorbelt_window"):
                        dpg.add_text("COM port: ")
                        dpg.add_combo(conveyorbelt_helper.get_comports(), tag="comport_combobox", width=275)                   
                        dpg.add_button(label="Refresh",tag="comport_refresh" , callback=lambda: refresh_combobox_comport())
                    dpg.add_button(label="Connect To Conveyor Belt",tag="conveyorbelt_connect_button" , callback=lambda: conveyorbelt_helper.connect_to_conveyor_belt(ui_context, dpg), parent="conveyorbelt_window")
            
            # AI TAB
            with dpg.tab(label="AI Classification"):
                classification_header = dpg.add_text("Classification")
                dpg.bind_item_font(classification_header, large_font)

                dpg.add_text("Instructions")
                dpg.add_child_window(tag="instructions_window", autosize_x=True, height=180)
                with dpg.group(horizontal=False, parent="instructions_window"):
                    # INSTRUCTIONS POPUP
                    with dpg.group(label="Instructions", horizontal=False, show=True, tag="instructions_popup"):
                        dpg.add_text("Instructions for Making a Classification", bullet=True)
                        
                        instructions = """
1. Prediction Categories
The system classifies each strawberry as one of:
- Fresh
- Old
- Spoiled

2. Preparing the System
- Connect to the camera: press "Connect to FX10" in HSI Camera Tab
- Connect to the conveyor belt: select the device in the dropdown and press "Connect to Conveyor Belt" in Conveyor Belt Tab

3. Placing the Strawberry
- Place a strawberry at the start of the conveyor belt, under the camera.

4. Environment Setup
- Close the curtains or blinds.
- Avoid direct light or reflections.

5. Start the Scan
- Press the 'Start Scan' button to begin scanning.

6. Stop the Scan
- Press 'Stop Scan' when the strawberry has been fully scanned.

7. Selecting the Datacube
- Refresh the dropdown menu after making a scan by pressing the 'Refresh' button.
- Select the datacube to predict from the dropdown menu.

8. Classify the Result
- Press 'Classificate' to make a prediction.
- The result will be shown in the 'Result window'.
- The certainty value indicates how confident the AI is in its prediction.
                    """

                        dpg.add_text(instructions, wrap=0)
                    
                dpg.add_spacer()
                dpg.add_text("Make a Classification")
                dpg.add_child_window(tag="classification_window", autosize_x=True, height=90)
                with dpg.group(horizontal=False, parent="classification_window"):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Start Scan", callback=lambda: camera_helper.start_datacube(ui_context, True))
                        dpg.add_button(label="Stop Scan", callback=lambda: camera_helper.stop_datacube(ui_context))
                    with dpg.group(horizontal=True, ):
                        dpg.add_text("Select a datacube to classify: ")
                        dpg.add_combo(datacube_getter_for_extractor(), tag="ai_datacubes_extractor_combobox", width=275, default_value=datacube_getter_for_extractor()[0])
                        dpg.add_button(label="Refresh",tag="ai_refresh_extractor" , callback=lambda: refresh_comboboxes())
                    dpg.add_button(label="Classificate", callback=lambda: strawberry_classifier.call_prediction(ui_context, dpg, dpg.get_value("ai_datacubes_extractor_combobox")))    

                # RESULT WINDOW
                dpg.add_text("Result")
                dpg.add_child_window(tag="result_window", autosize_x=True, height=35)
                with dpg.group(horizontal=False, parent="result_window"):
                    dpg.add_text("The result of the classification is: ", tag="ai_result")
                
                # # SETTINGS WINDOW
                # dpg.add_text("Settings")
                # dpg.add_child_window(tag="settings_window", autosize_x=True, height=35)
                # with dpg.group(horizontal=True, parent="settings_window"):
                #     dpg.add_text("AI Model : ")
                #     array = ["Forest", "CNN"]
                #     dpg.add_combo(array, default_value=array[1])

        # POP UP INSTRUCTIONS
        # Instructions popup (auto opens at app start)
        with dpg.window(label="Instructions", modal=True, show=True, tag="startup_instructions_popup", no_close=True, width=650, height=600):
            dpg.add_text("Instructions", bullet=True)
            dpg.add_separator()
            dpg.add_text(instructions, wrap=560)
            dpg.add_spacer(height=10)
            dpg.add_text("You can also find these instructions in the AI Classification Tab")
            dpg.add_button(label="Got it!", width=100, callback=lambda: dpg.configure_item("startup_instructions_popup", show=False))
                
        # LOG WINDOW
        dpg.add_text("Log:")   
        dpg.add_child_window(tag="log_window", autosize_x=True, height=100)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Clear Log", callback=lambda: clear_log())
            dpg.add_button(label="Close App", callback=lambda: camera_helper.close())
             
    # GENERAL SETTINGS OF GUI
    dpg.create_viewport(title='HSI GUI', width=700, height=710)
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
    
import glob

# This must be only whole datacube not crops, since the naming starts with data_cube we can filter by name.
def datacube_getter_for_extractor():
    files = sorted(glob.glob('DataCubes/**/dc_**.npy'))
    return files if files else [""]

# Gets the datacubes list from files and put them in a list for combobox to view
def datacube_getter():
    files = sorted(glob.glob('DataCubes/**/*.npy', recursive=True))
    return files if files else [""]

def refresh_comboboxes():
    new_items = datacube_getter()
    dpg.configure_item("datacubes_combobox", items=new_items)
    new_items = datacube_getter_for_extractor()
    dpg.configure_item("datacubes_extractor_combobox", items=new_items)
    dpg.configure_item("ai_datacubes_extractor_combobox", items = new_items)

def refresh_combobox_comport():
    dpg.configure_item("comport_combobox", default_value= "")
    items = conveyorbelt_helper.get_comports()
    dpg.configure_item("comport_combobox", items= items)