import serial
import time
from util.log_levels import LogLevel
import serial.tools.list_ports

def get_comports():
    ports = serial.tools.list_ports.comports()
    items = []
    for port in ports:
        str = f"{port.device} - {port.description}"
        items.append(str)

    return items

def connect_to_conveyor_belt(ui_context, dpg):
    dpg = ui_context["dpg"]
    item_value = dpg.get_value("comport_combobox")

    if item_value:
        port = item_value.split(" - ")[0]  # Alleen COM3 uit "COM3 - description"
        conveyor = ConveyorBeltConnection(port, 115200, ui_context)
        conveyor.send_gcode("M310 0")
    else:
        ui_context["log_func"](LogLevel.WARNING, "No COM Port selected. Are you sure that's the one?")

class ConveyorBeltConnection:
    def __init__(self, port, baudrate, ui_context):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            if self.ser.is_open:
                ui_context["log_func"](LogLevel.INFO, f"Connected to: {self.ser.port}")
            else:
                ui_context["log_func"](LogLevel.ERROR, f"Serial not open after init.")
        except serial.SerialException as e:
            ui_context["log_func"](LogLevel.ERROR, f"Failed to connect to {port}: {e}")
            self.ser = None

    def send_gcode(self, command_str):
        if not self.ser or not self.ser.is_open:
            print("Serial connection not open.")
            return

        try:
            self.ser.write(command_str.encode("ascii"))
            print(f"Sent: {command_str.strip()}")
            time.sleep(0.1)
            response = self.ser.readline()
            if response:
                print(f"Received: {response.decode('ascii').strip()}")
            else:
                print("No response received.")
        except serial.SerialException as e:
            print(f"Serial communication error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")