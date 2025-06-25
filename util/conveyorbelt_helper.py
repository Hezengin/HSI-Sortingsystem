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
        port = item_value.split(" - ")[0]  # Pak alleen COMx
        ui_context["log_func"](LogLevel.INFO, f"Connecting to port: {port}")

        conveyor = ConveyorBeltConnection(port, 115200, ui_context)
        if conveyor.ser and conveyor.ser.is_open:
            ui_context["conveyor"] = conveyor
            conveyor.send_gcode("M310 1")  # Zet apparaat in serial mode
        else:
            ui_context["log_func"](LogLevel.ERROR, f"Connection to {port} failed.")
    else:
        ui_context["log_func"](LogLevel.WARNING, "No COM Port selected. Are you sure that's the one?")

class ConveyorBeltConnection:
    def __init__(self, port, baudrate, ui_context):
        self.ser = None
        self.ui_context = ui_context

        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(1)  # Kleine delay om apparaat wakker te maken

            if self.ser.is_open:
                ui_context["log_func"](LogLevel.INFO, f"Connected to: {self.ser.port}")
            else:
                ui_context["log_func"](LogLevel.ERROR, f"Serial not open after init.")
                self.ser = None
        except serial.SerialException as e:
            ui_context["log_func"](LogLevel.ERROR, f"Failed to connect to {port}: {e}")
            self.ser = None
        except Exception as e:
            ui_context["log_func"](LogLevel.ERROR, f"Unexpected error while connecting: {e}")
            self.ser = None

    def send_gcode(self, command_str):
            if not self.ser or not self.ser.is_open:
                print("Serial connection not open.")
                self.ui_context["log_func"](LogLevel.WARNING, "Serial connection is not open.")
                return

            try:
                full_command = command_str.strip() + "\r\n"  # Belangrijk: line ending
                self.ser.write(full_command.encode("ascii"))
                print(f"Sent: {command_str.strip()}")

                time.sleep(0.2)  # Wacht op reactie van het apparaat
                response = self.ser.readline()
                if response:
                    decoded = response.decode("ascii", errors="ignore").strip()
                    print(f"Received: {decoded}")
                    self.ui_context["log_func"](LogLevel.INFO, f"Conveyor replied: {decoded}")
                else:
                    print("No response received.")
                    self.ui_context["log_func"](LogLevel.WARNING, "No response received from conveyor.")
            except serial.SerialException as e:
                print(f"Serial communication error: {e}")
                self.ui_context["log_func"](LogLevel.ERROR, f"Serial error: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
                self.ui_context["log_func"](LogLevel.ERROR, f"Unexpected send_gcode error: {e}")
