import serial
import time


# module is already connected so it is open no need to open it again
class ConveyorBeltConnection:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate

        ser = serial.Serial(port, baudrate)
        self.serial = ser
        

        if(ser.is_open == False):
            print("Serial not connected")

    def send_gcode(command_str):
        try:
            command = command_str
            self.serial.write(command.encode('ascii'))
            print(f"Sent: {command.strip()}")

            time.sleep(0.1)
            response = ser.readline()
            if response:
                print(f"Received: {response.decode('ascii').strip()}")
            else:
                print("No response received.")

        except serial.SerialException as e:
            print(f"Serial communication error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

