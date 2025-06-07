from gcode_sender import ConveyorBeltConnection

cbc = ConveyorBeltConnection("COM6", 115200)
# cbc.send_gcode("IsXConveyor")
cbc.send_gcode("M310 1")
