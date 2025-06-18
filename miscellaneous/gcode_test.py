from util.conveyorbelt_helper import ConveyorBeltConnection

cbc = ConveyorBeltConnection("COM5", 115200)
# cbc.send_gcode("IsXConveyor")
cbc.send_gcode("M310 1")