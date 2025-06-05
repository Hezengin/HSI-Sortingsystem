import serial

ser = serial.Serial("COM7", 115200)

# module is already connected so it is open no need to open it again
print(ser.is_open)
ser.close()
print(ser.is_open)
