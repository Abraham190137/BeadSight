import time
import serial

def send_command(ser, command:str, delay:float=1):
    ser.write((command + "\n").encode())
    print(command)
    time.sleep(delay)

ser = serial.Serial('/dev/ttyUSB1', 115200) 

send_command(ser, "G90; use absolute positioning")
send_command(ser, "G28; home all axes")

# make square
send_command(ser, "G0 X50 Y50 Z50", 30)
send_command(ser, "G0 X100 Y50 Z50", 10)
send_command(ser, "G0 X100 Y100 Z50", 10)
send_command(ser, "G0 X50 Y100 Z50", 10)
send_command(ser, "G0 X50 Y50 Z50", 10)

print("Done")
ser.close()


