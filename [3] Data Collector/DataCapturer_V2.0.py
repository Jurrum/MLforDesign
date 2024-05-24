import serial
import time
import cv2
import os
import numpy as np


# Constants
SERIAL_PORT = 'COM8'
BAUD_RATE = 9600
LINE_LIMIT = 2000  # 10000 = 8:20 minutes
BASE_DIRECTORY = os.path.dirname(__file__)
FILEPATH = os.path.join(BASE_DIRECTORY, 'Database')
FRAMERATE = 30.0  # This can be a starting frame rate

# Filepaths
accelerometer_file_path = FILEPATH + "accelerometer_data.csv"
gyroscope_file_path = FILEPATH + "gyroscope_data.csv"
video_file_path = FILEPATH + "video.mp4"

# Function to write to a file
def write_to_file(file, data):
    file.write(data + "\n")

# Set up the serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Wait for the connection to settle

# Initialize the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# Get frame size from the camera
ret, frame = cap.read()
if ret:
    height, width, _ = frame.shape
    resolution = (width, height)
else:
    print("Failed to capture frame for resolution determination")
    cap.release()
    exit()    

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(video_file_path, fourcc, FRAMERATE, resolution)

# Open files for accelerometer and gyroscope data
accel_file = open(accelerometer_file_path, "w")
gyro_file = open(gyroscope_file_path, "w")

# Function to calculate frame rate based on sensor timestamps
def calculate_frame_rate(timestamps):
    global FRAMERATE
    if len(timestamps) < 2:
        return 30.0  # Default frame rate
    intervals = np.diff(timestamps)
    average_interval = np.mean(intervals)
    return 1.0 / average_interval

# Start a loop that saves the measurements and frames 
try:
    line_count = 0
    start_time = None
    timestamps = []

    while line_count < LINE_LIMIT:
        

        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Write the frame
        out.write(frame)

        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            prefix, timestamp, x, y, z = line.split(',')
            
            if prefix in ["ACC", "GYR"]:  # Ensure correct data format
                timestamp = float(timestamp)

                # Set start time to the first timestamp
                if start_time is None:
                    start_time = timestamp

                # Normalize the timestamp
                normalized_timestamp = timestamp - start_time
                timestamps.append(normalized_timestamp)

                # Calculate and adjust frame rate
                FRAMERATE = calculate_frame_rate(timestamps)
                
                line_data = f"{normalized_timestamp},{x},{y},{z}"
                print(LINE_LIMIT - line_count)
                # Write to appropriate file
                if prefix == "ACC":
                    write_to_file(accel_file, line_data)
                elif prefix == "GYR":
                    write_to_file(gyro_file, line_data)

                line_count += 1

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the camera and file handles
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    ser.close()
    accel_file.close()
    gyro_file.close()
