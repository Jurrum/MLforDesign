import serial
import time
import cv2
import os
import numpy as np

# Constants
SERIAL_PORT = 'COM8'
BAUD_RATE = 9600
LINE_LIMIT = 2000
BUFFER_SIZE = 64 * 1024  # Increase buffer size
FRAMERATE = 30.0

last_frame_capture_time = time.time()  # Store the time of the last frame capture

# Filepaths
BASE_DIRECTORY = os.path.dirname(__file__)
FILEPATH = os.path.join(BASE_DIRECTORY, 'Database')
accelerometer_file_path = os.path.join(FILEPATH, "accelerometer_data.csv")
gyroscope_file_path = os.path.join(FILEPATH, "gyroscope_data.csv")
video_file_path = os.path.join(FILEPATH, "video.mp4")

# Initialize the serial connection with a larger buffer
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# Get frame size from the camera
ret, frame = cap.read()
if not ret:
    print("Failed to capture frame for resolution determination")
    cap.release()
    exit()

height, width, _ = frame.shape
resolution = (width, height)

FRAME_DURATION_MS = 1000.0 / FRAMERATE  # Duration of one frame in milliseconds


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(video_file_path, fourcc, FRAMERATE, resolution)

# Open files for accelerometer and gyroscope data
accel_file = open(accelerometer_file_path, "w")
gyro_file = open(gyroscope_file_path, "w")

# Function to process serial data
def process_serial_data(line, start_time, timestamps, accel_file, gyro_file):
    prefix, timestamp, x, y, z = line.split(',')
    timestamp = convert_timestamp(float(timestamp))  # Convert the timestamp here
    if start_time is None:
        start_time = timestamp
    normalized_timestamp = timestamp - start_time
    timestamps.append(normalized_timestamp)
    line_data = f"{normalized_timestamp:.3f},{x},{y},{z}"
    if prefix == "ACC":
        accel_file.write(line_data)
    elif prefix == "GYR":
        gyro_file.write(line_data)
    return start_time

#  Convert Millis to seconds
def convert_timestamp(milliseconds):
    # Convert milliseconds to seconds and maintain three decimal places
    return float(f"{milliseconds / 1000:.3f}")

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
    partial_line = ""
    frame_count = 0
    last_frame_time = 0

    while line_count < LINE_LIMIT:
        # Read from serial and process data
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting).decode('utf-8', errors='replace')
            lines = (partial_line + data).split('\n')
            partial_line = lines[-1]  # Save incomplete line if any
            for line in lines[:-1]:  # Process complete lines
                if line.startswith(("ACC", "GYR")):
                    start_time = process_serial_data(line, start_time, timestamps, accel_file, gyro_file)
                    line_count += 1

                    # Check if it's time to capture a frame
                    current_time = time.time()
                    if (current_time - last_frame_capture_time) >= FRAME_DURATION_MS / 1000.0:
                        ret, frame = cap.read()
                        if ret:
                            out.write(frame)
                            frame_count += 1
                        last_frame_capture_time = current_time

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
