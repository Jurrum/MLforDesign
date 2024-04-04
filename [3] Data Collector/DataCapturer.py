import serial
import time
import cv2


# Set up the serial connection (Change 'COM' to your Arduino's port)
ser = serial.Serial('COM8', 9600, timeout=1)
time.sleep(2) # Wait for the connection to settle


#Filepaths

filepath = "G:\\My Drive\\[-] Universiteit\\MSc - IPD\\[2] Jaar 1\\Q2 Research ML Design\\[3] Data Collector\\Database\\"
accelerometer_file_path = filepath + "accelerometer_data.csv"
gyroscope_file_path = filepath + "gyroscope_data.csv"
video_file_path = filepath + "video.mp4"
framerate = 20.0

# Function to write to a file
def write_to_file(path, data):
    with open(path, "a") as file:
        file.write(data + "\n")

# Initialize the camera
url = ''  # Change to the IP of a camera if a remote camera is used
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
out = cv2.VideoWriter(video_file_path, fourcc, framerate, resolution)

        
# Collect and process data
line_count = 0
start_time = None
line_limit = 500 # for 15 minutes this should be 180.000

try:
    while line_count < line_limit:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        if ret:
            out.write(frame)

        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            data = line.split(',')

            if len(data) == 4:  # Ensure correct data format
                timestamp = float(data[0])

                # Set start time to the first timestamp
                if start_time is None:
                    start_time = timestamp

                # Normalize the timestamp
                normalized_timestamp = timestamp - start_time

                x = data[1]
                y = data[2]
                z = data[3]

                # Alternate between accelerometer and gyroscope data
                if line_count % 2 == 0:
                    write_to_file(accelerometer_file_path, f"{normalized_timestamp},{x},{y},{z}")
                else:
                    write_to_file(gyroscope_file_path, f"{normalized_timestamp},{x},{y},{z}")

                line_count += 1

# Release the camera and file handles   
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release the camera and file handles
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    ser.close()