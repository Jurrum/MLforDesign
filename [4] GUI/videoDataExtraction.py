import av
import struct
import pandas as pd

def extract_gpmf_data(packet):
    # Extract GPMF data from packet
    gpmf_data = []
    while len(packet) >= 8:
        tag, size = struct.unpack('<4sI', packet[:8])
        data = packet[8:8+size]
        gpmf_data.append((tag, data))
        packet = packet[8+size:]
    return gpmf_data

def parse_gpmf_data(gpmf_data):
    # Parse GPMF data and extract accelerometer and gyroscope data
    accelerometer_data = []
    gyroscope_data = []
    for tag, data in gpmf_data:
        if tag == b'ACCL':
            for i in range(0, len(data), 6):
                x, y, z = struct.unpack('<hhh', data[i:i+6])
                accelerometer_data.append((x, y, z))
        elif tag == b'GYRO':
            for i in range(0, len(data), 6):
                x, y, z = struct.unpack('<hhh', data[i:i+6])
                gyroscope_data.append((x, y, z))
    return accelerometer_data, gyroscope_data

def extract_telemetry(video_path, output_accl_csv, output_gyro_csv):
    container = av.open(video_path)
    telemetry_stream = None

    # Identifying the telemetry stream
    for stream in container.streams:
        if stream.type == 'data':
            telemetry_stream = stream
            break

    if not telemetry_stream:
        print("No telemetry data stream found in the video file.")
        return

    acc_data = []
    gyro_data = []

    for packet in container.demux(telemetry_stream):
        # Extract GPMF data from packet
        gpmf_data = extract_gpmf_data(bytes(packet))
        # Parse GPMF data
        accelerometer_data, gyroscope_data = parse_gpmf_data(gpmf_data)
        acc_data.extend(accelerometer_data)
        gyro_data.extend(gyroscope_data)

    # Convert lists to DataFrame
    accl_df = pd.DataFrame(acc_data, columns=['x', 'y', 'z'])
    gyro_df = pd.DataFrame(gyro_data, columns=['x', 'y', 'z'])

    # Save the data to CSV files
    accl_df.to_csv(output_accl_csv, index=False)
    gyro_df.to_csv(output_gyro_csv, index=False)
    print(f"Data extracted and saved to {output_accl_csv} and {output_gyro_csv}")

if __name__ == "__main__":
    video_file_path = 'C:\\Users\\jbdbo\Desktop\\Q2 Research ML Design\\[2] Git\\Capstone-AI-IoT\\Data\\data-mok\\GL010035_LRV.mp4'  # Replace with your video file path
    output_accl_csv = 'Accelerometer_Data.csv'
    output_gyro_csv = 'Gyroscope_Data.csv'
    extract_telemetry(video_file_path, output_accl_csv, output_gyro_csv)
