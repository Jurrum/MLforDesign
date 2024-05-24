# Import necessary packages
import numpy as np
import pandas as pd
import pickle
from AI_for_Designers.data_processing import Preprocessing, empty_files
from AI_for_Designers.active_learning import ActiveLearning
from AI_for_Designers.novelty_detection import NoveltyDetection
from AI_for_Designers.data_stats import Stats

# Define the product name and check its format
Product = ''  # Enter the name of the product inside the quotation marks ('')
# check_product_name(Product)  # Function to check product name format

# Install necessary packages
# This block is commented out because package installation is typically done once and not included in a script
# !pip3 install numpy==1.21.4
# !pip3 install pandas==1.5.2
# ...

# Set the frame size and frame offset for preprocessing
frame_size = 2.0  # Example value, adjust as necessary
frame_offset = 0.2  # Example value, adjust as necessary

# Create and empty the datafiles for preprocessing
empty_files([f'Preprocessed-data/{Product}/features_{Product}.txt',
             f'Preprocessed-data/{Product}/features_{Product}_scaled.csv',
             f'Preprocessed-data/{Product}/processed_data_files.txt'])

# Preprocess the data
pre = Preprocessing(Product)
pre.windowing(input_file=[r'',  # Enter your accelerometer file
                          r''  # Enter your gyroscope file
                          ],
              video_file=r'',  # Enter the video file
              start_offset=0,
              stop_offset=0,
              skip_n_lines_at_start=0,
              video_offset=0,
              size=frame_size,
              offset=frame_offset)

# Scale features using the SuperStandardScaler
pre.SuperStandardScaler(fr'Preprocessed-data\{Product}\features_{Product}.txt')

# Define labels for active learning
labels = ['label1', 'label2']  # Modify as needed
active_learning_iterations = 100  # Set the number of active learning iterations

# Active learning
AL = ActiveLearning(fr'Preprocessed-data/{Product}/features_{Product}_scaled.csv', Product, labels, frame_size)
AL.training(active_learning_iterations)
AL.write_to_file()

# Testing the model
frames_to_test = 50  # Set the number of frames for testing
AL.testing(frames_to_test)

# Novelty Detection
ND = NoveltyDetection(fr'Preprocessed-data/{Product}/features_{Product}_scaled_AL_predictionss.csv', 
                      fr'Preprocessed-data/{Product}/processed_data_files.txt')
novelties = ND.detect(0.1)
ND.play_novelties(novelties, frame_size)

# Display product usage information
stats = Stats(fr'Preprocessed-data/{Product}/features_{Product}_scaled_AL_predictionss.csv', labels)
stats.print_percentages()
stats.show_ghan_chart(frame_offset)

# Process new data
# Import and preprocess new data as done above, and predict using the trained model
# ...

# End of script
print("End of the Machine Learning for IDE Students application script.")
