
#Importing all the local packages that have been made for this application.
from AI_for_Designers.data_processing import Preprocessing, empty_files
from AI_for_Designers.active_learning import ActiveLearning
from AI_for_Designers.novelty_detection import NoveltyDetection
from AI_for_Designers.data_stats import Stats
from AI_for_Designers.Notebook import check_product_name, amount_of_samples
import os


# Step for preprocessing the data
def preprocess_data(accelerometer_filename, gyroscope_filename, video_filename, frame_size, frame_offset, product_name):
    product_dir = f'Preprocessed-data/{product_name}'
    upload_folder = os.path.join('session_data', 'uploads')
    
    # Check that the product directory exists
    if not os.path.exists(product_dir):
        os.makedirs(product_dir)

    # Construct the full file paths
    accelerometer_file_path = os.path.join(upload_folder, accelerometer_filename)
    gyroscope_file_path = os.path.join(upload_folder, gyroscope_filename)
    video_file_path = os.path.join(upload_folder, video_filename) if video_filename else None
    


    # making the data files for the storage of the features, no need to change anything here.
    empty_files([f'Preprocessed-data/{product_name}/features_{product_name}.txt',
                f'Preprocessed-data/{product_name}/features_{product_name}_scaled.csv',
                f'Preprocessed-data/{product_name}/processed_data_files.txt'])

    product_dir = f'Preprocessed-data/{product_name}'
  
    # Make a preprocessing object that corresponds with the product
    pre = Preprocessing(product_name)

    # Insert the data into the preprocessing object.

    pre.windowing(input_file=[accelerometer_file_path, # Here you can enter your accelerometer file, only change what is in the brackets.
                            gyroscope_file_path  # Here you can enter your gyroscope file, only change what is in the brackets.
                ], 
                video_file= video_file_path, # Here you can enter the video corresponding to the datafiles.
                start_offset = 0, # Here you can enter the start offset for this file.
                stop_offset = 0, # Here you can enter the stop offset for this file.
                skip_n_lines_at_start = 0, # Here you can enter the number of lines you want to skip at the start of the data files
                video_offset = 0, # Here you can enter the video offset for this file
                size = frame_size, # This is the frame size, do not edit here.
                offset = frame_offset) # This is the frame offset, do not edit here.

    # Below you find an eample of extra datafile with video , you can uncomment this if needed or copy it.
    # When you have multiple datarecordings you can add more data to the preprocessing object. 
    # You can use the example code below and use it as amany times as you have datarecordings.


    # Initiate the scaler on the preprocessing object, in order to get scaled features. No need to change anything here.
    pre.SuperStandardScaler(fr'Preprocessed-data\{product_name}\features_{product_name}.txt') 

    print(amount_of_samples(f'Preprocessed-data/{product_name}/processed_data_files.txt'))
    pass


"""
Active learning stage

Here  a new instance for the active learning model is inititiated
"""
from AI_for_Designers.active_learning import ActiveLearning

def initiate_active_learning(product_name, frame_size, labels, active_learning_iterations,label_input_queue):
    # Assuming the path to the scaled features file is constructed like this
    features_file_path = fr'Preprocessed-data/{product_name}/features_{product_name}_scaled.csv'

    # Initialize the Active Learning object
    AL = ActiveLearning(features_file_path, product_name, labels, frame_size)
    AL.set_queue(label_input_queue)

    # Run the training process
    labels = AL.training(active_learning_iterations)

    # Save the results
    AL.write_to_file()
   
    # Return any data or confirmation as needed
    return AL
