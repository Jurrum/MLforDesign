from flask import Flask, render_template, request, session, redirect, url_for, jsonify,  send_from_directory
from flask_session import Session
from flask import render_template
from werkzeug.utils import secure_filename
from queue import Queue
from threading import Event
from shutil import copy2

import shutil
import os
import threading

from machineLearningApp import preprocess_data, initiate_active_learning

app = Flask(__name__)
app.secret_key = 'BananaTree'
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'session_data')
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

active_learning_instance = None

# Set the number of OpenMP threads to avoid MKL memory leak on Windows
os.environ['OMP_NUM_THREADS'] = '3'

stop_active_learning = Event()

# Create a shared queue
label_input_queue = Queue()

# Utility Functions
def directory_exists(dir_path):
    return os.path.exists(dir_path) and os.path.isdir(dir_path)

def create_directory_if_not_exists(dir_path):
    if not directory_exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def clear_session_data():
    session_dir = app.config['SESSION_FILE_DIR']
    for file in os.listdir(session_dir):
        file_path = os.path.join(session_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def run_active_learning(product_name, frame_size, labels, iterations):
    
    # Start the process
    
    print(f"Starting active learning for {product_name}")
   
    initiate_active_learning(product_name, frame_size, labels, iterations, label_input_queue)

    print(f"Active learning finished {product_name}")

   


# Route Handlers
@app.route('/')
def home():
    session['completed_steps'] = ['home']
    return render_template('home.html')

@app.route('/start', methods=['POST'])
def start():
    session['completed_steps'] = ['home', 'setup']
    return redirect(url_for('setup'))

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    print("Accessed /setup route")  # Debug print

    if request.method == 'POST':
        # In the setup get the Name of the product, frame size and offset
        product_name = request.form.get('product_name')
        frame_offset = float(request.form.get('frame_offset'))
        frame_size = float(request.form.get('frame_size'))
        active_iterations = int(request.form.get('active_learning_iterations'))
        labels = [value for key, value in request.form.items() if key.startswith('label_') and value]

        # Store data in session
        session['product_name'] = product_name
        session['frame_size'] = frame_size
        session['frame_offset'] = frame_offset
        session['labels'] = labels
        session['active_learning_iterations'] = active_iterations
        session['label_submission_count'] = 0  # Initialize the counter


        print("Session Data:", session)  # Debug print
        print("the name and frames should now be stored")


        session['completed_steps'].append('setup')
        return redirect(url_for('preprocessing'))
    return render_template('setup.html')

def clear_directory(directory):
    """Removes all files in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    product_name = session.get('product_name')
    frame_size = session.get('frame_size')
    frame_offset = session.get('frame_offset')
    
    if request.method == 'POST':

        # Clear the plot images directory
        plot_dir = os.path.join(app.root_path, 'static', 'plots')
        clear_directory(plot_dir)
        
        print(request.files)  # Debug: Print the files received
        accelerometer_file = request.files.get('accelerometer_file')
        gyroscope_file = request.files.get('gyroscope_file')
        video_recording_file = request.files.get('video_recording_file')

        if accelerometer_file and accelerometer_file.filename and gyroscope_file and gyroscope_file.filename:
            #Check folder exists
            upload_folder = os.path.join(app.config['SESSION_FILE_DIR'], 'uploads')
            static_folder = os.path.join(app.root_path, 'static', 'session_data', 'uploads')
            create_directory_if_not_exists(upload_folder)
            create_directory_if_not_exists(static_folder)
            session['static_folder_path'] = static_folder

            # Secure the filenames and save the files
            accelerometer_filename = secure_filename(accelerometer_file.filename)
            gyroscope_filename = secure_filename(gyroscope_file.filename)
            
            # Save the files in the upload folder
            accelerometer_file_path = os.path.join(upload_folder, accelerometer_filename)
            gyroscope_file_path = os.path.join(upload_folder, gyroscope_filename)
            accelerometer_file.save(accelerometer_file_path)
            gyroscope_file.save(gyroscope_file_path)
            print('Saved files to upload folder correctly')

            # Copy the files to the static folder
            shutil.copy(accelerometer_file_path, os.path.join(static_folder, accelerometer_filename))
            shutil.copy(gyroscope_file_path, os.path.join(static_folder, gyroscope_filename))
            print('Saved files to static folder correctly')


            #If there is a video file (optional) it should be saved and copied as well   
            video_filename = None
            if video_recording_file and video_recording_file.filename:
                video_filename = secure_filename(video_recording_file.filename)
                video_file_path = os.path.join(upload_folder, video_filename)
                session['video_file_path'] = video_file_path
                session['video_filename'] = video_filename
                video_recording_file.save(video_file_path)
                shutil.copy(video_file_path, os.path.join(static_folder, video_filename))


            # Add functionality that preprocesses the data that is uploaded
            preprocess_data(accelerometer_filename, 
                            gyroscope_filename, 
                            video_filename, 
                            frame_size, 
                            frame_offset, 
                            product_name)

            # Redirect to the next page for active learning
            return redirect(url_for('active_learning'))
        else:
            print("The acc & gyr files are not uploaded...")


        session['completed_steps'].append('preprocessing')
    return render_template('preprocessing.html')



"""
Active learning route

"""
@app.route('/active_learning', methods=['GET', 'POST'])
def active_learning():
 
    labels = session.get('labels', [])
    active_learning_iterations = session.get('active_learning_iterations')
    product_name = session.get('product_name')
    frame_size = session.get('frame_size')

    # Check if the counter has reached the required number of iterations
    if session.get('label_submission_count', 0) >= active_learning_iterations:
        # If yes, redirect to the testing page
        session['completed_steps'].append('active_learning')  # Mark active learning as completed
        return render_template('testing.html', labels=session.get('labels', []))
    
    
    if request.method == 'GET':
        # Start Active Learning in a new thread
        active_thread = threading.Thread(target=run_active_learning, args=(
            product_name,
            frame_size,
            labels,
            active_learning_iterations
        ))
        active_thread.start()
        active_learning_instance
        
        # Return the Active Learning page while the model is being trained in the background
        return render_template('active_learning.html', labels=session.get('labels', []))
    

    # Active learning should be redirected to the testingpage automatically / a next phase button should appear
    if session.get('label_submission_count') >= active_learning_iterations:
        
        # Handle active learning submission...
        session['completed_steps'].append('active_learning')
        return render_template('testing.html', labels=session.get('labels', []))
    session.modified = True
    return render_template('active_learning.html', labels=labels)

@app.route('/stop_active_learning')
def stop_active_learning_route():
    stop_active_learning.set()
    return jsonify(success=True, message="Active learning stopping.")

# in the Active learning module a user should be able to add new labels
@app.route('/add_label', methods=['POST'])
def add_label():#
    data = request.get_json()
    new_label = data.get('label')
    if new_label:
        # Add the new label to your labels array in the session
        if 'labels' not in session:
            session['labels'] = []
        session['labels'].append(new_label)
        # Mark the session as modified to ensure it gets saved
        session.modified = True
        return jsonify(success=True)
    return jsonify(success=False), 400

@app.route('/label_input', methods=['POST'])
def label_input():
    label = request.json.get('label')
    print(label)
    
    # Add the label to the queue
    label_input_queue.put(label)

    active_learning_iterations = session.get('active_learning_iterations')
    
    # Increment the counter
    if 'label_submission_count' in session:
        session['label_submission_count'] += 1
        session.modified = True  # Make sure to mark the session as modified
    
    print("Label submission count:", session['label_submission_count'], '/', session.get('active_learning_iterations'))

    if session.get('label_submission_count') >= active_learning_iterations:
        
        # Handle active learning submission...
        session['completed_steps'].append('active_learning')
        return render_template('testing.html', labels=session.get('labels', []))

    # Check if the counter matches the required iterations
    if session['label_submission_count'] >= session.get('active_learning_iterations', 0):
        return jsonify(success=True, message="Label received", redirect_to=url_for('testing'))
    else:
        return jsonify(success=True, message="Label received")
    
@app.route('/check_completion')
def check_completion():
    if session.get('label_submission_count', 0) >= session.get('active_learning_iterations', 0):
        return jsonify(completed=True)
    return jsonify(completed=False)


"""
Route that creates the video frame for the active learning step
"""

@app.route('/get_video_frame')
def get_video_frame():
    # Get the name of the video from the session data
    video_filename = session.get('video_filename')
    if not video_filename:
        return 'Video file not found', 404

    # Read the timestamp
    session_data_dir = "session_data"
    timestamp_file_path = os.path.join(session_data_dir, "timestamp.txt")
    try:
        with open(timestamp_file_path, "r") as file:
            timestamp = float(file.read())
    except FileNotFoundError:
        return 'Timestamp not found', 404
    except ValueError:
        return 'Invalid timestamp format', 400

    # Generate video URL
    video_url = url_for('static', filename=f'session_data/uploads/{video_filename}')
    
    # HTML content for dynamic insertion
    video_html_content = f'''
        <video id="videoPlayer" height="300px" src="{video_url}" muted></video>
        <div id="videoTimestamp" data-timestamp="{timestamp}"></div>
        <script>
            // Pass the timestamp to the global scope for video_labeler.js to use
            window.timestamp = {timestamp}; // Define the timestamp here
        </script>
    '''

    return video_html_content


@app.route('/get_plot')
def get_plot():
    try:
        # Source directory where the plots are initially located
        source_folder = os.path.join(app.root_path, '..', 'session_data', 'plots')

        print("Source folder:", source_folder)
        
        # Destination directory inside the static folder where the plots should be copied to
        dest_folder = os.path.join(app.root_path, 'static', 'plots')
        
        # Ensure the destination directory exists
        os.makedirs(dest_folder, exist_ok=True)
        
        # Find the latest plot image in the source directory
        plot_files = [f for f in os.listdir(source_folder) if f.startswith('plot_to_label') and f.endswith('.png')]
        if not plot_files:
            return jsonify(error="No plot file found"), 404
       
        
        latest_plot = max(plot_files, key=lambda x: os.path.getmtime(os.path.join(source_folder, x)))
        source_file_path = os.path.join(source_folder, latest_plot)
        dest_file_path = os.path.join(dest_folder, latest_plot)
        
        # Copy the latest plot file to the static directory
        copy2(source_file_path, dest_file_path)
        
        # Generate the URL for the image
        image_url = url_for('static', filename=f'plots/{latest_plot}')
        return jsonify(image_url=image_url)
    except OSError as e:
        return jsonify(error=f"Error accessing or moving the plot file: {e}"), 500




@app.route('/get_current_plot_filename', methods=['GET'])
def get_current_plot_filename():
    # Retrieve the filename from the session
    filename = session.get('current_plot_filename', 'default_plot_filename.png')
    print("Current plot filename:", filename)  # For debugging
    return jsonify(filename=filename)

    
@app.route('/submit_label', methods=['POST'])
def submit_label():
    global active_learning_instance

    if active_learning_instance:
        label = request.json.get('label')
        # Update your model with the new label
        active_learning_instance.update_with_label(label)
        return jsonify(success=True)
    else:
        return jsonify(error="Active Learning instance not initialized"), 400

"""
Testing route
 - Here the user will be shown the initial results from the active learning phase
 - The clusters have been formed and a video playback will be shown with the predicted labels
 - The user will see whether the model has been able to predict the labels correctly
"""

@app.route('/testing', methods=['GET', 'POST'])
def testing():
    # Load the URL for the video uploaded in the session
    video_url = url_for('static', filename=f'session_data/uploads/{session.get("video_filename")}')

    product_name = session.get("product_name")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Preprocessed-data', product_name))

    source_file_path = os.path.join(base_dir, f'features_{product_name}_scaled_AL_predictions.csv')

    target_dir = os.path.join(app.static_folder, 'session_data/results')
    target_file_path = os.path.join(target_dir, f'features_{product_name}_scaled_AL_predictions.csv')

    # Check if the source file exists
    if os.path.exists(source_file_path):
        # Ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)
        # Copy the file to the target directory
        copy2(source_file_path, target_file_path)
        print("File copied successfully.")
    else:
        print(f"Source file does not exist at {source_file_path}")

    # Now the file is in the static directory, generate the URL for the copied file
    timestamp_url = url_for('static', filename=f'session_data/results/features_{product_name}_scaled_AL_predictions.csv')
    
    if request.method == 'POST':
        # Handle testing submission...
        session['completed_steps'].append('testing')
        return redirect(url_for('novelty_detection'))
    
    return render_template('testing.html', video_url=video_url, timestamp_url=timestamp_url)



@app.route('/novelty_detection', methods=['GET', 'POST'])
def novelty_detection():
    if request.method == 'POST':
        # Handle novelty detection submission...
        session['completed_steps'].append('novelty_detection')
        return redirect(url_for('analytics'))
    return render_template('novelty_detection.html')

@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    if request.method == 'POST':
        # Handle analytics submission...
        session['completed_steps'].append('analytics')
        return redirect(url_for('new_data'))
    return render_template('analytics.html')

@app.route('/new_data', methods=['GET', 'POST'])
def new_data():
    if request.method == 'POST':
        # Handle new data submission...
        session['completed_steps'].append('new_data')
        # Decide where to redirect after completing the new data step
        return redirect(url_for('home'))
    return render_template('new_data.html')



# Run the Application
if __name__ == '__main__':
    app.run(debug=True)