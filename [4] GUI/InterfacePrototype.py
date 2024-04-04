from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session 
from werkzeug.utils import secure_filename
import os
import threading


app = Flask(__name__)
app.secret_key = 'BananaTree'
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'session_data')
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)


# Define a dictionary mapping route names to their respective HTML files
pages = {
    'home': 'home.html',
    'setup': 'setup.html',
    'preprocessing': 'preprocessing.html',
    'active_learning': 'active_learning.html',
    'testing': 'testing.html',
    'novelty_detection': 'novelty_detection.html',
    'analytics': 'analytics.html',
    'new_data': 'new_data.html'
}

# Initialize session variables
@app.before_first_request
def initialize_session():
    session['completed_steps'] = ['home']  # Initialize with 'home' as the completed step

# Function to handle logic for each route
def handle_route(page_name):
    if request.method == 'POST':
        if page_name == 'setup':
            # Example: Process form data from setup page
            product_name = request.form.get('product_name')
            frame_offset = request.form.get('frame_offset')
            frame_size = request.form.get('frame_size')
        if page_name == 'preprocessing':
            # Handle file upload in preprocessing step
            accelerometer_file = request.files.get('accelerometer_file')
            gyroscope_file = request.files.get('gyroscope_file')
                # Additional files as needed

            if accelerometer_file and gyroscope_file:
                accelerometer_filename = secure_filename(accelerometer_file.filename)
                gyroscope_filename = secure_filename(gyroscope_file.filename)

                # Ensure your upload folder exists
                upload_folder = os.path.join(app.config['SESSION_FILE_DIR'], 'uploads')
                os.makedirs(upload_folder, exist_ok=True)

                # Save files
                accelerometer_file.save(os.path.join(upload_folder, accelerometer_filename))
                gyroscope_file.save(os.path.join(upload_folder, gyroscope_filename))

            
            print("this is before the redirect from preprocessing")
            # Redirect to the next page after preprocessing
            return redirect(url_for('active_learning'))
            print("this is after the redirect of preprocessing")

        # General logic for updating completed steps for other pages
        completed_steps = session.setdefault('completed_steps', [])
        if page_name not in completed_steps:
            completed_steps.append(page_name)
            session.modified = True
  
        
    # Check if the current step is in the completed steps list
    if page_name != 'home' and page_name not in session.get('completed_steps', []):
        return redirect(url_for('home'))

    return render_template(pages[page_name])

# Dynamically create routes for each page except 'home'
for route in pages.keys():
    if route != 'home':
        app.add_url_rule(
            f'/{route}', 
            endpoint=route, 
            view_func=lambda route=route: handle_route(route),
            methods=['GET', 'POST']
        )

# Separate route for 'home'
@app.route('/')
def home():
    return render_template('home.html')

# Route for starting the process
@app.route('/start', methods=['POST'])
def start():
    # Reset or initialize the completed steps list
    session['completed_steps'] = ['home']
    return redirect(url_for('setup'))


# Check if a folder exists
def directory_exists(dir_path):
    """Check if a directory exists."""
    return os.path.exists(dir_path) and os.path.isdir(dir_path)

# Create a new directory
def create_directory_if_not_exists(dir_path):
    """Create a directory if it does not exist."""
    if not directory_exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

# Clear session data
def clear_session_data():
    session_dir = app.config['SESSION_FILE_DIR']
    for file in os.listdir(session_dir):
        file_path = os.path.join(session_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}") 


if __name__ == '__main__':
    app.run(debug=True)