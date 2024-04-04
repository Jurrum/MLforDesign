// label_script.js

function updateVideoFrame() {
    fetch('/get_video_frame')
        .then(response => response.text())
        .then(htmlContent => {
            const videoFrameContainer = document.getElementById('video_frame_container');
            videoFrameContainer.innerHTML = htmlContent;

            const videoTimestampElement = videoFrameContainer.querySelector('#videoTimestamp');
            const timestamp = videoTimestampElement ? parseFloat(videoTimestampElement.dataset.timestamp) : null;

            if (timestamp) {
                initVideoPlayer('videoPlayer', timestamp, 1); // 1 second window for looping
            } else {
                console.error('Timestamp is not defined or invalid.');
            }
        })
        .catch(error => console.error('Error fetching video frame:', error));
}

// Function to load the plot image
function loadPlotImage() {
    fetch('/get_plot')
    .then(response => response.json())
    .then(data => {
        if(data.image_url) {
            const plotImage = document.getElementById('plotImage');
            plotImage.src = data.image_url;
        } else {
            console.error('No image found or multiple images exist.');
        }
    })
    .catch(error => console.error('Error:', error));
}




document.addEventListener('DOMContentLoaded', function() {
    const labelListContainer = document.getElementById('labelListContainer');

    labelListContainer.addEventListener('click', function(event) {
        if (event.target && event.target.classList.contains('label-button')) {
            const labelValue = event.target.value;

            // Send an AJAX request to Flask to submit the label
            fetch('/label_input', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ label: labelValue })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Response:', data);
                updateVideoFrame(); // Update the video frame after the label is submitted
                loadPlotImage(); // Load the plot image when a label is submitted
            })
            .catch(error => console.error('Error:', error));
        }
    });
    
    });
