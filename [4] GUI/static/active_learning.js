// active_learning.js
document.addEventListener('DOMContentLoaded', function() {
    fetch('/get_video_frame')
        .then(response => response.text())
        .then(htmlContent => {
            const videoFrameContainer = document.getElementById('video_frame_container');
            videoFrameContainer.innerHTML = htmlContent;
            
            // Retrieve the timestamp from the hidden div's data attribute
            const videoTimestampElement = videoFrameContainer.querySelector('#videoTimestamp');
            const timestamp = videoTimestampElement ? parseFloat(videoTimestampElement.dataset.timestamp) : null;
            
            if (timestamp) {
                // Now call the initVideoPlayer function with the retrieved timestamp
                initVideoPlayer('videoPlayer', timestamp, 1); // 1 second window for looping
            } else {
                console.error('Timestamp is not defined or invalid.');
            }
        })
        .catch(error => console.error('Error fetching video frame:', error));
});

