// video_labeler.js

// Listen for the custom event 'videoFrameLoaded' before initializing the video player
document.addEventListener('videoFrameLoaded', function() {
    // Ensure the timestamp is defined before calling initVideoPlayer
    if (typeof timestamp !== 'undefined') {
        initVideoPlayer('videoPlayer', timestamp, 1); // 1 second window for looping
    } else {
        console.error('Timestamp is not defined.');
    }
});

function initVideoPlayer(htmlId, timestamp, windowSize) {
    let video = document.getElementById(htmlId);
    if (!video) {
        console.error('Video element not found:', htmlId);
        return;
    }
    video.currentTime = timestamp;
    video.play();

    // Clear any previous intervals to prevent multiple loops from being set up
    if (video.loopInterval) {
        clearInterval(video.loopInterval);
    }

    video.loopInterval = setInterval(function() {
        if (video.readyState > 2 && !video.seeking && video.currentTime >= timestamp + windowSize) {
            video.currentTime = timestamp;
        }
    }, 100); // A 100ms interval is more reasonable

    video.onended = function() {
        clearInterval(video.loopInterval);
    };
}

function pauseVideo(htmlId) {
    let video = document.getElementById(htmlId);
    if (video) {
        video.pause();
        if (video.loopInterval) {
            clearInterval(video.loopInterval);
        }
    }
}

// Function to play the video
function playVideo() {
    var video = document.getElementById('videoPlayer');
    video.play();
}

// Function to pause the video
function pauseVideo() {
    var video = document.getElementById('videoPlayer');
    video.pause();
}

// Event listeners for play and pause buttons
document.getElementById('playButton').addEventListener('click', playVideo);
document.getElementById('pauseButton').addEventListener('click', pauseVideo);

