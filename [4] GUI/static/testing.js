document.addEventListener('DOMContentLoaded', (event) => {
    const videoContainer = document.getElementById('videoContainer');
    const videoUrl = videoContainer.getAttribute('data-video-url');
    const labelDisplay = document.getElementById('labelDisplay');
    const labelsUrl = labelDisplay.getAttribute('data-csv-url');
    const videoTimer = document.getElementById('videoTimer');
    const videoTimelineContainer = document.getElementById('videoTimelineContainer');

    // Create the video element dynamically
    const videoPlayer = document.createElement('video');
    videoPlayer.setAttribute('id', 'videoPlayer');
    videoPlayer.setAttribute('width', '640');
    videoPlayer.setAttribute('height', '480');
    videoPlayer.src = videoUrl;
    videoPlayer.muted = true; // Mute the video to allow autoplay in most browsers
    videoContainer.appendChild(videoPlayer);

    // Buttons
    const playButton = document.getElementById('playButton');
    const pauseButton = document.getElementById('pauseButton');

    playButton.addEventListener('click', () => videoPlayer.play());
    pauseButton.addEventListener('click', () => videoPlayer.pause());

    // Example label colors
    const labelColors = {};

    function assignLabelColors(labels) {
        // Add more colors as needed if there are more than 10 labels
        const baseColors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink', 'teal']; // Add more as needed
        labels.forEach((label, index) => {
            if (!labelColors[label.label]) {
                labelColors[label.label] = baseColors[index % baseColors.length];
            }
        });
    }

    function createLabelTimeline(labels) {
        const timelineContainer = document.getElementById('labelTimelineContainer');
        const fragment = document.createDocumentFragment();

        // Apply median filter
        const smoothedLabels = medianFilter(labels);

        smoothedLabels.forEach((label, index) => {
            const nextLabel = smoothedLabels[index + 1];
            const duration = nextLabel ? nextLabel.time - label.time : videoPlayer.duration - label.time;
            const labelDiv = document.createElement('div');
            labelDiv.classList.add('label-segment');
            labelDiv.style.position = 'absolute';
            labelDiv.style.left = `${(label.time / videoPlayer.duration) * 100}%`;
            labelDiv.style.width = `${(duration / videoPlayer.duration) * 100}%`;
            labelDiv.style.backgroundColor = labelColors[label.label] || '#000'; // Fallback color if not defined
            fragment.appendChild(labelDiv);
        });

        timelineContainer.appendChild(fragment);
    }

    function medianFilter(labels) {
        const windowSize = 3; // Adjust window size as needed
        const halfWindowSize = Math.floor(windowSize / 2);
        const smoothedLabels = [];

        for (let i = 0; i < labels.length; i++) {
            const startIndex = Math.max(0, i - halfWindowSize);
            const endIndex = Math.min(labels.length - 1, i + halfWindowSize);
            const window = labels.slice(startIndex, endIndex + 1);
            const medianLabel = getMedianLabel(window);
            smoothedLabels.push(medianLabel);
        }

        return smoothedLabels;
    }

    function getMedianLabel(labels) {
        const sortedLabels = labels.slice().sort((a, b) => a.time - b.time);
        const medianIndex = Math.floor(sortedLabels.length / 2);
        return sortedLabels[medianIndex];
    }

    function createLegend() {
        const legendContainer = document.getElementById('labelLegend');
        for (const label in labelColors) {
            const colorBox = document.createElement('span');
            colorBox.style.display = 'inline-block';
            colorBox.style.width = '20px';
            colorBox.style.height = '20px';
            colorBox.style.backgroundColor = labelColors[label];
            colorBox.style.marginRight = '5px';
            const textNode = document.createTextNode(label);
            const legendItem = document.createElement('div');
            legendItem.appendChild(colorBox);
            legendItem.appendChild(textNode);
            legendContainer.appendChild(legendItem);
        }
    }

    function createInsightsTable(labels) {
        const tableContainer = document.getElementById('insightsTable');
        const uniqueLabels = Array.from(new Set(labels.map(label => label.label)));

        const table = document.createElement('table');
        table.classList.add('insights-table');

        // Create table header
        const headerRow = document.createElement('tr');
        const headers = ['Label', 'Occurrence', 'Avg Length (s)', '%'];
        headers.forEach(headerText => {
            const headerCell = document.createElement('th');
            headerCell.textContent = headerText;
            headerRow.appendChild(headerCell);
        });
        table.appendChild(headerRow);

        // Create table rows
        uniqueLabels.forEach(label => {
            const labelOccurrences = labels.filter(item => item.label === label).length;
            const averageStreakLength = calculateAverageStreakLength(labels, label);
            const percentageOfTotal = (labelOccurrences / labels.length) * 100;

            const row = document.createElement('tr');
            const cells = [label, labelOccurrences, averageStreakLength, percentageOfTotal.toFixed(2) + '%'];
            cells.forEach(cellText => {
                const cell = document.createElement('td');
                cell.textContent = cellText;
                row.appendChild(cell);
            });
            table.appendChild(row);
        });

        tableContainer.appendChild(table);
    }

    function calculateAverageStreakLength(labels, label) {
        const streaks = [];
        let currentStreak = 0;

        labels.forEach((item, index) => {
            if (item.label === label) {
                currentStreak++;
            } else if (currentStreak > 0) {
                streaks.push(currentStreak);
                currentStreak = 0;
            }
            if (index === labels.length - 1 && currentStreak > 0) {
                streaks.push(currentStreak);
            }
        });

        const averageStreakLength = streaks.reduce((acc, val) => acc + val, 0) / streaks.length;
        return averageStreakLength.toFixed(2);
    }

    // Load labels and sync with video
    let labels = [];
    fetch(labelsUrl)
        .then(response => response.text())
        .then(csvText => {
            labels = parseCSV(csvText);
            assignLabelColors(labels); // Assign colors after parsing the labels
            createLabelTimeline(labels); // Create the timeline after colors have been assigned
            createLegend(); // Create the legend after the timeline
            createInsightsTable(labels); // Create insights table
        })
        .catch(error => console.error('Error loading or parsing labels:', error));
  
    // Parse CSV function
    function parseCSV(csvText) {
        const lines = csvText.trim().split("\n").slice(1); // Skipping the header
        return lines.map(line => {
            const parts = line.split(",");
            // Assuming 'label' is at index 1 and 'time' is at index 2 based on your record
            const label = parts[1].trim();
            const time = parseFloat(parts[2].trim());
            return { label, time };
        });
    }

    // Update the video timer, label display, and scrubber position
    videoPlayer.addEventListener('timeupdate', () => {
        const currentTime = videoPlayer.currentTime;
        const currentMinutes = Math.floor(currentTime / 60);
        const currentSeconds = (currentTime % 60).toFixed(2);

        // Update the timer display
        videoTimer.innerText = `Time: ${currentMinutes.toString().padStart(2, '0')}:${currentSeconds.toString().padStart(5, '0')}`;

        // Update the label display
        const currentLabel = labels.find(label => currentTime >= label.time && (!labels[labels.indexOf(label) + 1] || currentTime < labels[labels.indexOf(label) + 1].time));
        if (currentLabel) {
            labelDisplay.innerText = currentLabel.label;
        }
        
        // Update the timeline and scrubber
        const percent = (currentTime / videoPlayer.duration) * 100;
        const videoTimeline = document.getElementById('videoTimeline');
        videoTimeline.style.width = `${percent}%`;
        const scrubber = document.getElementById('scrubber');
        scrubber.style.left = `${percent}%`;
    });

    // Click on the timeline to seek within the video
    videoTimelineContainer.addEventListener('click', (e) => {
        const timelineWidth = videoTimelineContainer.offsetWidth;
        const offsetX = e.offsetX;
        const duration = videoPlayer.duration;
        const clickPosition = offsetX / timelineWidth;
        videoPlayer.currentTime = clickPosition * duration;

        // Update the scrubber position immediately
        const scrubber = document.getElementById('scrubber');
        scrubber.style.left = `${(clickPosition * 100).toFixed(2)}%`;
    });

});
