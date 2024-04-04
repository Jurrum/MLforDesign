// script.js
document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('startButton');
    startButton.addEventListener('click', () => {
        window.location.href = "{{ url_for('setup') }}"; // Redirect to Setup page
    });

    const labelButtons = document.querySelectorAll('.label-button');

    labelButtons.forEach(button => {
        button.addEventListener('click', (event) => {
            const selectedLabel = event.target.getAttribute('data-label');
            // Perform actions with the selected label
            console.log("Label selected:", selectedLabel);
            // Here you can add logic to process the label, such as sending it to the server
        });
    });
});

let labelCount = 3;

function addLabelInput() {
    labelCount++;
    let labelDiv = document.getElementById('labelInputs');
    let newLabelDiv = document.createElement('div');
    newLabelDiv.id = `labelDiv_${labelCount}`;
    newLabelDiv.innerHTML = `<label for="label_${labelCount}">Label ${labelCount}:</label>
                             <input type="text" id="label_${labelCount}" name="label_${labelCount}"><br><br>`;
    labelDiv.appendChild(newLabelDiv);
}

function removeLabelInput() {
    if (labelCount > 3) {
        let labelDiv = document.getElementById('labelInputs');
        let lastLabelDiv = document.getElementById(`labelDiv_${labelCount}`);
        if (lastLabelDiv) {
            labelDiv.removeChild(lastLabelDiv);
            labelCount--;
        }
    }
}


document.getElementById('addLabelButton').addEventListener('click', () => {
    let labelName = prompt("Enter the name for the new label:");
    if (labelName) {
        let confirmed = confirm(`Is this the correct label name: ${labelName}?`);
        if (confirmed) {
            // Send the new label to the Flask server
            fetch('/add_label', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ label: labelName })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log("Label added successfully");

                    // Create a new button for the label with the "label-button" class
                    let newLabelButton = document.createElement('button');
                    newLabelButton.textContent = labelName;
                    newLabelButton.type = 'button';

                    // Add the "label-button" class to the button
                    newLabelButton.classList.add('label-button');

                    // Append the new button to your label list container
                    document.getElementById('labelListContainer').appendChild(newLabelButton);
                } else {
                    console.error("Error adding label");
                }
            })
            .catch(error => console.error('Error:', error));
        }
    }
});


function fetchAndDisplayFrame() {
    fetch('/get_video_frame')
        .then(response => response.json())
        .then(data => {
            // Update your frame display area with the received frame data
        });
}

document.getElementById('labelForm').addEventListener('submit', (event) => {
    event.preventDefault();
    const label = document.getElementById('label').value;
    fetch('/submit_label', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ label: label })
    })
    .then(() => {
        // Fetch and display the next frame
        fetchAndDisplayFrame();
    });
});

















