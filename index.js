document.getElementById('forecast-button').addEventListener('click', function(event) {
    event.preventDefault(); // Prevent default form submission if within a form
    generate_forecast();
});

function generate_forecast() {
    const dataset = document.getElementById('dataset-dropdown').value;
    const model = document.getElementById('model-dropdown').value;

    console.log("Sending dataset:", dataset); // Debug: Log dataset
    console.log("Sending model:", model);     // Debug: Log model

    fetch('/generate_forecast', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dataset: dataset, model: model })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error); // Handle server-side errors
        }
        const tableBody = document.getElementById('forecast-table').getElementsByTagName('tbody')[0];
        tableBody.innerHTML = ""; // Clear previous results
        data.forecast.forEach(item => {
            let row = tableBody.insertRow();
            let dateCell = row.insertCell(0);
            dateCell.innerHTML = item.date;
            let valueCell = row.insertCell(1);
            valueCell.innerHTML = item.forecasted_values; // Make sure this matches the key sent from Flask
        });
    })
    .catch(error => {
        console.error('Failed to fetch:', error);
        document.getElementById('forecast-output').innerHTML = 'Failed to generate forecast: ' + error.message;
    });
}

function fetch_stored_forecasts() {
    const dataset = document.getElementById('dataset-dropdown').value;
    const model = document.getElementById('model-dropdown').value;

    fetch(`/get_forecasts?dataset=${dataset}&model=${model}`)
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error); // Handle server-side errors
        }
        const tableBody = document.getElementById('stored-forecast-table').getElementsByTagName('tbody')[0];
        tableBody.innerHTML = ""; // Clear previous results
        data.forEach(item => {
            let row = tableBody.insertRow();
            let dateCell = row.insertCell(0);
            dateCell.innerHTML = item.date;
            let valueCell = row.insertCell(1);
            valueCell.innerHTML = item.forecasted_value; // Make sure this matches the key sent from Flask
        });
    })
    .catch(error => {
        console.error('Failed to fetch:', error);
        document.getElementById('forecast-output').innerHTML = 'Failed to fetch stored forecasts: ' + error.message;
    });
}

function handleUpload() {
    const file = document.getElementById('data-file').files[0];
    // Handle file upload here, possibly send to Flask to retrain models or update datasets
}
