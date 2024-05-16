Forecasting System for Financial Data


Overview

This forecasting system utilizes monthly stock prices from the S&P 500 index to generate forecasts for the next period. It provides a user-friendly interface for selecting datasets, models, uploading data, and viewing forecast results, residuals, and accuracy metrics.



Dataset Selection & Description
The dataset comprises monthly stock prices from the S&P 500 index, offering comprehensive coverage across various industries and sectors. Monthly frequency strikes a balance between granularity and data volume, capturing meaningful trends while reducing noise compared to daily data.


Frontend Development
The frontend provides an intuitive interface for users to interact with different functionalities of the system. Key features include:
1. Selection of datasets and models
2. Uploading custom data
3. Viewing forecast results, residuals, and accuracy metrics
4. Error messages for issue resolution


JavaScript Functionality
The JavaScript file adds interactivity to the webpage, enabling users to generate forecasts based on selected datasets and models. Key functionalities include:
1. Listening for user clicks on the "Generate Forecast" button
2. Retrieving selected dataset and model values
3. Sending POST requests to the server using the Fetch API
4. Handling successful responses and errors gracefully
5. Placeholder function for handling file uploads (currently unused)


Flask Application (app.py)
The Flask application serves as the backend for the forecasting system, providing functionalities such as:
1. Loading pre-trained models and scalers for different model types
2. Generating forecasts based on selected dataset and model type
3. Flask routes for rendering the webpage, serving favicon, and generating forecasts
4. Error handling to catch and log exceptions during forecast generation

   
Database
SQLite is used to store the next 5 expected dates and their forecasted values according to the trend/pattern the model was trained by.


Usage
To run the system locally, follow these steps:

1. Clone the repository.
2. Install required dependencies (requirements.txt).
3. Open VS code  
4. Run the Flask application (app.py) on Terminal.
5. Access the system through a web browser.
