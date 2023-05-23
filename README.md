# Youtube Popularity Predictor

This is a machine learning project aimed at predicting the popularity of YouTube shorts videos. It takes video metadata, comments, and even analyses the audio content of a video to calculate a popularity score. The project is specifically designed to work with YouTube's API and Discord's Webhook API.

## Features

- **YouTube API Integration**: The project features YouTube API integration for data collection. The API is used to gather metadata like the video's length, publication date, the number of likes, comments, views, etc. It can also retrieve the audio file of the video, which is processed for audio feature extraction.

- **Data Collection**: It gathers video metadata and audio data from the YouTube API and stores it in a CSV file. The data is then preprocessed to remove any outliers and NaN values.

- **Audio Feature Extraction**: For every YouTube video, the audio is processed and features are extracted using Librosa. The features include spectral centroids, spectral rolloff, spectral fluxes, root mean square, zero-crossings, MFCCs, LPCs, chroma STFT, spectral contrast, and tonnetz. 

- **Popularity Score Calculation**: A custom function calculates the popularity score of each video based on the provided metadata and audio features.

- **Machine Learning**: An XGBRegressor model is trained using the video metadata and audio features to predict the popularity score. The model hyperparameters are tuned using grid search and cross-validation for optimal performance. The best model is then saved for future use.

- **Discord Notifications**: The script uses Discord's Webhook API to send notifications about the script's progress and any potential errors. It also sends a final summary of the model's training results, including the best hyperparameters and the mean absolute error of the predictions.

## How to Use

Please refer to the script comments and the `crawl_channel()` function to understand how to input a YouTube channel ID and start the data collection and model training process. The script requires a valid YouTube API Key and Discord Webhook URL to function. The Discord Webhook is used to get updates on model training results on the go.

## Requirements

You need to have Python 3.8.x installed along with the libraries in `requirements.txt`

These can be installed with the following command:

`pip install -r requirements.txt`

For any questions or issues, please open a GitHub issue or reach out via email.
