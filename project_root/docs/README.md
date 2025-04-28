# AI-Driven Fake News Detection System

## Project Overview

This project implements a comprehensive AI-driven system for detecting fake news articles. It utilizes advanced machine learning and deep learning techniques to classify news articles as either "fake" or "real." The system features a fully automated pipeline, encompassing data collection, preprocessing, feature engineering, model training, evaluation, deployment, explainability, continuous learning, testing, and CI/CD. This ensures minimal manual intervention and high accuracy, generalizability, and usability.

## Project Structure
```
project_root/
├── data/
│   ├── raw/                    # Raw data collected from APIs
│   └── processed/              # Preprocessed data
├── models/                 # Trained models
├── web/                 # web page
│   ├── index.html # main page
├── reports/                # Evaluation reports
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   └── fetch_data.py      # API data fetching
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── preprocess_text.py  # Text preprocessing
│   │   ├── preprocess_image.py # Image preprocessing
│   │   └── preprocess_video.py # Video preprocessing
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   └── extract_features.py # Feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_model.py       # Text-based model
│   │   ├── image_model.py      # Image-based model
│   │   ├── multimodal_model.py # Multimodal model
│   │   ├── train.py            # Model training
│   │   └── evaluate.py         # Model evaluation
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py              # API deployment (Flask/FastAPI)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py          # Helper functions
│   └── tests/
│       ├── __init__.py
│       ├── test_data_collection.py
│       ├── test_preprocessing.py
│       ├── test_models.py
│       │   ├── style.css # css style
│       └── test_api.py
├── docs/
│    └── README.md             # Project documentation
├── requirements.txt            # Project dependencies
└── .github/workflows/         # CI/CD configuration
```
## Setup Instructions

### 1. Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

#### Using `virtualenv`
```
bash
# Create a virtual environment
virtualenv venv

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Activate the environment (Windows)
venv\Scripts\activate
```
#### Using `conda`
```
bash
# Create a conda environment
conda create --name fake-news-env python=3.10

# Activate the environment
conda activate fake-news-env
```
### 2. Install Dependencies

Install the required Python packages using `pip`:
```
bash
pip install -r requirements.txt
```
### 3. API Keys

This project uses external APIs to collect news data. You'll need to obtain API keys for:

-   **X API (formerly Twitter)**:
    -   Go to [X Developer Platform](https://developer.x.com/)
    -   Create a developer account and a new project
    -   Generate API keys and tokens.
    - Set the environment variables: `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`.
-   **NewsAPI**:
    -   Go to [NewsAPI](https://newsapi.org/)
    -   Sign up for an account and get an API key.
    - Set the environment variable: `NEWSAPI_KEY`.

    -   **Facebook Graph API**
        -   Go to [Facebook Developers](https://developers.facebook.com/)
        -   Create an app and obtain the necessary permissions.
        - Set the environment variable: `FACEBOOK_ACCESS_TOKEN`

**Environment Variables Configuration:**
Once you have the keys you must set environment variables in your system.
- In Linux/macOS, add the exports to your `~/.bashrc` or `~/.zshrc`:
```
bash
export X_API_KEY="your_x_api_key"
export X_API_SECRET="your_x_api_secret"
export X_ACCESS_TOKEN="your_x_access_token"
export X_ACCESS_TOKEN_SECRET="your_x_access_token_secret"
export NEWSAPI_KEY="your_newsapi_key"
export FACEBOOK_ACCESS_TOKEN="your_facebook_access_token"
```
- In Windows, go to "System Properties" -> "Environment Variables" and create new system variables.

### 4. NLTK Data

If you use NLTK functionalities, you may need to download NLTK data:
```
bash
python -m nltk.downloader all
```
## How to Run the Pipeline

The project is designed to be fully automated. The pipeline includes the following steps:

1.  **Data Collection**: Fetching data from X and NewsAPI.
2.  **Data Preprocessing**: Cleaning, normalizing, and augmenting data.
3.  **Feature Engineering**: Extracting relevant features from text and metadata.
4.  **Model Training**: Training text and multimodal models.
5.  **Model Evaluation**: Assessing model performance using k-fold cross-validation.
6.  **API Deployment**: Launching the API for real-time classification.

To run the entire pipeline, execute the following (you need to write these in the scripts according to your needs):
```
bash
# Run data collection
python src/data_collection/fetch_data.py

# Run data preprocessing
python src/data_processing/preprocess_text.py
python src/data_processing/preprocess_image.py
python src/data_processing/preprocess_video.py

# Run feature engineering
python src/feature_engineering/extract_features.py

# Train the models
python src/models/train.py

# Evaluate the models
python src/models/evaluate.py

# Deploy the API
python src/api/app.py
```
This scripts will start the API on localhost.

## API Usage

The API allows you to classify news articles in real time.

### Starting the API

After running the automated pipeline to train the models, start the API using:
```
bash
python src/api/app.py
```
The API will be available at `http://127.0.0.1:5000` (default).

### API Endpoints

#### `/classify`

-   **Method**: `POST`
-   **Description**: Classifies a news article as fake or real.
-   **Request Body**:
```
json
    {
        "text": "The news article text goes here.",
        "url": "Optional: The news article URL"
    }
    
```
-   **Response**:
```
json
    {
        "classification": "fake" or "real",
        "explanation": "Explanation of the classification using LIME/SHAP",
        "confidence_score": 0.95
    }
    
```
#### `/health`
-   **Method**: `GET`
-   **Description**: Checks API health.
-   **Response**:
```
json
    {
        "status": "ok"
    }
    
```
### Example Request (using `curl`)
```
bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "This is a sample news article."}' http://127.0.0.1:5000/classify
```
### Example Request (using Python `requests` library)
```
python
import requests
import json

url = "http://127.0.0.1:5000/classify"
headers = {"Content-Type": "application/json"}
data = {"text": "This is a sample news article."}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```
## Example Use Cases

1.  **News Verification**: Users can paste a news article text or URL into the API to quickly determine if it is likely fake or real.
2.  **Social Media Monitoring**: The system can be integrated into social media platforms to automatically flag potential fake news posts.
3.  **Educational Tool**: The system can be used in educational settings to teach media literacy and critical thinking.
4.  **Journalism**: Journalists can use the API to quickly assess the credibility of news sources and articles.
5.  **Content moderation**: The system can be integrated in websites to identify potentially dangerous content.

## Continuous Learning

The system is designed for continuous learning. It includes a feedback loop to periodically retrain the model with new data. You can set up scripts to run these retraining tasks periodically or trigger them manually.

## Testing

The `tests/` directory contains unit and integration tests for all components. You can run these tests using `pytest`:
```
bash
pytest src/tests
```
## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and continuous deployment. The `.github/workflows/` directory contains the workflow configuration. The workflows will automatically:

-   Run tests on code changes.
-   Build the project.
-   Deploy the API on successful builds.
- Periodically retrain the model with new data.

## Dependencies

The required dependencies are listed in `requirements.txt`. Key libraries include:

-   **TensorFlow/PyTorch**: For deep learning models.
-   **scikit-learn**: For traditional machine learning models and evaluation.
-   **NLTK/spaCy**: For natural language processing.
-   **OpenCV/moviepy**: For image and video processing.
-   **Flask/FastAPI**: For API deployment.
-   **LIME/SHAP**: For model explainability.
-   **tweepy**: For interacting with the X API.
-   **optuna**: For hyperparameter tuning.
- **pytest**: For testing.

## Additional Notes

-   The system is designed for binary classification (fake/real) but can be extended for multi-class classification.
-   The project is modular, making it easy to modify or add new data sources or models.
-   Automation is prioritized to minimize manual intervention.
- Ensure you have the correct path for the different python files.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.