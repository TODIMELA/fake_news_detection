from fastapi import FastAPI, HTTPException, Request, Form
from pydantic import BaseModel
import numpy as np
from lime.lime_text import LimeTextExplainer
import shap
import joblib
from src.models.text_model import BertClassifier, RobertaClassifier, DistilBertClassifier
from src.data_processing.preprocess_text import TextPreprocessor
from src.feature_engineering.extract_features import FeatureExtractor
from fastapi.responses import JSONResponse
import json
import torch
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path


web_dir = Path(__file__).parent.parent.parent / "web"
app = FastAPI()

# Model loading
MODEL_TYPE = "bert"  # Default model type, can be changed
try:
    if MODEL_TYPE == "bert":
        model = BertClassifier(num_labels=2, use_attention=True)
    elif MODEL_TYPE == "roberta":
        model = RobertaClassifier(num_labels=2, use_attention=True)
    elif MODEL_TYPE == "distilbert":
        model = DistilBertClassifier(num_labels=2, use_attention=True)
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported")
    
    model.load_state_dict(torch.load(f"models/text_model_{MODEL_TYPE}.pth"))
    model.eval()
except Exception as e:
    raise Exception(f"Error loading model: {e}")
    
try:
    vectorizer = joblib.load(f"models/tfidf_vectorizer_{MODEL_TYPE}.joblib")
except Exception as e:
    raise Exception(f"Error loading vectorizer: {e}")
    
try:
    preprocessor = TextPreprocessor()
except Exception as e:
    raise Exception(f"Error loading preprocessor: {e}")

app.mount("/static", StaticFiles(directory=web_dir), name="static")
templates = Jinja2Templates(directory=web_dir)

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=["real", "fake"])

# Initialize SHAP explainer
feature_extractor = FeatureExtractor()
shap_explainer = shap.Explainer(lambda x: model(torch.tensor(x["input_ids"]), torch.tensor(x["attention_mask"]))[0].detach().numpy(), feature_extractor.tfidf)


# Function to get LIME explanations
def get_lime_explanation(text, model, vectorizer):
    def predict_proba_lime(texts):
        # Process the texts using the same steps as in classification
        processed_texts = [preprocessor.preprocess(t) for t in texts]
        tfidf_matrix, _ = feature_extractor.tfidf(processed_texts)
        with torch.no_grad():
            predictions = model(torch.tensor(tfidf_matrix.toarray()).float(), torch.ones_like(torch.tensor(tfidf_matrix.toarray())[:, :1]).float())[0]
            return torch.softmax(predictions, dim=1).numpy()

    explanation = explainer.explain_instance(text, predict_proba_lime, num_features=10)
    return explanation.as_list()

# Function to get SHAP explanation
def get_shap_explanation(text, model, vectorizer):
    processed_text = preprocessor.preprocess(text)
    tfidf_matrix, _ = feature_extractor.tfidf([processed_text])
    with torch.no_grad():
        shap_values = shap_explainer(tfidf_matrix)
    return [{"feature": shap_values.feature_names[i], "value": shap_values.values[0, i]} for i in range(len(shap_values.feature_names))]


# Endpoint to classify news
@app.post("/classify/")  
async def classify_news(request: Request, text: str = Form(...), source: str = Form(...)):
    try:
        # Preprocess the text and extract features
        processed_text = preprocessor.preprocess(text)
        tfidf_matrix, _ = feature_extractor.tfidf([processed_text])
        input_ids = torch.tensor(tfidf_matrix.toarray()).float()
        attention_mask = torch.ones_like(input_ids[:, :1]).float()

        # Predict and get prediction probabilities
        with torch.no_grad():
            predictions, attentions = model(input_ids, attention_mask)
            prediction_probabilities = torch.softmax(predictions, dim=1)
            prediction_class = torch.argmax(prediction_probabilities, dim=1).item()
        
        # Get explanations
        lime_explanation = get_lime_explanation(processed_text, model, vectorizer)  
        shap_explanation = get_shap_explanation(processed_text, model, vectorizer) 

        # Prepare response data
        result = "fake" if prediction_class == 1 else "real"
        probability_fake = prediction_probabilities[0][1].item()
        probability_real = prediction_probabilities[0][0].item()

        return {
            "result": result,
            "probability_fake": probability_fake,
            "probability_real": probability_real,
            "lime_explanation": lime_explanation, 
            "shap_explanation": shap_explanation, 
            "animation": result,
            "highlight": [item["feature"] for item in shap_explanation if item["value"] > 0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/results/")
async def results(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})


# Instructions for local deployment (add to your README or documentation)
"""
## Local Deployment Instructions

1.  **Install Dependencies**:
```
bash
    pip install -r requirements.txt
    
```
2.  **Run the API**:
```
bash
    uvicorn api.app:app --reload
    
```
Replace `api.app` with the correct path to this file if it is not in the `api` directory.

3.  **Test the API**:
    -   Send a POST request to `http://127.0.0.1:8000/classify/` with a JSON payload like:
```
json
        {
            "text": "This is a test news article."
        }
        
```
-   You can use tools like `curl`, `Postman`, or a Python `requests` script to test the API.

4. **Stop the api**
    Press `Ctrl+C`

"""