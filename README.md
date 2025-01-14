# Document Classification Model

## Introduction

This project implements a Logistic Regression model for classifying PDFs of product catalogues into categories such as lighting, fuse, cables, and a negative class 'others'. The model achieves 99% overall accuracy and high recall on the dataset (currently private).

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

|-- .gitignore
|-- README.md
|-- requirements.txt
|-- inference_app
|   |-- app.py
|   |-- Dockerfile
|   |-- inference.py
|   |-- requirements.txt
|   |-- trained_model.pkl
|-- scripts
    |-- data_loader.py
    |-- infer.py
    |-- training.py
    |-- datasets
    |-- metrics
    |-- models

## Usage

### Quick Start

#### Training the Model

To train the model, run the following command:

`python scripts/training.py --train_json scripts/datasets/train_json.json`

#### Performing Inference

To perform inference using the trained model, run the following command:

`python scripts/infer.py --train_json scripts/datasets/train_json.json`

#### Creating Dataset JSON from Excel

To create a dataset JSON from an Excel file, run the following command:

`python scripts/data_loader.py --dataset_path path/to/dataset.xlsx`

This will generate train_json.json and test_json.json files in the datasets directory by default.

#### Building the Inference App

Navigate to the directory inferece_app. Run the dockerfile aptly named Dockerfile. It is necessary for the model's pickle file to be present in the app directory so that the container that gets created has the model copied to it as well.   

`docker build -f Dockerfile . -t document_classification_inference`

Run the container that is created with 

`docker run -p 5000:5000 document_classification_inference`

The Flask app is ready to use in a dev server. This app is not built with gunicorn (WIP) and therefore should not be used for production purposes.

#### Running the Inference App

A version of the app is hosted with AWS lightsail. To hit the API, use requests as shown below:

```python
import requests 
BASE = "https://flask-service-trials.8nfqzr06sjfd2.eu-north-1.cs.amazonlightsail.com/predict" 
url = "https://media.distributordatasolutions.com/synd_belden/2023q2/documents/cd68a6eb3f35cc315cc789d3bfa98de9efde9e75.pdf" 
response = requests.get(BASE, json = {"url":url}) 
print(response.json())
```

NB: the web server might not be available at all times, this is also not production ready.