# AI-Enhanced-Medical-Image-Prediction

## Overview
This project provides a FastAPI-based backend for medical image prediction using multiple deep learning models. The models have been trained on various medical datasets to detect conditions such as COVID-19, brain tumors, skin cancer, kidney stones, tuberculosis, bone fractures and more... To ensure efficient and stable inference, the models have been converted to the ONNX format((Open Neural Network Exchange)), making it easier to deploy them regardless of the original framework used for training.

## Features
- **Multiple Disease Detection**: Supports various medical conditions through different models.
- **ONNX Model Inference**: Enables fast and stable inference independent of the original deep learning framework.
- **FastAPI Deployment**: Provides a simple and scalable API for integrating predictions into other applications.
- **Easy Integration**: The API abstracts model complexities, allowing backend teams to send images and receive predictions without ML expertise.

## Models Included
The following models are available for inference:
- **Covid-19 Model**: Classifies chest X-rays to detect COVID-19.
- **Brain Tumor Model**: Detects brain tumors in MRI scans.
- **Skin Cancer Model**: Detects and classifies skin lesions while segmenting cancerous regions.
- **Kidney Stone Model**: Detects kidney stones from medical CT scans.
- **Tuberculosis Model**: Identifies tuberculosis from chest X-rays.
- **Bone Fracture Model**: Detects bone fractures from X-ray images.

## Installation
To set up and run the API, follow these steps:

1. **Clone the repository**
   ```sh
   git clone https://github.com/mostafa-ml/AI-Enhanced-Medical-Image-Prediction
   ```

2. **Navigate to the Project Directory**
   ```sh
   cd AI-Enhanced-Medical-Image-Prediction
   ```

3. **Download the models**
   
   The models files are too large to be uploaded to GitHub, so I’ve made them available via Google Drive.
   
   Download them from this [link](https://drive.google.com/drive/folders/1exyGxBjuVpFMFniDarKifTIDPFEiYZ_O?usp=sharing).

   After downloading, extract the files into a folder named models.
   

5. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

6. **Run the FastAPI server**
   ```sh
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```

## API Usage
Once the server is running, you can interact with the API.

### 1. Predict an Image
**Endpoint:** `/predict/{model_name}`  
**Method:** `POST`  
**Description:** Sends an image to the specified model and gets a prediction.

#### Request:
```sh
curl -X POST "http://127.0.0.1:8000/predict/covid19/" -H "Content-Type: multipart/form-data" -F "file=@test images\Covid19\COVID-1024.png"
```

#### Response:
```json
{
  "predicted_class": "COVID",
  "confidence": 98.75
}
```

### 2. List Available Models
**Endpoint:** `/models`
**Method:** `GET`

#### Request:
```sh
curl -X 'GET' 'http://localhost:8000/models'
```

#### Response:
```json
{
  "available_models": [
    "Covid19Model",
    "BrainTumorModel",
    "SkinCancerSegmentationModel",
    "SkinCancerClassificationModel",
    "KidneyStoneModel",
    "TuberculosisModel",
    "BoneFractureModel"
  ]
}
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

