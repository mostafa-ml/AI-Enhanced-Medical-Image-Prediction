# AI-Enhanced Medical Image Prediction

## Overview
This project provides a FastAPI-based backend for medical image prediction using multiple deep learning models. The models have been trained on various medical datasets to detect conditions such as COVID-19, brain tumors, skin cancer, kidney stones, tuberculosis, bone fractures and more... To ensure efficient and stable inference, the models have been converted to the ONNX format((Open Neural Network Exchange)), making it easier to deploy them regardless of the original framework used for training.

## Features
- **Multiple Disease Detection**: Supports various medical conditions through different models.
- **ONNX Model Inference**: Enables fast and stable inference independent of the original deep learning framework.
- **FastAPI Deployment**: Provides a scalable and easy-to-use API for seamless integration to other applications.
- **Easy Integration**: The API abstracts model complexities, allowing backend teams to send images and receive predictions without ML expertise.
- **Comprehensive Documentation:**  Includes detailed instructions for setup, usage, and understanding the API's capabilities.

## Models Included
The following models are available for inference through the API:
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
   
   The models files are too large to be hosted directly on GitHub, so I’ve made them available via Google Drive.
   
   Download them from this [link](https://drive.google.com/drive/folders/1exyGxBjuVpFMFniDarKifTIDPFEiYZ_O?usp=sharing).

   After downloading, extract the files into a directory named models within the project directory.
   ```
   AI-Enhanced-Medical-Image-Prediction/
   ├── models/
   │   ├── covid-19.onnx
   │   ├── brain-tumor.onnx
   │   ├── ... (other model files)
   ├── main.py
   ├── models.py
   ├── ... (other files)
   ```

5. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

6. **Run the FastAPI server**
   ```sh
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

## API Usage
The API provides a consistent interface for interacting with all models.  All requests are POST requests, and image data is sent as a file upload.

#### URL = Base URL + Endpoint
   **Base URL:** `http://127.0.0.1:8000`
   
   **Endpoint:** `/predict/{model_name}`

#### Available Endpoints
| Endpoint                     | Description                                                                 | Method |
|------------------------------|-----------------------------------------------------------------------------|--------|
| `/predict/covid19/`          | Predicts COVID-19 from a chest X-ray image.                                 | `POST` |
| `/predict/brain-tumor/`      | Detects brain tumors in an MRI scan.                                        | `POST` |
| `/predict/kidney-stone/`     | Detects kidney stones from a medical CT or ultrasound scan.                 | `POST` |
| `/predict/skin-cancer/`      | Predicts skin cancer type (classification) and provides segmentation.       | `POST` |
| `/predict/tuberculosis/`     | Predicts tuberculosis from a chest X-ray image.                             | `POST` |
| `/predict/bone-fracture/`    | Predicts bone fracture type from an X-ray image.                            | `POST` |


## Making Requests with Postman
Sends an image to the specified model and gets a prediction.

1. **Open Postman and create a new request.**
2. **Set the request type to POST:**
   * Select POST from the dropdown list next to the URL bar.
3. **Enter the URL:**
   * For example, to send an image for the COVID-19 prediction, enter:
      ```sh
      http://127.0.0.1:8000/predict/covid19/
      ```
4. **Set the request body type to form-data:**
   * In the Postman interface, under the Body tab, select the form-data option.
   * This allows you to send the file as part of the form submission.
5. **Add the image file:**
   * In the form-data section, you'll see a Key column and a Value column.
   * Set the Key to file (this matches the parameter in the FastAPI endpoint).
   * For the Value, click on the Select Files button and choose the image file you want to send.
6. **Send the request:**
   * Once everything is set, click the Send button.
   * You should see the response from the FastAPI server in the lower part of Postman.

### see this image: 

![image](https://github.com/user-attachments/assets/fd5b1639-c08c-4599-a69f-d89f029f3b01)


## License
This project is licensed under the MIT License. See the LICENSE file for details.

