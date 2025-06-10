from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
from models import BaseModel, Covid19Model, BrainTumorModel,KidneyStoneModel, SkinCancerModel, \
TuberculosisModel, AlzheimerModel, DentalModel, EyeDiseasesModel, ColonDiseasesModel, OralDiseasesModel
from prescription import predict
from chatbot import MedicalChatbot
import os

app = FastAPI()

async def predict_helper(file: UploadFile, model: BaseModel) -> Dict[str, Any]:
    """Helper function to handle predictions for all models."""
    try:
        contents = await file.read()
        img = io.BytesIO(contents)

        res = model.predict(img)

        return {**res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/covid19/")
async def predict_covid19(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for COVID-19 predictions."""
    return await predict_helper(file, Covid19Model())

@app.post("/predict/brain-tumor/")
async def predict_brain_tumor(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Brain Tumor predictions."""
    return await predict_helper(file, BrainTumorModel())

@app.post("/predict/skin-cancer/")
async def predict_skin_cancer(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Skin Cancer predictions."""
    return await predict_helper(file, SkinCancerModel())

@app.post("/predict/kidney-stone/")
async def predict_kidney_stone(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Kidney Stone predictions."""
    return await predict_helper(file, KidneyStoneModel())

@app.post("/predict/tuberculosis/")
async def predict_tuberculosis(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Tuberculosis predictions."""
    return await predict_helper(file, TuberculosisModel())

@app.post("/predict/alzheimer/")
async def predict_alzheimer(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Alzheimer predictions."""
    return await predict_helper(file, AlzheimerModel())

@app.post("/predict/eye-diseases/")
async def predict_eye_diseases(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Eye Diseases predictions."""
    return await predict_helper(file, EyeDiseasesModel())

@app.post("/predict/colon-diseases/")
async def predict_colon_diseases(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Colon Diseases predictions."""
    return await predict_helper(file, ColonDiseasesModel())

@app.post("/predict/oral-diseases/")
async def predict_oral_diseases(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Oral Diseases predictions."""
    return await predict_helper(file, OralDiseasesModel())

@app.post("/predict/dental/")
async def predict_dental(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Dental predictions."""
    try:
        contents = await file.read()
        img = io.BytesIO(contents)

        res = DentalModel(img)

        return {**res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/prescription/")
async def predict_prescription(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for prescription predictions."""
    try:
        contents = await file.read()
        img = io.BytesIO(contents)

        res = predict(img)

        return {**res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(prompt: str):
    """Endpoint to interact with the chatbot."""
    chatbot = MedicalChatbot()
    return StreamingResponse(chatbot.get_chat_response(prompt), media_type="text/plain")

@app.get('/test-models')
async def test_models():
    """Test endpoint to check model paths and files."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        
        result = {
            "cwd": os.getcwd(),
            "base_dir": base_dir,
            "models_dir": models_dir,
            "models_dir_exists": os.path.exists(models_dir),
        }
        
        if os.path.exists(models_dir):
            result["models_files"] = os.listdir(models_dir)
            
        covid_model_path = os.path.join(models_dir, "covid-19.onnx")
        result["covid_model_path"] = covid_model_path
        result["covid_model_exists"] = os.path.exists(covid_model_path)
        
        return result
    except Exception as e:
        return {"error": str(e)}

# Root endpoint
@app.get('/')
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "API is running ✅"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)  # 127.0.0.1  8000
