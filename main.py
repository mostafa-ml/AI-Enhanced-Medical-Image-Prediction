from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
from models import BaseModel, Covid19Model, BrainTumorModel,KidneyStoneModel, SkinCancerModel, \
TuberculosisModel, BoneFractureModel, AlzheimerModel, EyeDiseasesModel, DentalModel
from prescription import predict
from chatbot import MedicalChatbot

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

@app.post("/predict/bone-fracture/")
async def predict_bone_fracture(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Bone Fracture predictions."""
    return await predict_helper(file, BoneFractureModel())

@app.post("/predict/alzheimer/")
async def predict_alzheimer(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Alzheimer predictions."""
    return await predict_helper(file, AlzheimerModel())

@app.post("/predict/eye-diseases/")
async def predict_eye_diseases(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Endpoint specifically for Eye Diseases predictions."""
    return await predict_helper(file, EyeDiseasesModel())

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

@app.post("/chat/")
async def chat(prompt: str):
    """Endpoint to interact with the chatbot."""
    chatbot = MedicalChatbot()
    return StreamingResponse(chatbot.get_chat_response(prompt), media_type="text/plain")

# Root endpoint
@app.get('/')
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "API is running ✅"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8501) # 127.0.0.1 8000
