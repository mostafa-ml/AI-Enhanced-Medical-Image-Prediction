from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
import io
from models import BaseModel, Covid19Model, BrainTumorModel, KidneyStoneModel, SkinCancerModel, TuberculosisModel, BoneFractureModel

app = FastAPI()

async def predict_helper(file: UploadFile, model: BaseModel) -> Dict[str, Any]:
    """Helper function to handle predictions for all models."""
    try:
        # Read the uploaded file
        contents = await file.read()
        img = io.BytesIO(contents)

        # Run prediction
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

# Root endpoint
@app.get('/')
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "ONNX Models API is running!"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
