import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer

def predict_from_single_image(model_path, image_path, device=None):
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    # Load model and processor
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Make sure the tokenizer is properly set up
        tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
        processor.tokenizer = tokenizer
        
        # Set model to evaluation mode
        model.eval()
        
        # print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return e
    
    try:
        # Load original image
        original_image = Image.open(image_path).convert("RGB")
        
        # Convert to grayscale and back to RGB (to maintain 3-channel format)
        grayscale_image = original_image.convert("L").convert("RGB")
        
        # Process grayscale image for the model
        pixel_values = processor(grayscale_image, return_tensors="pt").pixel_values.to(device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            
        # Decode prediction
        predicted_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # print("\n----- RESULT -----")
        # print(f"Predicted text: {predicted_text}")
        
        return predicted_text
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return e

def predict(img_file):
    MODEL_PATH = r"models\prescription_model_weights"

    res = predict_from_single_image(MODEL_PATH, img_file)

    return {"Detected text": res}
