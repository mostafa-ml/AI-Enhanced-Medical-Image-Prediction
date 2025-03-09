from typing import Dict, Any
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io
import base64
import os

MODELS_DIR = r"models"
MODELS_PATHS: Dict[str, str] = {
    "Covid19Model": os.path.join(MODELS_DIR, "covid-19.onnx"),
    "BrainTumorModel": os.path.join(MODELS_DIR, "brain-tumor.onnx"),
    "SkinCancerSegmentationModel": os.path.join(MODELS_DIR, "skin-cancer-segmentation.onnx"),
    "SkinCancerClassificationModel": os.path.join(MODELS_DIR, "skin-cancer-classification.onnx"),
    "KidneyStoneModel": os.path.join(MODELS_DIR, "kidney-stone.onnx"),
    "TuberculosisModel": os.path.join(MODELS_DIR, "tuberculosis.onnx"),
    "BoneFractureModel": os.path.join(MODELS_DIR, "bone_fracture.onnx"),
    "AlzheimerModel": os.path.join(MODELS_DIR, "alzheimer.onnx"),
    "EyeDiseases": os.path.join(MODELS_DIR, "eye_diseases.onnx"),
    "Lumbar": os.path.join(MODELS_DIR, "Lumbar.onnx"),
    "Dental": os.path.join(MODELS_DIR, "dental_best_model.pt")
}

class BaseModel:
    def __init__(self, model_path: str):
        self.model = ort.InferenceSession(model_path)

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess the input image."""
        raise NotImplementedError("Preprocessing method must be implemented by subclass.")

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        """Postprocess the model's output."""
        raise NotImplementedError("Postprocessing method must be implemented by subclass.")

    def predict(self, img_file) -> Dict[str, Any]:
        """Run inference on the input image."""
        input_data = self.preprocess(img_file)
        pred = self.model.run(None, {self.model.get_inputs()[0].name: input_data})
        return self.postprocess(pred)

    def run_test(self, img_np) -> Dict[str, Any]:
        """Run inference test on the input image."""
        pred = self.model.run(None, {self.model.get_inputs()[0].name: img_np})
        return pred

class Covid19Model(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS["Covid19Model"])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the COVID-19 model."""
        img = Image.open(img_file)
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)   # Add batch dimension [None, 224, 224, 3]
        return img

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        """Postprocess predictions."""
        class_names = {0: 'COVID', 1: 'Lung Opacity', 2: 'Normal', 3: 'Viral Pneumonia'}
        pred_class = np.argmax(pred[0], axis=1)[0]
        confidence = int(np.max(pred[0], axis=1)[0] * 100_00) / 100
        return {"predicted_class": class_names[pred_class], "confidence": confidence}

class BrainTumorModel(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS["BrainTumorModel"])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the Brain Tumor model."""
        img = Image.open(img_file)
        img = img.convert("L")
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
        img = np.expand_dims(img, axis=0)   # Add batch dimension [None, 224, 224, 1]
        return img

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        """Postprocess predictions."""
        class_names = {0: 'Brain Tumor', 1: 'Normal'}
        pred_class = np.where(pred[0][0][0] > 0.5, 1, 0)
        if pred_class == 1:
            confidence = int(pred[0][0][0] * 100_00) / 100
        else:
            confidence = int((1 - pred[0][0][0]) * 100_00) / 100
        return {"predicted_class": class_names[int(pred_class)], "confidence": confidence}

class SkinCancerSegmentationModel(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS["SkinCancerSegmentationModel"])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the Skin Cancer Segmentation model."""
        img = Image.open(img_file)
        self.original_img = img  # Save the original image to be used in the postprocessing
        img = img.convert("RGB")
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)   # Add batch dimension [None, 256, 256, 3]
        return img

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        """Postprocess predictions."""
        mask = np.squeeze(pred[0][0])    # [256, 256]

        # Resize the mask to the original image size
        mask = Image.fromarray((mask * 255).astype(np.uint8))  # Convert to PIL Image
        mask = mask.resize(size=self.original_img.size, resample=Image.Resampling.NEAREST)  # Resize to original size
        mask = np.array(mask) / 255.0  # Normalize to [0, 1]

        fig, ax = plt.subplots()
        ax.imshow(np.array(self.original_img))
        ax.imshow(mask, cmap='Reds', alpha=0.5)
        ax.axis('off')

        # Save the figure to a BytesIO object
        buffered = io.BytesIO()
        plt.savefig(buffered, format='png', bbox_inches="tight", pad_inches=0)
        plt.close(fig)  # Close the figure to free memory

        # Encode the image as base64
        masked_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"detected_image": masked_image_base64}

class SkinCancerClassificationModel(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS["SkinCancerClassificationModel"])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the Skin Cancer Classification model."""
        img = Image.open(img_file)
        img = img.convert("RGB")
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)   # Add batch dimension [None, 256, 256, 3]
        return img

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        """Postprocess predictions."""
        class_names = {0: 'MEL (melanoma)', 1: 'NV (melanocytic nevi)', 2: 'BCC (basal cell carcinoma)',
                        3: 'AKIEC (Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease)',
                        4: 'BKL (benign keratosis-like lesions)', 5: 'DF (dermatofibroma)', 6: 'VASC (vascular lesions)'}
        pred_class = np.argmax(pred[0], axis=1)[0]
        confidence = int(np.max(pred[0], axis=1)[0] * 100_00) / 100
        return {"predicted_class": class_names[pred_class], "confidence": confidence}

class SkinCancerModel:
    def __init__(self):
        self.cls_model = SkinCancerClassificationModel()
        self.seg_model = SkinCancerSegmentationModel()

    def predict(self, img_file) -> Dict[str, Any]:
        """Run inference on the input image."""
        cls_res = self.cls_model.predict(img_file)
        seg_res = self.seg_model.predict(img_file)
        return {**cls_res, **seg_res}

class KidneyStoneModel(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS['KidneyStoneModel'])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the Kidney Stone model."""
        img = Image.open(img_file)
        img = img.convert("RGB")
        img = img.resize((640, 640))
        self.original_img = img  # Save the original image to be used in the postprocessing
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # Transpose the image from (H, W, C) to (C, H, W), required for PyTorch-style models.
        img = np.expand_dims(img, axis=0)   # Add batch dimension [1, 3, 640, 640]
        return img

    def postprocess(self, pred) -> np.ndarray:
        """Postprocess predictions."""
        class_names = {0: "Kidney Stone"}

        detections = pred[0][0]    # [N, 6] -> [x1, y1, x2, y2, confidence, cls]

        img = self.original_img.copy()
        draw = ImageDraw.Draw(img)

        confidence_threshold = 0.5
        iou_threshold = 0.5  # IoU threshold for NMS

        # Filter out low-confidence detections
        detections = [det for det in detections if det[4] > confidence_threshold]

        # Sort detections by confidence score in descending order
        detections.sort(key=lambda x: x[4], reverse=True)

        # Perform Non-Maximum Suppression (NMS)
        keep = []
        while len(detections) > 0:
            # Keep the detection with the highest confidence
            keep.append(detections[0])
            # Remove it from the list
            detections = detections[1:]
            # Compute IoU with the remaining detections
            ious = [self._compute_iou(keep[-1], det) for det in detections]
            # Remove detections with IoU > threshold
            detections = [det for i, det in enumerate(detections) if ious[i] < iou_threshold]

        for detection in keep:
            # Convert YOLO format (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = self._yolo_to_xyxy(box=detection[:-2])

            confidence, class_id = detection[-2:]

            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
            label = f"{class_names.get(int(class_id), 'Unknown')} {int((confidence * 100_00)) / 100}%"
            draw.text((x_min, y_min - 15), label, fill='red', font=ImageFont.truetype(font="arial.ttf", size=15, ))

        # Encode the image as a base64 string
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")  # Save the image to a BytesIO object
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")  # Encode as base64
        return {"detected_image": img_base64}

    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union (IoU) between two bounding boxes."""
        # Extract coordinates
        box1 = self._yolo_to_xyxy(box1)
        box2 = self._yolo_to_xyxy(box2)

        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
    
        # Compute IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def _yolo_to_xyxy(self, box):
        """Convert YOLO format (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)."""
        center_x, center_y, width, height = box[:4]
        return (
            center_x - width / 2,  # x_min
            center_y - height / 2,  # y_min
            center_x + width / 2,  # x_max
            center_y + height / 2,  # y_max
        )

class TuberculosisModel(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS["TuberculosisModel"])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the Tuberculosis Classification model."""
        img = Image.open(img_file)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img,dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)   # Add batch dimension [None, 224, 224, 3]
        return img

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        class_names = {0: "Normal", 1: "Tuberculosis"}

        pred_cls = np.where(pred[0][0][0] > 0.5, 1, 0)

        if pred_cls == 1:
            confidence = int(pred[0][0][0] * 100_00) / 100
        else:
            confidence = int((1 - pred[0][0][0]) * 100_00) / 100
        return {'predicted_class': class_names[int(pred_cls)], 'confidence': confidence}

class BoneFractureModel(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS["BoneFractureModel"])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the Tuberculosis Classification model."""
        img = Image.open(img_file)
        img = img.convert('RGB')
        img = img.resize((96, 96))
        img = np.array(img,dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)   # Add batch dimension [None, 224, 224, 3]
        return img

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        class_names = ["Elbow Positive", "Fingers Positive", "Forearm Fracture", "Humerus Fracture", "Shoulder Fracture", "Wrist Positive"]

        pred_cls = np.argmax(pred[0], axis=1)[0]
        confidence = int(np.max(pred[0], axis=1)[0] * 100_00) / 100

        return {'predicted_class': class_names[pred_cls], 'confidence': confidence}

class AlzheimerModel(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS["AlzheimerModel"])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the Alzheimer model."""
        img = Image.open(img_file)
        img = img.convert("RGB")
        img = img.resize((244, 244))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)   # Add batch dimension [None, 224, 224, 3]
        return img

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        """Postprocess predictions."""
        class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']
        pred_class = np.argmax(pred[0], axis=1)[0]
        confidence = int(np.max(pred[0], axis=1)[0] * 100_00) / 100
        return {"predicted_class": class_names[pred_class], "confidence": confidence}

class EyeDiseasesModel(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS["EyeDiseases"])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the Eye Diseases Model."""
        img = Image.open(img_file)
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)   # Add batch dimension [None, 224, 224, 3]
        return img

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        """Postprocess predictions."""
        class_names = {0: 'Choroidal Neovascularization (CNV)', 1: 'Diabetic Macular Edema (DME)', 2: 'DRUSEN', 3: 'NORMAL'}
        pred_class = np.argmax(pred[0], axis=1)[0]
        confidence = int(np.max(pred[0], axis=1)[0] * 100_00) / 100
        return {"predicted_class": class_names[pred_class], "confidence": confidence}

class KneeOsteoarthritisModel(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS["KneeOsteoarthritis"])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the Knee Osteoarthritis Model."""
        img = Image.open(img_file)
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = img.reshape([3, 224, 224])
        img = np.expand_dims(img, axis=0)   # Add batch dimension [None, 3, 224, 224]
        return img

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        """Postprocess predictions."""
        class_names = {
            0: 'Grade 0: Healthy knee',
            1: 'Grade 1 (Doubtful): Doubtful joint narrowing with possible osteophytic lipping',
            2: 'Grade 2 (Minimal): Definite presence of osteophytes and possible joint space narrowing',
            3: 'Grade 3 (Moderate): Multiple osteophytes, definite joint space narrowing, with mild sclerosis.',
            4:'Grade 4 (Severe): Large osteophytes, significant joint narrowing, and severe sclerosis.'
            }
        pred_class = np.argmax(pred[0], axis=1)[0]
        confidence = int(np.max(pred[0], axis=1)[0] * 100_00) / 100
        return {"predicted_class": class_names[pred_class], "confidence": confidence}


class LumbarModel(BaseModel):
    def __init__(self):
        super().__init__(MODELS_PATHS["Lumbar"])

    def preprocess(self, img_file) -> np.ndarray:
        """Preprocess an image for the Knee Osteoarthritis Model."""
        img = Image.open(img_file)
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)   # Add batch dimension [None, 224, 224, 3]
        return img

    def postprocess(self, pred: np.ndarray) -> Dict[str, Any]:
        """Postprocess predictions."""
        class_names = {
            0: 'lsd',
            1: 'OSF',
            2: 'spider',
            3: 'tseg',
            }
        pred_class = np.argmax(pred[0], axis=1)[0]
        confidence = int(np.max(pred[0], axis=1)[0] * 100_00) / 100
        return {"predicted_class": class_names[pred_class], "confidence": confidence}

def DentalModel(img_file):
    model = YOLO(MODELS_PATHS["Dental"])

    img = Image.open(img_file)
    img = img.convert("RGB")

    res = model.predict(source=img, save=False)

    annotated_image = res[0].plot()
    annotated_image_pil = Image.fromarray(annotated_image[..., ::-1])  # Convert BGR to RGB

    # Encode the image as a base64 string
    buffered = io.BytesIO()
    annotated_image_pil.save(buffered, format="PNG")  # Save the image to a BytesIO object
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")  # Encode as base64

    return {"detected_image": img_base64}
