import torch
import config
from scripts.preprocess import extract_mfcc
from models.pronunciation_model import PronunciationErrorDetection

# Load mô hình đã huấn luyện
model = PronunciationErrorDetection()
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

def predict_pronunciation(file_path):
    """Dự đoán lỗi phát âm dựa trên file âm thanh"""
    mfcc = extract_mfcc(file_path)
    
    with torch.no_grad():
        output = model(mfcc)
        predicted_class = torch.argmax(output, dim=1).item()

    return config.ERROR_TYPES[predicted_class]
