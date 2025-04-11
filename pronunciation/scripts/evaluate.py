import torch
import pandas as pd
import librosa
from sklearn.metrics import accuracy_score

# Load mô hình đã train
model = PronunciationErrorDetection()
model.load_state_dict(torch.load("../models/pronunciation_model.pth"))
model.eval()

# Load file test_labels.csv
test_data = pd.read_csv("../data/test_labels.csv")

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return torch.tensor(mfccs).unsqueeze(0)

# Dự đoán trên tập test
all_preds, all_labels = [], []

for _, row in test_data.iterrows():
    file_path = f"../data/test/{row['filename']}"
    mfcc = extract_mfcc(file_path)
    
    with torch.no_grad():
        output = model(mfcc.unsqueeze(0).float())
        pred = torch.argmax(output, dim=1).item()
    
    all_preds.append(pred)
    all_labels.append(row["label"])

# Tính độ chính xác
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
