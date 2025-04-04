from flask import Flask, request, jsonify
import whisper
import difflib
import os

app = Flask(__name__)

# Load Whisper model
model = whisper.load_model("base")  # Chọn model phù hợp

@app.route("/compare", methods=["POST"])
def compare():
    """Nhận file âm thanh, chuyển thành văn bản bằng Whisper, rồi so sánh với câu gốc."""
    if "audio" not in request.files:
        return jsonify({"error": "Không có file âm thanh"}), 400

    audio = request.files["audio"]  # Lấy file từ người dùng
    original = request.form["original"]  # Câu gốc mà hệ thống hiển thị

    audio_path = "data/test/audio.wav"
    audio.save(audio_path)  # Lưu file để xử lý

    # Dùng Whisper nhận dạng giọng nói
    result = model.transcribe(audio_path, language="ja")
    recognized = result["text"].strip()

    # So sánh phát âm bằng difflib
    similarity = difflib.SequenceMatcher(None, original, recognized).ratio()

    # Đánh giá độ chính xác
    if similarity > 0.8:
        feedback = "✅ Phát âm khá đúng!"
    elif similarity > 0.5:
        feedback = "⚠️ Có một số lỗi nhỏ."
    else:
        feedback = "❌ Phát âm chưa đúng, cần luyện tập thêm!"

    return jsonify({"recognized_text": recognized, "similarity": similarity, "feedback": feedback})

if __name__ == "__main__":
    app.run(debug=True)
