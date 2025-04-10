import sounddevice as sd
import soundfile as sf
import whisper
import difflib
import random
import os

# 1. Lấy ngẫu nhiên 1 câu từ data/sentences.txt
def get_random_sentence(file_path="data/sentences.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return random.choice(lines).strip()

# 2. Ghi âm giọng nói người dùng
def record_audio(filename="data/raw/user_audio.wav", duration=5, samplerate=16000):
    print(f"🎙 Đang ghi âm trong {duration} giây... Hãy đọc câu hiển thị.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print("✅ Ghi âm hoàn tất!")

# 3. Dùng Whisper để chuyển âm thanh sang văn bản
def transcribe_audio(filename="data/raw/user_audio.wav"):
    print("🧠 Đang chuyển giọng nói sang văn bản bằng Whisper...")
    model = whisper.load_model("base")  # hoặc "small", "medium" nếu cần chính xác hơn
    result = model.transcribe(filename, language="ja")
    return result["text"]

# 4. So sánh độ giống nhau giữa câu chuẩn và câu người học đọc
def compare_sentences(original, recognized):
    ratio = difflib.SequenceMatcher(None, original, recognized).ratio()
    print(f"\n📝 Câu chuẩn    : {original}")
    print(f"🗣️  Người học đọc: {recognized}")
    print(f"🎯 Độ chính xác  : {ratio*100:.2f}%")
    if ratio > 0.8:
        print("✅ Phát âm khá chính xác!")
    else:
        print("❌ Cần cải thiện phát âm.")

# === MAIN ===
if __name__ == "__main__":
    sentence = get_random_sentence()
    print(f"\n📢 Hãy đọc câu sau bằng tiếng Nhật:\n➡️  {sentence}")

    os.makedirs("data/raw", exist_ok=True)
    record_audio()
    result_text = transcribe_audio()
    compare_sentences(sentence, result_text)
