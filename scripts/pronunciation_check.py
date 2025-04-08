import os
import random
import sounddevice as sd
import soundfile as sf
import whisper
import difflib

# Step 1: Pick a random sentence
def get_japanese_sentence(file="data/sentences.txt"):
    with open(file, encoding="utf-8") as f:
        return random.choice(f.readlines()).strip()

# Step 2: Record user's speech
def record_audio(out_path="data/raw/user.wav", duration=5, fs=16000):
    print(f"\n🎙 Recording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(out_path, recording, fs)
    print("✅ Recording done!")

# Step 3: Transcribe with Whisper
def transcribe_with_whisper(audio_path):
    print("\n🔍 Transcribing with Whisper...")
    model = whisper.load_model("base")  # base/small/medium/large
    result = model.transcribe(audio_path, language="ja")
    return result["text"]

# Step 4: Compare transcribed vs original
def compare_texts(original, recognized):
    similarity = difflib.SequenceMatcher(None, original, recognized).ratio()
    print("\n📘 Expected:   ", original)
    print("🗣️  You said:  ", recognized)
    print(f"\n✅ Accuracy: {similarity * 100:.2f}%")

    if similarity > 0.85:
        print("🎉 Good pronunciation!")
    else:
        print("❌ Needs improvement!")

# === MAIN ===
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    sentence = get_japanese_sentence()
    print("\n📝 Please read this sentence aloud:\n➡️", sentence)

    record_audio()
    user_text = transcribe_with_whisper("data/raw/user.wav")
    compare_texts(sentence, user_text)
