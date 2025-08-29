import streamlit as st
import whisper
import tempfile
import os

st.title("Whisper Audio Transcription App 🎤")
st.write("Upload a long audio file (MP3, WAV, M4A, OGG, etc.) and get an accurate transcript.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "m4a"])

if uploaded_file is not None:
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Transcribing... this may take a while depending on audio length.")

    # Load Whisper model (choose according to accuracy vs speed)
    # Options: tiny, base, small, medium, large
    model = whisper.load_model("base")  # change to "small" or "medium" if needed

    # Transcribe audio
    result = model.transcribe(tmp_path, fp16=False)

    # Delete temp file
    os.remove(tmp_path)

    # Show transcript
    final_text = result["text"]
    st.subheader("Transcript:")
    st.text_area("Transcription", final_text, height=400)

    # Download button
    st.download_button("Download Transcript", final_text, file_name="transcript.txt")
