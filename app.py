import streamlit as st
import whisper
import tempfile
import os

st.title("Whisper Audio Transcription App 🎤")
st.write("Upload a long audio file (MP3, MPEG, WAV, M4A, OGG, etc.) and get an accurate transcript.")

# File uploader (includes mpeg)
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "mpeg", "wav", "ogg", "m4a"])

if uploaded_file is not None:
    # Keep original file extension for compatibility
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("⏳ Transcribing using Whisper Medium... please wait, this may take a while for long audios.")

    # Load Whisper medium model (high accuracy, ~769MB in size)
    model = whisper.load_model("small")

    # Transcribe full audio file
    result = model.transcribe(tmp_path, fp16=False)

    # Delete temp file
    os.remove(tmp_path)

    # Final transcript
    final_text = result["text"]

    st.subheader("Transcript:")
    st.text_area("Transcription", final_text, height=400)

    # Download button
    st.download_button("Download Transcript", final_text, file_name="transcript.txt")
