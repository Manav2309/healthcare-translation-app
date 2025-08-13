import streamlit as st
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import requests
import json
import os
from gtts import gTTS
import tempfile
import base64
from io import BytesIO
import time
from llm_config import openrouter_config

# Page configuration
st.set_page_config(
    page_title="Healthcare Translation App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Language mapping for gTTS
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Russian": "ru"
}

class SoundDeviceSource:
    """
    Custom audio source for speech_recognition using sounddevice
    """
    def __init__(self, device_index=None, sample_rate=16000, chunk_size=1024):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_data = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def record(self, duration=5):
        """
        Record audio for specified duration
        """
        try:
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=self.device_index
            )
            sd.wait()  # Wait for recording to complete
            
            # Convert to the format expected by speech_recognition
            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            return sr.AudioData(
                audio_int16.tobytes(),
                self.sample_rate,
                2  # 2 bytes per sample for int16
            )
            
        except Exception as e:
            raise sr.RequestError(f"Could not record audio: {e}")

def check_remote_config():
    """
    Check remote configuration for kill switch
    Returns True if app should continue, False if disabled
    """
    try:
        # Replace with your actual config URL (e.g., GitHub Gist raw URL)
        config_url = "https://raw.githubusercontent.com/yourusername/config/main/app_config.json"
        
        response = requests.get(config_url, timeout=5)
        if response.status_code == 200:
            config = response.json()
            return config.get("active", True), config.get("message", "Service running normally.")
        else:
            # If config fetch fails, allow app to run (fail-safe)
            return True, "Service running normally."
    except Exception as e:
        # If config fetch fails, allow app to run (fail-safe)
        st.warning(f"Could not fetch remote config: {str(e)}. Running in local mode.")
        return True, "Service running normally."

def get_available_devices():
    """
    Get list of available audio input devices
    """
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device['name']))
        return input_devices
    except Exception:
        return [(None, "Default Device")]

def speech_to_text(device_index=None, duration=5):
    """
    Convert speech to text using speech_recognition library with sounddevice
    """
    recognizer = sr.Recognizer()
    
    try:
        # Use custom sounddevice source
        source = SoundDeviceSource(device_index=device_index)
        
        st.info(f"🎤 Recording for {duration} seconds... Speak now!")
        
        # Record audio
        audio = source.record(duration=duration)
        
        st.info("🔄 Processing speech...")
        
        # Use Google Speech Recognition
        text = recognizer.recognize_google(audio)
        return text
    
    except sr.UnknownValueError:
        st.error("🤷 Could not understand the audio. Please speak clearly.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition service error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        return None

def text_to_speech(text, lang_code="en"):
    """
    Convert text to speech using gTTS and return audio bytes
    """
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Create a BytesIO object to store audio
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.getvalue()
    
    except Exception as e:
        st.error(f"❌ Text-to-speech error: {str(e)}")
        return None

def create_audio_player(audio_bytes):
    """
    Create an audio player widget for Streamlit
    """
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

def main():
    """
    Main Streamlit application
    """
    # Check remote kill switch
    is_active, message = check_remote_config()
    
    if not is_active:
        st.error("🚫 Service temporarily disabled by administrator")
        st.info(message)
        st.stop()
    
    # App header
    st.title("🏥 Healthcare Translation App")
    st.markdown("*Real-time multilingual communication for healthcare providers and patients*")
    st.markdown("*Powered by OpenRouter & GPT-4o*")
    
    # Check API availability
    api_available, api_message = openrouter_config.check_api_availability()
    if not api_available:
        st.error(f"❌ API Issue: {api_message}")
        st.info("Please check your internet connection and API configuration.")
    else:
        st.success(f"✅ {api_message}")
    
    # Initialize session state
    if 'original_text' not in st.session_state:
        st.session_state.original_text = ""
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = ""
    
    # Audio device selection (in sidebar)
    with st.sidebar:
        st.header("🎤 Audio Settings")
        
        # Get available devices
        devices = get_available_devices()
        device_options = [f"{name}" for idx, name in devices]
        device_indices = [idx for idx, name in devices]
        
        selected_device_name = st.selectbox(
            "Select microphone:",
            device_options,
            index=0
        )
        
        selected_device_index = device_indices[device_options.index(selected_device_name)]
        
        # Recording duration
        recording_duration = st.slider(
            "Recording duration (seconds):",
            min_value=3,
            max_value=30,
            value=5,
            step=1
        )
    
    # Language selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗣️ Source Language")
        source_lang = st.selectbox(
            "Select source language:",
            list(LANGUAGE_CODES.keys()),
            index=0,
            key="source_lang"
        )
    
    with col2:
        st.subheader("🌐 Target Language")
        target_lang = st.selectbox(
            "Select target language:",
            list(LANGUAGE_CODES.keys()),
            index=1,
            key="target_lang"
        )
    
    st.markdown("---")
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📝 Original Text ({source_lang})")
        
        # Voice input button
        if st.button("🎤 Start Recording", type="primary", use_container_width=True):
            with st.spinner("Recording and processing..."):
                text = speech_to_text(
                    device_index=selected_device_index,
                    duration=recording_duration
                )
                if text:
                    st.session_state.original_text = text
                    st.success("✅ Speech captured successfully!")
        
        # Text input area
        original_text = st.text_area(
            "Or type your text here:",
            value=st.session_state.original_text,
            height=200,
            key="original_input"
        )
        
        if original_text != st.session_state.original_text:
            st.session_state.original_text = original_text
        
        # Speak original text
        if st.session_state.original_text and st.button("🔊 Speak Original", use_container_width=True):
            source_code = LANGUAGE_CODES[source_lang]
            audio_bytes = text_to_speech(st.session_state.original_text, source_code)
            if audio_bytes:
                create_audio_player(audio_bytes)
    
    with col2:
        st.subheader(f"🌐 Translated Text ({target_lang})")
        
        # Translation button
        translate_button_disabled = not api_available or not st.session_state.original_text
        if st.button("🔄 Translate", type="primary", use_container_width=True, disabled=translate_button_disabled):
            if st.session_state.original_text:
                with st.spinner("Translating with GPT-4o..."):
                    translated = openrouter_config.translate_text(st.session_state.original_text, target_lang)
                    if translated:
                        st.session_state.translated_text = translated
                        st.success("✅ Translation completed!")
            else:
                st.warning("⚠️ Please provide text to translate.")
        
        # Display translated text
        if st.session_state.translated_text:
            st.text_area(
                "Translated text:",
                value=st.session_state.translated_text,
                height=200,
                disabled=True
            )
            
            # Speak translated text
            if st.button("🔊 Speak Translation", use_container_width=True):
                target_code = LANGUAGE_CODES[target_lang]
                audio_bytes = text_to_speech(st.session_state.translated_text, target_code)
                if audio_bytes:
                    create_audio_player(audio_bytes)
        else:
            st.text_area(
                "Translated text:",
                value="Translation will appear here...",
                height=200,
                disabled=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>🔒 No patient data is stored. All processing is done via secure API.</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
