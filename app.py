import streamlit as st
import speech_recognition as sr
import requests
import json
import os
from gtts import gTTS
import tempfile
import base64
from io import BytesIO
import time
from llm_config import openrouter_config

# Check if PyAudio is available
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    st.warning("⚠️ Audio recording not available in this deployment environment. You can still use text input for translation.")

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

def speech_to_text():
    """
    Convert speech to text using speech_recognition library
    """
    if not PYAUDIO_AVAILABLE:
        st.error("❌ Audio recording is not available in this deployment environment. Please use text input instead.")
        return None
        
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Listen for audio with timeout
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
            
        st.info("🔄 Processing speech...")
        # Use Google Speech Recognition
        text = recognizer.recognize_google(audio)
        return text
    
    except sr.WaitTimeoutError:
        st.error("⏰ Listening timeout. Please try again.")
        return None
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
        
        # Voice input button - conditional based on PyAudio availability
        if PYAUDIO_AVAILABLE:
            if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                with st.spinner("Recording and processing..."):
                    text = speech_to_text()
                    if text:
                        st.session_state.original_text = text
                        st.success("✅ Speech captured successfully!")
        else:
            st.info("💡 Audio recording is not available in this deployment. Please use the text input below.")
        
        # Text input area
        original_text = st.text_area(
            "Type your text here:",
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
