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
import numpy as np
from streamlit_audio_recorder import audio_recorder
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

def process_audio_bytes(audio_bytes):
    """
    Convert audio bytes to text using speech_recognition
    """
    if not audio_bytes:
        return None
        
    recognizer = sr.Recognizer()
    
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Load audio file with speech_recognition
        with sr.AudioFile(tmp_file_path) as source:
            # Adjust for ambient noise and record
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Recognize speech using Google Speech Recognition
        text = recognizer.recognize_google(audio_data)
        return text
        
    except sr.UnknownValueError:
        st.error("🤷 Could not understand the audio. Please speak clearly and try again.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition service error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred while processing audio: {str(e)}")
        return None
    finally:
        # Ensure temporary file is cleaned up
        try:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        except:
            pass

def text_to_speech(text, lang_code="en"):
    """
    Convert text to speech using gTTS and return audio bytes
    """
    if not text or not text.strip():
        st.error("❌ No text provided for speech synthesis.")
        return None
        
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

def create_audio_player(audio_bytes, autoplay=True):
    """
    Create an audio player widget for Streamlit
    """
    if audio_bytes:
        try:
            b64 = base64.b64encode(audio_bytes).decode()
            autoplay_attr = "autoplay" if autoplay else ""
            audio_html = f"""
            <audio controls {autoplay_attr} style="width: 100%;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error creating audio player: {str(e)}")

def validate_inputs(original_text, source_lang, target_lang):
    """
    Validate user inputs before processing
    """
    if not original_text or not original_text.strip():
        st.warning("⚠️ Please provide text to translate (either by recording or typing).")
        return False
        
    if source_lang == target_lang:
        st.warning("⚠️ Source and target languages are the same. Please select different languages.")
        return False
        
    return True

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
    if 'last_audio_key' not in st.session_state:
        st.session_state.last_audio_key = None
    
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
        
        # Browser-based audio recording
        st.markdown("**🎤 Record Audio:**")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=16000
        )
        
        # Process audio if new recording is available
        if audio_bytes and audio_bytes != st.session_state.last_audio_key:
            st.session_state.last_audio_key = audio_bytes
            
            with st.spinner("🔄 Processing your speech..."):
                recognized_text = process_audio_bytes(audio_bytes)
                
                if recognized_text:
                    st.session_state.original_text = recognized_text
                    st.success("✅ Speech recognized successfully!")
                    st.info(f"Recognized: \"{recognized_text}\"")
        
        # Text input area
        original_text = st.text_area(
            "Or type your text here:",
            value=st.session_state.original_text,
            height=150,
            key="original_input",
            help="You can either record audio above or type your text here"
        )
        
        # Update session state if text was manually edited
        if original_text != st.session_state.original_text:
            st.session_state.original_text = original_text
        
        # Speak original text
        if st.session_state.original_text and st.button("🔊 Speak Original", use_container_width=True):
            source_code = LANGUAGE_CODES[source_lang]
            with st.spinner("Generating speech..."):
                audio_bytes = text_to_speech(st.session_state.original_text, source_code)
                if audio_bytes:
                    create_audio_player(audio_bytes, autoplay=False)
    
    with col2:
        st.subheader(f"🌐 Translated Text ({target_lang})")
        
        # Translation button
        translate_button_disabled = not api_available or not st.session_state.original_text.strip()
        
        if st.button("🔄 Translate", type="primary", use_container_width=True, disabled=translate_button_disabled):
            if validate_inputs(st.session_state.original_text, source_lang, target_lang):
                with st.spinner("Translating with GPT-4o..."):
                    try:
                        translated = openrouter_config.translate_text(st.session_state.original_text, target_lang)
                        if translated and translated.strip():
                            st.session_state.translated_text = translated
                            st.success("✅ Translation completed!")
                        else:
                            st.error("❌ Translation failed. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Translation error: {str(e)}")
        
        # Display translated text
        if st.session_state.translated_text:
            st.text_area(
                "Translated text:",
                value=st.session_state.translated_text,
                height=150,
                disabled=True
            )
            
            # Speak translated text
            if st.button("🔊 Speak Translation", use_container_width=True):
                target_code = LANGUAGE_CODES[target_lang]
                with st.spinner("Generating translated speech..."):
                    audio_bytes = text_to_speech(st.session_state.translated_text, target_code)
                    if audio_bytes:
                        create_audio_player(audio_bytes, autoplay=False)
        else:
            st.text_area(
                "Translated text:",
                value="Translation will appear here after clicking 'Translate'...",
                height=150,
                disabled=True
            )
    
    # Clear/Reset functionality
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.original_text = ""
            st.session_state.translated_text = ""
            st.session_state.last_audio_key = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9em;'>🔒 No patient data is stored. All processing is done via secure API calls.</div>",
        unsafe_allow_html=True
    )
    
    # Usage instructions in sidebar
    with st.sidebar:
        st.header("📖 How to Use")
        st.markdown("""
        1. **Select Languages**: Choose source and target languages
        2. **Record Audio**: Click the microphone button to record
        3. **Or Type Text**: Enter text manually in the text area
        4. **Translate**: Click the translate button
        5. **Listen**: Use the speak buttons to hear audio playback
        
        **Tips:**
        - Speak clearly and at normal pace
        - Ensure good microphone access in your browser
        - Use headphones to avoid audio feedback
        """)
        
        st.header("🔧 Troubleshooting")
        st.markdown("""
        **Audio Issues:**
        - Allow microphone permissions in your browser
        - Check your microphone is working
        - Try refreshing the page if audio doesn't work
        
        **Translation Issues:**
        - Ensure you have internet connectivity
        - Check that API services are available
        - Try shorter text segments for better accuracy
        """)

if __name__ == "__main__":
    main()
