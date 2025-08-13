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
import wave
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
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

# WebRTC configuration for STUN servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class AudioProcessor:
    """
    Audio processor for handling WebRTC audio frames
    """
    def __init__(self):
        self.audio_frames = []
        self.is_recording = False
        self.sample_rate = 16000
        self.lock = threading.Lock()
    
    def start_recording(self):
        with self.lock:
            self.audio_frames = []
            self.is_recording = True
    
    def stop_recording(self):
        with self.lock:
            self.is_recording = False
    
    def add_frame(self, frame):
        with self.lock:
            if self.is_recording:
                self.audio_frames.append(frame)
    
    def get_audio_data(self):
        with self.lock:
            if not self.audio_frames:
                return None
            
            # Concatenate all audio frames
            audio_data = np.concatenate(self.audio_frames)
            
            # Convert to int16 format for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            return audio_int16
    
    def clear_frames(self):
        with self.lock:
            self.audio_frames = []

# Global audio processor instance
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()

def audio_frame_callback(frame):
    """
    Callback function to process audio frames from WebRTC
    """
    audio_array = frame.to_ndarray()
    
    # Convert to mono if stereo
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    # Normalize audio
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32) / 32767.0
    
    st.session_state.audio_processor.add_frame(audio_array)
    
    return frame

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
        return True, "Service running normally."

def save_audio_to_wav(audio_data, sample_rate=16000):
    """
    Save audio data to a temporary WAV file
    """
    if audio_data is None or len(audio_data) == 0:
        return None
    
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_filename = temp_file.name
        temp_file.close()
        
        # Write WAV file
        with wave.open(temp_filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_filename
    
    except Exception as e:
        st.error(f"❌ Error saving audio: {str(e)}")
        return None

def transcribe_audio(wav_filename):
    """
    Transcribe audio file using speech_recognition
    """
    if not wav_filename or not os.path.exists(wav_filename):
        return None
    
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(wav_filename) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
        
        # Use Google Speech Recognition
        text = recognizer.recognize_google(audio_data)
        return text
    
    except sr.UnknownValueError:
        st.error("🤷 Could not understand the audio. Please speak clearly and try again.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition service error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Transcription error: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        try:
            if wav_filename and os.path.exists(wav_filename):
                os.unlink(wav_filename)
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
            <audio controls {autoplay_attr} style="width: 100%; margin: 10px 0;">
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
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    
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
        
        # WebRTC Audio Recording Section
        st.markdown("**🎤 Live Microphone Input:**")
        
        # Recording controls
        col_rec1, col_rec2, col_rec3 = st.columns([1, 1, 1])
        
        with col_rec1:
            if st.button("🔴 Start Recording", disabled=st.session_state.is_recording):
                st.session_state.is_recording = True
                st.session_state.audio_processor.start_recording()
                st.rerun()
        
        with col_rec2:
            if st.button("⏹️ Stop & Process", disabled=not st.session_state.is_recording):
                st.session_state.is_recording = False
                st.session_state.audio_processor.stop_recording()
                
                # Process recorded audio
                with st.spinner("🔄 Processing your speech..."):
                    audio_data = st.session_state.audio_processor.get_audio_data()
                    
                    if audio_data is not None and len(audio_data) > 0:
                        # Save to WAV file
                        wav_filename = save_audio_to_wav(audio_data)
                        
                        if wav_filename:
                            # Transcribe audio
                            recognized_text = transcribe_audio(wav_filename)
                            
                            if recognized_text:
                                st.session_state.original_text = recognized_text
                                st.success("✅ Speech recognized successfully!")
                                st.info(f"Recognized: \"{recognized_text}\"")
                            else:
                                st.error("❌ Could not transcribe audio. Please try again.")
                    else:
                        st.warning("⚠️ No audio data captured. Please try recording again.")
                
                # Clear audio buffer
                st.session_state.audio_processor.clear_frames()
                st.rerun()
        
        with col_rec3:
            if st.button("🗑️ Clear Audio"):
                st.session_state.audio_processor.clear_frames()
                st.session_state.is_recording = False
                st.rerun()
        
        # WebRTC Streamer (hidden but active for audio capture)
        if st.session_state.is_recording:
            st.info("🎙️ Recording in progress... Speak now!")
            
            webrtc_ctx = webrtc_streamer(
                key="audio-recorder",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=1024,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": False, "audio": True},
                audio_frame_callback=audio_frame_callback,
            )
        
        # Text input area
        st.markdown("**✏️ Or type your text:**")
        original_text = st.text_area(
            "Enter text here:",
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
            st.session_state.is_recording = False
            st.session_state.audio_processor.clear_frames()
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
        2. **Record Audio**: 
           - Click "Start Recording" to begin
           - Speak clearly into your microphone
           - Click "Stop & Process" to transcribe
        3. **Or Type Text**: Enter text manually in the text area
        4. **Translate**: Click the translate button
        5. **Listen**: Use the speak buttons to hear audio playback
        
        **Tips:**
        - Allow microphone permissions when prompted
        - Speak clearly and at normal pace
        - Use headphones to avoid audio feedback
        - Ensure stable internet connection
        """)
        
        st.header("🔧 Troubleshooting")
        st.markdown("""
        **Audio Issues:**
        - Allow microphone permissions in your browser
        - Check your microphone is working in other apps
        - Try refreshing the page if audio doesn't work
        - Ensure you're using HTTPS (required for microphone access)
        
        **Translation Issues:**
        - Ensure you have internet connectivity
        - Check that API services are available
        - Try shorter text segments for better accuracy
        """)
        
        st.header("🌐 Browser Compatibility")
        st.markdown("""
        **Supported Browsers:**
        - ✅ Chrome (recommended)
        - ✅ Firefox
        - ✅ Safari (macOS/iOS)
        - ✅ Edge
        
        **Requirements:**
        - HTTPS connection (automatic on Streamlit Cloud)
        - Microphone permissions
        - Modern browser with WebRTC support
        """)

if __name__ == "__main__":
    main()
