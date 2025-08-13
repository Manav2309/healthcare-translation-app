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

# Custom CSS for better UI
st.markdown("""
<style>
.recording-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    background-color: #ff4444;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite alternate;
    margin-right: 8px;
}

@keyframes pulse {
    from { opacity: 1; }
    to { opacity: 0.3; }
}

.status-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid;
}

.status-recording {
    background-color: #fff5f5;
    border-color: #f56565;
    color: #c53030;
}

.status-processing {
    background-color: #fffaf0;
    border-color: #ed8936;
    color: #c05621;
}

.status-ready {
    background-color: #f0fff4;
    border-color: #48bb78;
    color: #2f855a;
}

.big-button {
    font-size: 1.2em !important;
    padding: 0.75rem 1.5rem !important;
    margin: 0.5rem 0 !important;
}

.audio-controls {
    display: flex;
    gap: 10px;
    align-items: center;
    justify-content: center;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

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

# Enhanced WebRTC configuration with STUN and TURN servers
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        # Add TURN server for stricter networks (optional)
        # {
        #     "urls": ["turn:your-turn-server.com:3478"],
        #     "username": "your-username",
        #     "credential": "your-password"
        # }
    ],
    "iceCandidatePoolSize": 10
})

class AudioProcessor:
    """
    Enhanced audio processor for handling WebRTC audio frames
    """
    def __init__(self):
        self.audio_frames = []
        self.is_recording = False
        self.sample_rate = 16000
        self.lock = threading.Lock()
        self.min_frames = 16  # Minimum frames before processing
        self.max_silence_frames = 32  # Max silent frames before auto-stop
        self.silence_threshold = 0.01
        self.silence_count = 0
    
    def start_recording(self):
        with self.lock:
            self.audio_frames = []
            self.is_recording = True
            self.silence_count = 0
            st.session_state.recording_status = "recording"
    
    def stop_recording(self):
        with self.lock:
            self.is_recording = False
            st.session_state.recording_status = "stopped"
    
    def add_frame(self, frame):
        with self.lock:
            if self.is_recording:
                self.audio_frames.append(frame)
                
                # Check for silence (auto-stop feature)
                if np.max(np.abs(frame)) < self.silence_threshold:
                    self.silence_count += 1
                else:
                    self.silence_count = 0
                
                # Auto-stop after prolonged silence
                if self.silence_count > self.max_silence_frames and len(self.audio_frames) > self.min_frames:
                    self.is_recording = False
                    st.session_state.recording_status = "auto_stopped"
    
    def get_audio_data(self):
        with self.lock:
            if not self.audio_frames or len(self.audio_frames) < self.min_frames:
                return None
            
            # Concatenate all audio frames
            audio_data = np.concatenate(self.audio_frames)
            
            # Apply noise reduction (simple high-pass filter)
            if len(audio_data) > 100:
                # Remove DC offset
                audio_data = audio_data - np.mean(audio_data)
                
                # Normalize audio
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.8
            
            # Convert to int16 format for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            return audio_int16
    
    def clear_frames(self):
        with self.lock:
            self.audio_frames = []
            self.silence_count = 0
    
    def get_recording_info(self):
        with self.lock:
            return {
                "frame_count": len(self.audio_frames),
                "duration": len(self.audio_frames) / (self.sample_rate / 1024) if self.audio_frames else 0,
                "is_recording": self.is_recording
            }

# Global audio processor instance
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()

if 'recording_status' not in st.session_state:
    st.session_state.recording_status = "ready"

def audio_frame_callback(frame):
    """
    Enhanced callback function to process audio frames from WebRTC
    """
    try:
        audio_array = frame.to_ndarray()
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Normalize audio
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32) / 32767.0
        
        # Add frame to processor
        st.session_state.audio_processor.add_frame(audio_array)
        
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
    
    return frame

def display_recording_status():
    """
    Display enhanced recording status with visual feedback
    """
    status = st.session_state.recording_status
    info = st.session_state.audio_processor.get_recording_info()
    
    if status == "recording":
        st.markdown(
            f'<div class="status-box status-recording">'
            f'<span class="recording-indicator"></span>'
            f'<strong>🎙️ Recording in progress...</strong><br>'
            f'Duration: {info["duration"]:.1f}s | Frames: {info["frame_count"]}'
            f'</div>',
            unsafe_allow_html=True
        )
    elif status == "processing":
        st.markdown(
            '<div class="status-box status-processing">'
            '<strong>🔄 Processing your speech...</strong>'
            '</div>',
            unsafe_allow_html=True
        )
    elif status == "auto_stopped":
        st.markdown(
            '<div class="status-box status-ready">'
            '<strong>⏸️ Recording auto-stopped (silence detected)</strong>'
            '</div>',
            unsafe_allow_html=True
        )
    elif status == "ready":
        st.markdown(
            '<div class="status-box status-ready">'
            '<strong>✅ Ready to record</strong>'
            '</div>',
            unsafe_allow_html=True
        )

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
    Save audio data to a temporary WAV file with better error handling
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
    Enhanced transcribe audio file using speech_recognition
    """
    if not wav_filename or not os.path.exists(wav_filename):
        return None
    
    recognizer = sr.Recognizer()
    
    # Adjust recognizer settings for better accuracy
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.operation_timeout = 10
    
    try:
        with sr.AudioFile(wav_filename) as source:
            # Adjust for ambient noise with longer duration
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            audio_data = recognizer.record(source)
        
        # Use Google Speech Recognition with language hint
        text = recognizer.recognize_google(
            audio_data,
            language=None,  # Auto-detect language
            show_all=False
        )
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
    Create an enhanced audio player widget for Streamlit
    """
    if audio_bytes:
        try:
            b64 = base64.b64encode(audio_bytes).decode()
            autoplay_attr = "autoplay" if autoplay else ""
            audio_html = f"""
            <div style="margin: 15px 0;">
                <audio controls {autoplay_attr} style="width: 100%; height: 40px;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            </div>
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
    Main Streamlit application with enhanced UI
    """
    # Check remote kill switch
    is_active, message = check_remote_config()
    
    if not is_active:
        st.error("🚫 Service temporarily disabled by administrator")
        st.info(message)
        st.stop()
    
    # App header with better styling
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
    if 'webrtc_ctx' not in st.session_state:
        st.session_state.webrtc_ctx = None
    
    # Language selection with better layout
    st.markdown("### 🌐 Language Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        source_lang = st.selectbox(
            "🗣️ Source Language (What you speak):",
            list(LANGUAGE_CODES.keys()),
            index=0,
            key="source_lang"
        )
    
    with col2:
        target_lang = st.selectbox(
            "🎯 Target Language (Translation to):",
            list(LANGUAGE_CODES.keys()),
            index=1,
            key="target_lang"
        )
    
    st.markdown("---")
    
    # Main interface with enhanced layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"### 📝 Original Text ({source_lang})")
        
        # Enhanced WebRTC Audio Recording Section
        st.markdown("#### 🎤 Live Microphone Input")
        
        # Display recording status
        display_recording_status()
        
        # Recording controls with better styling
        col_rec1, col_rec2, col_rec3 = st.columns([1, 1, 1])
        
        with col_rec1:
            start_disabled = st.session_state.recording_status == "recording"
            if st.button("🔴 Start Recording", disabled=start_disabled, key="start_rec", help="Click to start recording your voice"):
                st.session_state.audio_processor.start_recording()
                st.rerun()
        
        with col_rec2:
            stop_disabled = st.session_state.recording_status not in ["recording", "auto_stopped"]
            if st.button("⏹️ Stop & Process", disabled=stop_disabled, key="stop_rec", help="Stop recording and convert speech to text"):
                st.session_state.recording_status = "processing"
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
                                st.session_state.recording_status = "ready"
                                st.success("✅ Speech recognized successfully!")
                                st.info(f"**Recognized:** \"{recognized_text}\"")
                            else:
                                st.session_state.recording_status = "ready"
                                st.error("❌ Could not transcribe audio. Please try again.")
                        else:
                            st.session_state.recording_status = "ready"
                            st.error("❌ Could not process audio file.")
                    else:
                        st.session_state.recording_status = "ready"
                        st.warning("⚠️ No audio data captured. Please try recording again.")
                
                # Clear audio buffer
                st.session_state.audio_processor.clear_frames()
                st.rerun()
        
        with col_rec3:
            if st.button("🗑️ Clear Audio", key="clear_audio", help="Clear recorded audio and reset"):
                st.session_state.audio_processor.clear_frames()
                st.session_state.recording_status = "ready"
                st.rerun()
        
        # WebRTC Streamer (enhanced configuration)
        if st.session_state.recording_status == "recording":
            st.markdown("**🎙️ Speak now - your voice is being recorded...**")
            
            webrtc_ctx = webrtc_streamer(
                key="healthcare-audio-recorder",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=2048,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={
                    "video": False, 
                    "audio": {
                        "echoCancellation": True,
                        "noiseSuppression": True,
                        "autoGainControl": True,
                        "sampleRate": 16000
                    }
                },
                audio_frame_callback=audio_frame_callback,
                async_processing=True,
            )
            
            st.session_state.webrtc_ctx = webrtc_ctx
        
        # Text input area with better styling
        st.markdown("#### ✏️ Or Type Your Text")
        original_text = st.text_area(
            "Enter text here:",
            value=st.session_state.original_text,
            height=120,
            key="original_input",
            help="You can either record audio above or type your text here",
            placeholder="Type your message or use the microphone above..."
        )
        
        # Update session state if text was manually edited
        if original_text != st.session_state.original_text:
            st.session_state.original_text = original_text
        
        # Speak original text
        if st.session_state.original_text:
            if st.button("🔊 Speak Original Text", use_container_width=True, key="speak_orig"):
                source_code = LANGUAGE_CODES[source_lang]
                with st.spinner("Generating speech..."):
                    audio_bytes = text_to_speech(st.session_state.original_text, source_code)
                    if audio_bytes:
                        create_audio_player(audio_bytes, autoplay=False)
    
    with col2:
        st.markdown(f"### 🌐 Translated Text ({target_lang})")
        
        # Translation button with better styling
        translate_button_disabled = not api_available or not st.session_state.original_text.strip()
        
        if st.button(
            "🔄 Translate Now", 
            type="primary", 
            use_container_width=True, 
            disabled=translate_button_disabled,
            key="translate_btn",
            help="Translate the text using AI"
        ):
            if validate_inputs(st.session_state.original_text, source_lang, target_lang):
                with st.spinner("🤖 Translating with GPT-4o..."):
                    try:
                        translated = openrouter_config.translate_text(st.session_state.original_text, target_lang)
                        if translated and translated.strip():
                            st.session_state.translated_text = translated
                            st.success("✅ Translation completed successfully!")
                        else:
                            st.error("❌ Translation failed. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Translation error: {str(e)}")
        
        # Display translated text with better formatting
        if st.session_state.translated_text:
            st.text_area(
                "Translated text:",
                value=st.session_state.translated_text,
                height=120,
                disabled=True,
                key="translated_output"
            )
            
            # Speak translated text
            if st.button("🔊 Speak Translation", use_container_width=True, key="speak_trans"):
                target_code = LANGUAGE_CODES[target_lang]
                with st.spinner("Generating translated speech..."):
                    audio_bytes = text_to_speech(st.session_state.translated_text, target_code)
                    if audio_bytes:
                        create_audio_player(audio_bytes, autoplay=False)
        else:
            st.text_area(
                "Translated text:",
                value="Translation will appear here after clicking 'Translate Now'...",
                height=120,
                disabled=True,
                key="translated_placeholder"
            )
    
    # Clear/Reset functionality with better positioning
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🗑️ Clear All Content", use_container_width=True, key="clear_all"):
            st.session_state.original_text = ""
            st.session_state.translated_text = ""
            st.session_state.recording_status = "ready"
            st.session_state.audio_processor.clear_frames()
            st.success("✅ All content cleared!")
            time.sleep(1)
            st.rerun()
    
    # Footer with enhanced styling
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>🔒 <strong>Privacy Protected:</strong> No patient data is stored. All processing is done via secure API calls.</div>",
        unsafe_allow_html=True
    )
    
    # Enhanced usage instructions in sidebar
    with st.sidebar:
        st.header("📖 How to Use")
        st.markdown("""
        ### 🎤 Voice Recording:
        1. **Click "Start Recording"** - Allow microphone access
        2. **Speak clearly** into your microphone
        3. **Click "Stop & Process"** - Wait for transcription
        4. **Review the text** - Edit if needed
        
        ### ✏️ Text Input:
        - Type directly in the text area
        - Edit recognized speech if needed
        
        ### 🔄 Translation:
        1. **Select languages** (source and target)
        2. **Click "Translate Now"**
        3. **Listen to translation** using speak button
        
        ### 🎵 Audio Playback:
        - Use 🔊 buttons to hear original or translated text
        - Audio plays automatically in your browser
        """)
        
        st.header("🔧 Troubleshooting")
        st.markdown("""
        ### 🎙️ Microphone Issues:
        - **Allow permissions** when browser asks
        - **Check microphone** works in other apps
        - **Use HTTPS** (required for microphone access)
        - **Try different browser** if issues persist
        
        ### 🌐 Connection Issues:
        - **Stable internet** required for all features
        - **WebRTC connection** may take a few seconds
        - **Refresh page** if connection fails
        
        ### 🔄 Translation Issues:
        - **Check API status** indicator at top
        - **Try shorter sentences** for better accuracy
        - **Ensure different** source and target languages
        """)
        
        st.header("🌐 Browser Support")
        st.markdown("""
        ### ✅ Fully Supported:
        - **Chrome** (recommended)
        - **Firefox**
        - **Safari** (macOS/iOS)
        - **Edge**
        
        ### 📋 Requirements:
        - Modern browser with WebRTC
        - HTTPS connection (automatic on Streamlit Cloud)
        - Microphone permissions
        - Stable internet connection
        """)
        
        # Connection status indicator
        st.header("📡 Connection Status")
        if st.session_state.webrtc_ctx:
            if st.session_state.webrtc_ctx.state.playing:
                st.success("🟢 WebRTC Connected")
            else:
                st.warning("🟡 WebRTC Connecting...")
        else:
            st.info("🔵 WebRTC Ready")

if __name__ == "__main__":
    main()
