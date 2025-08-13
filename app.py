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

# Enhanced CSS for better UI
st.markdown("""
<style>
.recording-indicator {
    display: inline-block;
    width: 15px;
    height: 15px;
    background-color: #ff4444;
    border-radius: 50%;
    animation: pulse 1s ease-in-out infinite alternate;
    margin-right: 8px;
}

@keyframes pulse {
    from { opacity: 1; transform: scale(1); }
    to { opacity: 0.4; transform: scale(1.1); }
}

.status-box {
    padding: 1.2rem;
    border-radius: 0.8rem;
    margin: 1rem 0;
    border-left: 5px solid;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

.status-connected {
    background-color: #ebf8ff;
    border-color: #3182ce;
    color: #2c5282;
}

.big-button {
    font-size: 1.2em !important;
    padding: 0.75rem 1.5rem !important;
    margin: 0.5rem 0 !important;
    border-radius: 0.5rem !important;
}

.audio-controls {
    display: flex;
    gap: 15px;
    align-items: center;
    justify-content: center;
    margin: 20px 0;
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 10px;
}

.connection-status {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
    padding: 8px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: bold;
}

.connected { background-color: #d4edda; color: #155724; }
.connecting { background-color: #fff3cd; color: #856404; }
.disconnected { background-color: #f8d7da; color: #721c24; }

.real-time-transcription {
    background-color: #f8f9fa;
    border: 2px dashed #6c757d;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    min-height: 60px;
    font-style: italic;
    color: #6c757d;
}

.audio-level-indicator {
    width: 100%;
    height: 20px;
    background-color: #e9ecef;
    border-radius: 10px;
    overflow: hidden;
    margin: 10px 0;
}

.audio-level-bar {
    height: 100%;
    background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
    transition: width 0.1s ease;
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

# Enhanced WebRTC configuration with multiple STUN servers and TURN support
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        # Free TURN servers (you can add your own for better reliability)
        {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"},
        {"urls": ["turn:openrelay.metered.ca:443"], "username": "openrelayproject", "credential": "openrelayproject"},
    ],
    "iceCandidatePoolSize": 20,
    "iceTransportPolicy": "all",
    "bundlePolicy": "balanced"
})

class EnhancedAudioProcessor:
    """
    Enhanced audio processor with better frame handling and real-time feedback
    """
    def __init__(self):
        self.audio_frames = []
        self.is_recording = False
        self.sample_rate = 16000
        self.lock = threading.Lock()
        self.min_frames = 8  # Reduced for faster response
        self.max_silence_frames = 48  # Increased for better detection
        self.silence_threshold = 0.005  # More sensitive
        self.silence_count = 0
        self.audio_level = 0.0
        self.frame_count = 0
        self.last_audio_time = time.time()
        
    def start_recording(self):
        with self.lock:
            self.audio_frames = []
            self.is_recording = True
            self.silence_count = 0
            self.frame_count = 0
            self.last_audio_time = time.time()
            st.session_state.recording_status = "recording"
            st.session_state.real_time_text = "Listening..."
    
    def stop_recording(self):
        with self.lock:
            self.is_recording = False
            st.session_state.recording_status = "stopped"
    
    def add_frame(self, frame):
        with self.lock:
            if self.is_recording and frame is not None:
                # Ensure frame is numpy array
                if hasattr(frame, 'to_ndarray'):
                    audio_array = frame.to_ndarray()
                else:
                    audio_array = np.array(frame)
                
                # Convert to mono if stereo
                if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                    audio_array = np.mean(audio_array, axis=1)
                elif len(audio_array.shape) > 1:
                    audio_array = audio_array.flatten()
                
                # Ensure float32 format
                if audio_array.dtype != np.float32:
                    if audio_array.dtype == np.int16:
                        audio_array = audio_array.astype(np.float32) / 32767.0
                    else:
                        audio_array = audio_array.astype(np.float32)
                
                # Calculate audio level for real-time feedback
                self.audio_level = float(np.sqrt(np.mean(audio_array ** 2)))
                
                # Only add frames with sufficient audio
                if len(audio_array) > 0:
                    self.audio_frames.append(audio_array)
                    self.frame_count += 1
                    
                    # Check for silence with improved detection
                    if self.audio_level < self.silence_threshold:
                        self.silence_count += 1
                    else:
                        self.silence_count = 0
                        self.last_audio_time = time.time()
                    
                    # Auto-stop after prolonged silence (but only if we have enough audio)
                    if (self.silence_count > self.max_silence_frames and 
                        len(self.audio_frames) > self.min_frames and
                        time.time() - self.last_audio_time > 2.0):  # 2 seconds of silence
                        self.is_recording = False
                        st.session_state.recording_status = "auto_stopped"
                        st.session_state.real_time_text = "Recording stopped (silence detected)"
    
    def get_audio_data(self):
        with self.lock:
            if not self.audio_frames or len(self.audio_frames) < self.min_frames:
                return None
            
            try:
                # Concatenate all audio frames
                audio_data = np.concatenate(self.audio_frames)
                
                # Apply enhanced noise reduction
                if len(audio_data) > 100:
                    # Remove DC offset
                    audio_data = audio_data - np.mean(audio_data)
                    
                    # Apply simple high-pass filter to remove low-frequency noise
                    if len(audio_data) > 1000:
                        # Simple high-pass filter
                        b = np.array([0.96, -0.96])
                        a = np.array([1.0, -0.95])
                        from scipy import signal
                        try:
                            audio_data = signal.lfilter(b, a, audio_data)
                        except:
                            pass  # If scipy not available, skip filtering
                    
                    # Normalize audio with better dynamic range
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        # Apply soft limiting
                        audio_data = audio_data / max_val
                        audio_data = np.tanh(audio_data * 2) * 0.9
                
                # Convert to int16 format for WAV
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                return audio_int16
            except Exception as e:
                st.error(f"Audio processing error: {str(e)}")
                return None
    
    def clear_frames(self):
        with self.lock:
            self.audio_frames = []
            self.silence_count = 0
            self.frame_count = 0
            self.audio_level = 0.0
    
    def get_recording_info(self):
        with self.lock:
            duration = len(self.audio_frames) * 1024 / self.sample_rate if self.audio_frames else 0
            return {
                "frame_count": len(self.audio_frames),
                "duration": duration,
                "is_recording": self.is_recording,
                "audio_level": self.audio_level,
                "silence_count": self.silence_count
            }

# Global audio processor instance
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = EnhancedAudioProcessor()

if 'recording_status' not in st.session_state:
    st.session_state.recording_status = "ready"

if 'webrtc_ctx' not in st.session_state:
    st.session_state.webrtc_ctx = None

if 'connection_status' not in st.session_state:
    st.session_state.connection_status = "disconnected"

if 'real_time_text' not in st.session_state:
    st.session_state.real_time_text = "Ready to record..."

def enhanced_audio_frame_callback(frame):
    """
    Enhanced callback function with better error handling and frame processing
    """
    try:
        if frame is not None:
            st.session_state.audio_processor.add_frame(frame)
            
            # Update connection status
            st.session_state.connection_status = "connected"
            
            # Update real-time feedback
            info = st.session_state.audio_processor.get_recording_info()
            if info["is_recording"] and info["audio_level"] > 0.01:
                st.session_state.real_time_text = f"Recording... (Level: {info['audio_level']:.2f}, Duration: {info['duration']:.1f}s)"
        
    except Exception as e:
        st.error(f"Audio frame processing error: {str(e)}")
        st.session_state.connection_status = "error"
    
    return frame

def display_enhanced_recording_status():
    """
    Display enhanced recording status with real-time feedback
    """
    status = st.session_state.recording_status
    info = st.session_state.audio_processor.get_recording_info()
    connection = st.session_state.connection_status
    
    # Connection status indicator
    if connection == "connected":
        st.markdown(
            '<div class="connection-status connected">🟢 WebRTC Connected</div>',
            unsafe_allow_html=True
        )
    elif connection == "connecting":
        st.markdown(
            '<div class="connection-status connecting">🟡 Connecting...</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="connection-status disconnected">🔴 Disconnected</div>',
            unsafe_allow_html=True
        )
    
    # Main status display
    if status == "recording":
        st.markdown(
            f'<div class="status-box status-recording">'
            f'<span class="recording-indicator"></span>'
            f'<strong>🎙️ Recording in progress...</strong><br>'
            f'Duration: {info["duration"]:.1f}s | Frames: {info["frame_count"]} | Level: {info["audio_level"]:.3f}'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Audio level indicator
        level_percentage = min(info["audio_level"] * 100, 100)
        st.markdown(
            f'<div class="audio-level-indicator">'
            f'<div class="audio-level-bar" style="width: {level_percentage}%"></div>'
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
    
    # Real-time transcription area
    st.markdown(
        f'<div class="real-time-transcription">'
        f'<strong>Real-time Status:</strong> {st.session_state.real_time_text}'
        f'</div>',
        unsafe_allow_html=True
    )

# ... existing code ...

def check_remote_config():
    """
    Check remote configuration for kill switch
    Returns True if app should continue, False if disabled
    """
    try:
        # Use local config.json as fallback
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config.get("active", True), config.get("message", "Service running normally.")
    except Exception as e:
        # If config fetch fails, allow app to run (fail-safe)
        return True, "Service running normally."

def save_audio_to_wav(audio_data, sample_rate=16000):
    """
    Save audio data to a temporary WAV file with enhanced error handling
    """
    if audio_data is None or len(audio_data) == 0:
        return None
    
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_filename = temp_file.name
        temp_file.close()
        
        # Write WAV file with better parameters
        with wave.open(temp_filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_filename
    except Exception as e:
        st.error(f"Error saving audio file: {str(e)}")
        return None

def transcribe_audio(audio_file_path):
    """
    Transcribe audio using Google Speech Recognition with enhanced error handling
    """
    if not audio_file_path or not os.path.exists(audio_file_path):
        return None
    
    try:
        recognizer = sr.Recognizer()
        
        # Adjust recognizer settings for better accuracy
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        recognizer.operation_timeout = 10
        
        with sr.AudioFile(audio_file_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = recognizer.record(source)
        
        # Use Google Speech Recognition with language detection
        text = recognizer.recognize_google(audio, language="en-US", show_all=False)
        
        # Clean up temporary file
        try:
            os.unlink(audio_file_path)
        except:
            pass
        
        return text.strip() if text else None
        
    except sr.UnknownValueError:
        st.warning("⚠️ Could not understand the audio. Please speak more clearly.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition service error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Transcription error: {str(e)}")
        return None
    finally:
        # Ensure cleanup
        try:
            if audio_file_path and os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
        except:
            pass

def text_to_speech(text, language_code="en"):
    """
    Convert text to speech using gTTS with enhanced error handling
    """
    try:
        if not text or not text.strip():
            return None
        
        tts = gTTS(text=text, lang=language_code, slow=False)
        
        # Save to BytesIO buffer
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"❌ Text-to-speech error: {str(e)}")
        return None

def create_audio_player(audio_bytes, autoplay=False):
    """
    Create an HTML audio player for the generated speech
    """
    try:
        if audio_bytes:
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <audio controls {'autoplay' if autoplay else ''}>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        else:
            st.error("❌ Could not generate audio")
    except Exception as e:
        st.error(f"❌ Audio player error: {str(e)}")

def validate_inputs(text, source_lang, target_lang):
    """
    Validate user inputs with enhanced checks
    """
    if not text or not text.strip():
        st.error("❌ Please provide text to translate (either by recording or typing)")
        return False
    
    if source_lang == target_lang:
        st.warning("⚠️ Source and target languages are the same. Please select different languages.")
        return False
    
    if len(text.strip()) > 5000:
        st.error("❌ Text is too long. Please keep it under 5000 characters.")
        return False
    
    return True

def main():
    """
    Main Streamlit application with enhanced UI and functionality
    """
    # Check remote kill switch
    is_active, message = check_remote_config()
    
    if not is_active:
        st.error("🚫 Service temporarily disabled by administrator")
        st.info(message)
        st.stop()
    
    # App header with enhanced styling
    st.title("🏥 Healthcare Translation App")
    st.markdown("*Real-time multilingual communication for healthcare providers and patients*")
    st.markdown("*Powered by OpenRouter & GPT-4o with Enhanced WebRTC Audio*")
    
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
    
    # Language selection with enhanced layout
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
    
    # Initialize WebRTC connection early for better performance
    st.markdown("### 🎙️ Audio Connection Setup")
    
    # Always show WebRTC streamer for persistent connection
    webrtc_ctx = webrtc_streamer(
        key="healthcare-audio-persistent",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": False, 
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
                "sampleRate": 16000,
                "channelCount": 1
            }
        },
        audio_frame_callback=enhanced_audio_frame_callback,
        async_processing=True,
    )
    
    st.session_state.webrtc_ctx = webrtc_ctx
    
    # Update connection status based on WebRTC state
    if webrtc_ctx.state.playing:
        st.session_state.connection_status = "connected"
    elif webrtc_ctx.state.signalling:
        st.session_state.connection_status = "connecting"
    else:
        st.session_state.connection_status = "disconnected"
    
    # Main interface with enhanced layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"### 📝 Original Text ({source_lang})")
        
        # Enhanced WebRTC Audio Recording Section
        st.markdown("#### 🎤 Live Microphone Input")
        
        # Display enhanced recording status
        display_enhanced_recording_status()
        
        # Enhanced recording controls
        st.markdown('<div class="audio-controls">', unsafe_allow_html=True)
        col_rec1, col_rec2, col_rec3 = st.columns([1, 1, 1])
        
        with col_rec1:
            start_disabled = (st.session_state.recording_status == "recording" or 
                            st.session_state.connection_status != "connected")
            if st.button(
                "🔴 Start Recording", 
                disabled=start_disabled, 
                key="start_rec", 
                help="Click to start recording your voice",
                use_container_width=True
            ):
                if st.session_state.connection_status == "connected":
                    st.session_state.audio_processor.start_recording()
                    st.rerun()
                else:
                    st.error("❌ Please wait for WebRTC connection to establish")
        
        with col_rec2:
            stop_disabled = st.session_state.recording_status not in ["recording", "auto_stopped"]
            if st.button(
                "⏹️ Stop & Process", 
                disabled=stop_disabled, 
                key="stop_rec", 
                help="Stop recording and convert speech to text",
                use_container_width=True
            ):
                st.session_state.recording_status = "processing"
                st.session_state.audio_processor.stop_recording()
                
                with st.spinner("🔄 Processing your speech..."):
                    # Get audio data
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
                                st.session_state.real_time_text = "Speech recognized successfully!"
                                st.success("✅ Speech recognized successfully!")
                                st.info(f"**Recognized:** \"{recognized_text}\"")
                            else:
                                st.session_state.recording_status = "ready"
                                st.session_state.real_time_text = "Could not transcribe audio"
                                st.error("❌ Could not transcribe audio. Please try again.")
                        else:
                            st.session_state.recording_status = "ready"
                            st.session_state.real_time_text = "Could not process audio file"
                            st.error("❌ Could not process audio file.")
                    else:
                        st.session_state.recording_status = "ready"
                        st.session_state.real_time_text = "No audio data captured"
                        st.warning("⚠️ No audio data captured. Please try recording again.")
                
                # Clear audio buffer
                st.session_state.audio_processor.clear_frames()
                st.rerun()
        
        with col_rec3:
            if st.button(
                "🗑️ Clear Audio", 
                key="clear_audio", 
                help="Clear recorded audio and reset",
                use_container_width=True
            ):
                st.session_state.audio_processor.clear_frames()
                st.session_state.recording_status = "ready"
                st.session_state.real_time_text = "Ready to record..."
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Text input area with enhanced styling
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
        
        # Translation button with enhanced styling
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
        
        # Display translated text with enhanced formatting
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
    
    # Enhanced sidebar with troubleshooting and connection info
    with st.sidebar:
        st.header("🔧 System Status")
        
        # WebRTC Connection Status
        st.subheader("📡 Connection Status")
        if st.session_state.webrtc_ctx:
            if st.session_state.webrtc_ctx.state.playing:
                st.success("🟢 WebRTC Connected & Active")
            elif st.session_state.webrtc_ctx.state.signalling:
                st.warning("🟡 WebRTC Connecting...")
            else:
                st.error("🔴 WebRTC Disconnected")
                st.info("💡 Try refreshing the page if connection fails")
        else:
            st.info("🔵 WebRTC Initializing...")
        
        # Audio Status
        st.subheader("🎤 Audio Status")
        info = st.session_state.audio_processor.get_recording_info()
        st.write(f"**Recording:** {'Yes' if info['is_recording'] else 'No'}")
        st.write(f"**Audio Level:** {info['audio_level']:.3f}")
        st.write(f"**Frames Captured:** {info['frame_count']}")
        st.write(f"**Duration:** {info['duration']:.1f}s")
        
        st.header("🔧 Troubleshooting")
        st.markdown("""
        ### 🎙️ Microphone Issues:
        - **Allow permissions** when browser asks
        - **Check microphone** works in other apps
        - **Use HTTPS** (required for microphone access)
        - **Try different browser** if issues persist
        - **Refresh page** if WebRTC fails to connect
        
        ### 🌐 Connection Issues:
        - **Stable internet** required for all features
        - **WebRTC connection** may take 10-15 seconds
        - **Check firewall** settings if connection fails
        - **Try different network** if behind strict firewall
        
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

if __name__ == "__main__":
    main()
