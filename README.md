# 🏥 Healthcare Translation Web App

A lightweight Streamlit prototype for real-time multilingual translation between patients and healthcare providers, powered by OpenRouter's GPT-4o API.

![Healthcare Translation App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

## ✨ Features

- 🎤 **Voice-to-Text**: Real-time speech recognition using Google Speech API
- 🌐 **AI Translation**: High-quality translation using GPT-4o via OpenRouter
- 🔊 **Text-to-Speech**: Audio playback in multiple languages
- 📱 **Mobile-Friendly**: Responsive design for mobile devices
- 🔒 **Privacy-First**: No PHI storage, secure API processing
- 🛡️ **Remote Kill Switch**: Safe maintenance control via remote config
- 🏥 **Medical Focus**: Optimized for healthcare terminology

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenRouter API key
- Microphone access for voice input

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/healthcare-translation-app.git
cd healthcare-translation-app
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your OpenRouter API key
```

5. **Run the application**
```bash
streamlit run app.py
```

## 🔧 Configuration

### OpenRouter API Setup

1. Get your API key from [OpenRouter](https://openrouter.ai/)
2. Update the API key in `llm_config.py` or use environment variables
3. Optionally customize the model, site URL, and other settings

### Supported Languages

- English, Spanish, French, German, Italian
- Portuguese, Chinese, Japanese, Korean
- Arabic, Hindi, Russian

## 📁 Project Structure