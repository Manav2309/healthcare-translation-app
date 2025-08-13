from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenRouterConfig:
    """
    Configuration and utilities for OpenRouter API integration
    """
    
    def __init__(self):
        # Get all configuration from environment variables
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        self.site_url = os.getenv("SITE_URL", "https://healthcare-translator.streamlit.app")
        self.site_name = os.getenv("SITE_NAME", "Healthcare Translation App")
        
        # Translation settings from environment
        self.temperature = float(os.getenv("TRANSLATION_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("TRANSLATION_MAX_TOKENS", "1000"))
        self.test_max_tokens = int(os.getenv("TEST_MAX_TOKENS", "10"))
        
        # Validate required environment variables
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        # Initialize OpenAI client with OpenRouter
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
    def get_translation_prompt(self, text, target_lang):
        """
        Generate a clean translation prompt that ensures only translated text is returned
        """
        prompt_template = os.getenv("TRANSLATION_PROMPT_TEMPLATE", 
            """You are a professional medical translator. Translate the following medical text from the source language into {target_lang}.

IMPORTANT INSTRUCTIONS:
- Keep all medical terminology accurate and precise
- Maintain the same tone and formality level
- Return ONLY the translated text, no explanations or additional notes
- Do not add quotation marks around the translation
- Preserve any formatting or structure from the original text

Text to translate:
{text}""")
        
        return prompt_template.format(target_lang=target_lang, text=text)
    
    def clean_translation_response(self, response):
        """
        Clean up the API response to remove any extra formatting
        """
        if not response:
            return response
            
        translation = response.strip()
        
        # Remove quotes if the entire response is wrapped in them
        if (translation.startswith('"') and translation.endswith('"')) or \
           (translation.startswith("'") and translation.endswith("'")):
            translation = translation[1:-1]
        
        # Remove any remaining leading/trailing whitespace
        translation = translation.strip()
        
        return translation
    
    def translate_text(self, text, target_lang="Spanish"):
        """
        Translate text using OpenRouter API with GPT-4o
        """
        try:
            prompt = self.get_translation_prompt(text, target_lang)
            
            system_message = os.getenv("SYSTEM_MESSAGE", 
                "You are a professional medical translator. Always return only the translated text without any additional explanations, formatting, or quotation marks.")
            
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if completion.choices and len(completion.choices) > 0:
                translated_text = completion.choices[0].message.content
                cleaned_response = self.clean_translation_response(translated_text)
                return cleaned_response
            else:
                st.error("❌ No translation received from API")
                return None
                
        except Exception as e:
            st.error(f"❌ Translation error: {str(e)}")
            return None
    
    def check_api_availability(self):
        """
        Check if OpenRouter API is available and working
        """
        try:
            test_message = os.getenv("API_TEST_MESSAGE", "Translate 'Hello' to Spanish. Return only the translation.")
            
            # Test with a simple translation
            test_completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": test_message
                    }
                ],
                max_tokens=self.test_max_tokens
            )
            
            if test_completion.choices and len(test_completion.choices) > 0:
                return True, "OpenRouter API is available and working"
            else:
                return False, "OpenRouter API returned empty response"
                
        except Exception as e:
            return False, f"OpenRouter API error: {str(e)}"

# Global instance
openrouter_config = OpenRouterConfig()