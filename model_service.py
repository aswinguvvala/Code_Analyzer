"""
Model Service Abstraction Layer
Easily switch between different AI providers for production deployment
"""

import os
import asyncio
import json
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService(ABC):
    """Abstract base class for model services"""
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the service is available"""
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Get the name of the service"""
        pass

class GeminiService(ModelService):
    """Google Gemini API service"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.available = False
        self.requests_made = 0
        self.rate_limited = False
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.available = True
            logger.info("✅ Gemini service initialized")
        except Exception as e:
            logger.error(f"❌ Gemini initialization failed: {e}")
            self.available = False
    
    async def generate_response(self, messages: List[Dict], **kwargs) -> str:
        if not self.available or self.rate_limited:
            raise Exception("Gemini service not available")
        
        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages)
        
        max_retries = kwargs.get('max_retries', 3)
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    self.requests_made += 1
                    return response.text.strip()
                else:
                    raise Exception("Empty response from Gemini")
            except Exception as e:
                error_msg = str(e).lower()
                if 'quota' in error_msg or 'rate limit' in error_msg:
                    self.rate_limited = True
                    raise Exception("Rate limit exceeded")
                elif attempt == max_retries - 1:
                    raise Exception(f"Gemini API failed after {max_retries} attempts")
                else:
                    await asyncio.sleep(2 ** attempt)
        
        raise Exception("Gemini API failed")
    
    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
        return prompt
    
    def is_available(self) -> bool:
        return self.available and not self.rate_limited
    
    def get_service_name(self) -> str:
        return "Google Gemini"

class OllamaService(ModelService):
    """Local Ollama service"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.available = self._check_ollama_availability()
        
        if self.available:
            logger.info(f"✅ Ollama service initialized with {model_name}")
        else:
            logger.warning("❌ Ollama service not available")
    
    def _check_ollama_availability(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    async def generate_response(self, messages: List[Dict], **kwargs) -> str:
        if not self.available:
            raise Exception("Ollama service not available")
        
        import subprocess
        
        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages)
        
        try:
            cmd = ['ollama', 'run', self.model_name]
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            stdout, stderr = process.communicate(input=prompt, timeout=120)
            
            if process.returncode == 0:
                return stdout.strip()
            else:
                raise Exception(f"Ollama error: {stderr}")
                
        except subprocess.TimeoutExpired:
            process.kill()
            raise Exception("Ollama request timed out")
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")
    
    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
        prompt += "Assistant:"
        return prompt
    
    def is_available(self) -> bool:
        return self.available
    
    def get_service_name(self) -> str:
        return f"Ollama ({self.model_name})"

class OpenAIService(ModelService):
    """OpenAI API service (for production)"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        self.available = False
        self.client = None
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
            self.available = True
            logger.info(f"✅ OpenAI service initialized with {model_name}")
        except Exception as e:
            logger.error(f"❌ OpenAI initialization failed: {e}")
            self.available = False
    
    async def generate_response(self, messages: List[Dict], **kwargs) -> str:
        if not self.available:
            raise Exception("OpenAI service not available")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 1500),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def is_available(self) -> bool:
        return self.available
    
    def get_service_name(self) -> str:
        return f"OpenAI ({self.model_name})"

class AWSBedrockService(ModelService):
    """AWS Bedrock service (for production)"""
    
    def __init__(self, region: str = "us-east-1", model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        self.region = region
        self.model_id = model_id
        self.available = False
        self.client = None
        
        try:
            import boto3
            self.client = boto3.client('bedrock-runtime', region_name=region)
            self.available = True
            logger.info(f"✅ AWS Bedrock service initialized with {model_id}")
        except Exception as e:
            logger.error(f"❌ AWS Bedrock initialization failed: {e}")
            self.available = False
    
    async def generate_response(self, messages: List[Dict], **kwargs) -> str:
        if not self.available:
            raise Exception("AWS Bedrock service not available")
        
        try:
            # Convert messages to Claude format
            prompt = self._convert_messages_to_claude_format(messages)
            
            body = json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": kwargs.get('max_tokens', 1500),
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": 1,
            })
            
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('completion', '').strip()
        except Exception as e:
            raise Exception(f"AWS Bedrock error: {str(e)}")
    
    def _convert_messages_to_claude_format(self, messages: List[Dict]) -> str:
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"Human: {msg['content']}\n\n"
        prompt += "Assistant:"
        return prompt
    
    def is_available(self) -> bool:
        return self.available
    
    def get_service_name(self) -> str:
        return f"AWS Bedrock ({self.model_id})"

class ModelServiceManager:
    """Manages multiple model services with intelligent fallback"""
    
    def __init__(self):
        self.services: List[ModelService] = []
        self.primary_service: Optional[ModelService] = None
        self.fallback_services: List[ModelService] = []
        self.usage_stats = {"requests": 0, "failures": 0, "fallbacks": 0}
        
        # Initialize services based on environment
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize available services based on environment variables"""
        
        # Gemini API
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            service = GeminiService(gemini_key)
            if service.is_available():
                self.services.append(service)
                if not self.primary_service:
                    self.primary_service = service
        
        # OpenAI API
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            service = OpenAIService(openai_key)
            if service.is_available():
                self.services.append(service)
                if not self.primary_service:
                    self.primary_service = service
        
        # AWS Bedrock
        if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
            service = AWSBedrockService()
            if service.is_available():
                self.services.append(service)
                if not self.primary_service:
                    self.primary_service = service
        
        # Ollama (local fallback)
        ollama_service = OllamaService()
        if ollama_service.is_available():
            self.services.append(ollama_service)
            if not self.primary_service:
                self.primary_service = ollama_service
        
        # Set fallback services
        self.fallback_services = [s for s in self.services if s != self.primary_service]
        
        logger.info(f"Initialized {len(self.services)} model services")
        if self.primary_service:
            logger.info(f"Primary service: {self.primary_service.get_service_name()}")
    
    async def generate_response(self, messages: List[Dict], context: str = "analysis") -> str:
        """Generate response with intelligent fallback"""
        
        if not self.services:
            raise Exception("No model services available")
        
        # Try primary service first
        if self.primary_service and self.primary_service.is_available():
            try:
                self.usage_stats["requests"] += 1
                response = await self.primary_service.generate_response(messages)
                logger.info(f"✅ Response generated using {self.primary_service.get_service_name()}")
                return response
            except Exception as e:
                logger.warning(f"⚠️ Primary service failed: {e}")
                self.usage_stats["failures"] += 1
        
        # Try fallback services
        for service in self.fallback_services:
            if service.is_available():
                try:
                    self.usage_stats["requests"] += 1
                    self.usage_stats["fallbacks"] += 1
                    response = await service.generate_response(messages)
                    logger.info(f"✅ Response generated using fallback: {service.get_service_name()}")
                    return response
                except Exception as e:
                    logger.warning(f"⚠️ Fallback service failed: {e}")
                    self.usage_stats["failures"] += 1
        
        raise Exception(f"All model services failed for {context}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all services"""
        return {
            "total_services": len(self.services),
            "available_services": [s.get_service_name() for s in self.services if s.is_available()],
            "primary_service": self.primary_service.get_service_name() if self.primary_service else None,
            "fallback_services": [s.get_service_name() for s in self.fallback_services if s.is_available()],
            "usage_stats": self.usage_stats,
            "has_available_service": any(s.is_available() for s in self.services)
        }
    
    def add_service(self, service: ModelService):
        """Add a custom service"""
        if service.is_available():
            self.services.append(service)
            if not self.primary_service:
                self.primary_service = service
            else:
                self.fallback_services.append(service)
            logger.info(f"Added service: {service.get_service_name()}")
    
    def set_primary_service(self, service_name: str):
        """Set primary service by name"""
        for service in self.services:
            if service.get_service_name() == service_name:
                self.primary_service = service
                self.fallback_services = [s for s in self.services if s != service]
                logger.info(f"Set primary service to: {service_name}")
                return
        logger.warning(f"Service not found: {service_name}")

# Global instance
model_service_manager = ModelServiceManager()

# Convenience functions
async def generate_response(messages: List[Dict], context: str = "analysis") -> str:
    """Generate response using the global model service manager"""
    return await model_service_manager.generate_response(messages, context)

def get_model_status() -> Dict[str, Any]:
    """Get status of all model services"""
    return model_service_manager.get_status()

def add_custom_service(service: ModelService):
    """Add a custom model service"""
    model_service_manager.add_service(service) 