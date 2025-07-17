# 🚀 Production Deployment Guide

This guide covers deploying your Code Analyzer to production with different AI model services and hosting platforms.

## 🎯 Model Service Architecture

Your code has been refactored with a **model service abstraction layer** (`model_service.py`) that supports:

### 🔧 Supported AI Services:
- **Google Gemini** (Fast, free quota)
- **OpenAI GPT** (Reliable, paid)
- **AWS Bedrock** (Claude, enterprise-grade)
- **Ollama** (Local fallback)

### 🔍 Key Model Usage Points:
```python
# In app.py, these methods use AI models:
- _explain_single_file()     # Line ~1007: File explanations
- _generate_overall_insights() # Line ~1080: Overall analysis
- _call_llm_hybrid()         # Line ~736: Smart model routing
```

## 🌐 Deployment Options

### 1. 🚀 **Serverless (Recommended)**

#### **A. Railway (Easiest)**
```bash
# 1. Push to GitHub (see GitHub Setup below)
# 2. Connect to Railway: https://railway.app
# 3. Add environment variables in Railway dashboard:
#    - GEMINI_API_KEY=your_key
#    - OPENAI_API_KEY=your_key (optional)
# 4. Deploy automatically from GitHub

# Railway will use railway.toml for configuration
```

#### **B. Vercel (Fast)**
```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Deploy
vercel --prod

# 3. Set environment variables in Vercel dashboard
# Note: Limited to 50MB, good for light usage
```

#### **C. Google Cloud Run (Scalable)**
```bash
# 1. Build and push Docker image
docker build -t gcr.io/YOUR_PROJECT/code-analyzer .
docker push gcr.io/YOUR_PROJECT/code-analyzer

# 2. Deploy to Cloud Run
gcloud run deploy code-analyzer \
  --image gcr.io/YOUR_PROJECT/code-analyzer \
  --platform managed \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY=your_key
```

### 2. 🏗️ **AWS SageMaker (Enterprise)**

#### **For ML Model Inference:**
```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name code-analyzer

# 2. Build and push Docker image
docker build -t code-analyzer .
docker tag code-analyzer:latest AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/code-analyzer:latest
docker push AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/code-analyzer:latest

# 3. Deploy SageMaker endpoint
aws sagemaker create-model --model-name code-analyzer-model \
  --primary-container Image=AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/code-analyzer:latest \
  --execution-role-arn arn:aws:iam::AWS_ACCOUNT:role/SageMakerExecutionRole

# 4. Create endpoint configuration and endpoint
# (See deployment/aws-sagemaker.yml for full config)
```

#### **For Web Application:**
```bash
# Use AWS ECS or EKS for containerized deployment
# Or AWS Lambda for serverless (with size limitations)
```

### 3. 🐳 **Docker Deployment**

#### **Single Container:**
```bash
# 1. Build image
docker build -t code-analyzer .

# 2. Run with environment variables
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  code-analyzer
```

#### **With Docker Compose:**
```bash
# 1. Create .env file from env-template.txt
cp env-template.txt .env
# Fill in your API keys

# 2. Start services
docker-compose up -d

# 3. Access at http://localhost:8501
```

## 🔑 Environment Setup

### 1. **Create Environment File:**
```bash
cp env-template.txt .env
# Edit .env with your API keys
```

### 2. **Required Environment Variables:**
```bash
# Choose at least one AI service:
GEMINI_API_KEY=your_gemini_key        # Free, fast
OPENAI_API_KEY=your_openai_key        # Paid, reliable
AWS_ACCESS_KEY_ID=your_aws_key        # For Bedrock
AWS_SECRET_ACCESS_KEY=your_aws_secret # For Bedrock
```

### 3. **Optional Configuration:**
```bash
STREAMLIT_SERVER_PORT=8501
MAX_REPO_SIZE_MB=500
MAX_ANALYSIS_TIME_MINUTES=10
```

## 📊 Model Service Configuration

### **Using Different AI Services:**

#### **1. Google Gemini (Default)**
```bash
# Free tier: 15 requests/minute
# Get API key: https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your_key"
```

#### **2. OpenAI GPT**
```bash
# Paid service, more reliable
# Get API key: https://platform.openai.com/account/api-keys
export OPENAI_API_KEY="your_key"
```

#### **3. AWS Bedrock (Enterprise)**
```bash
# Uses Claude models, enterprise-grade
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_DEFAULT_REGION="us-east-1"
```

#### **4. Custom Model Service**
```python
# Add your own model service in model_service.py
from model_service import ModelService, add_custom_service

class YourCustomService(ModelService):
    # Implement the abstract methods
    pass

# Add to the manager
add_custom_service(YourCustomService())
```

## 🔧 GitHub Setup

### **1. Create GitHub Repository:**
```bash
# Create repo on GitHub, then:
git remote add origin https://github.com/yourusername/code-analyzer.git
git branch -M main
git push -u origin main
```

### **2. GitHub Actions (CI/CD):**
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        run: |
          # Your deployment commands here
```

## 🏆 **Recommended Deployment Strategy**

### **For Development:**
```bash
# Use local Ollama + Railway
1. Install Ollama locally for development
2. Deploy to Railway for staging
3. Use environment variables for API keys
```

### **For Production:**
```bash
# Use paid services for reliability
1. Primary: OpenAI GPT-3.5-turbo
2. Fallback: AWS Bedrock Claude
3. Deploy on Google Cloud Run or AWS ECS
4. Set up monitoring and logging
```

### **For Enterprise:**
```bash
# Use AWS SageMaker for ML inference
1. Deploy model to SageMaker endpoint
2. Use AWS Bedrock for AI services
3. VPC deployment for security
4. Auto-scaling and load balancing
```

## 🔍 **Model Usage Examples**

### **Switching Models in Production:**
```python
# In your environment
export OPENAI_API_KEY="your_key"  # Primary
export GEMINI_API_KEY="your_key"   # Fallback

# The model service will automatically:
# 1. Try OpenAI first (more reliable)
# 2. Fall back to Gemini if OpenAI fails
# 3. Use local Ollama as last resort
```

### **Custom Model Integration:**
```python
# For AWS SageMaker endpoint
class SageMakerService(ModelService):
    def __init__(self, endpoint_name):
        self.endpoint_name = endpoint_name
        self.client = boto3.client('sagemaker-runtime')
    
    async def generate_response(self, messages, **kwargs):
        # Call your SageMaker endpoint
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps({'messages': messages})
        )
        return json.loads(response['Body'].read())['response']
```

## 📈 **Performance Optimization**

### **1. Caching:**
```python
# Add Redis caching for repeated analyses
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_analysis(repo_url):
    return analyze_repository(repo_url)
```

### **2. Async Processing:**
```python
# Process multiple files concurrently
async def analyze_files_parallel(files):
    tasks = [analyze_file(f) for f in files]
    results = await asyncio.gather(*tasks)
    return results
```

### **3. Resource Management:**
```python
# Limit concurrent requests
semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
```

## 🚨 **Security Considerations**

### **1. API Key Management:**
```bash
# Use environment variables, never hardcode
# For production, use AWS Secrets Manager or similar
```

### **2. Rate Limiting:**
```python
# Implement rate limiting for API calls
# Use exponential backoff for retries
```

### **3. Input Validation:**
```python
# Validate repository URLs
# Sanitize file contents before analysis
```

## 🎯 **Next Steps**

1. **Choose your deployment method** (Railway recommended for simplicity)
2. **Get API keys** for your chosen AI service
3. **Set up environment variables**
4. **Test locally** with Docker
5. **Deploy to production**
6. **Monitor and optimize**

## 📞 **Support**

- **Model Service Issues**: Check `model_service.py` for service implementations
- **Deployment Issues**: Check the specific deployment configuration files
- **Performance Issues**: Enable logging and check usage statistics

---

**Your Code Analyzer is now ready for production deployment with multiple AI service options! 🚀** 