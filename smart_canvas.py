from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from PIL import Image
import io
import json
import uvicorn
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Smart Canvas API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure Gemini API
GEMINI_API_KEY = os.getenv("AIzaSyDs2e48gjTFdXebiE_7fIPl0PdJQ0Zh8G8")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

@app.get("/")
async def root():
    return {"message": "Smart Canvas API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Smart Canvas API"}

@app.post("/test")
async def test_endpoint():
    print("🧪 Test endpoint called!")
    return {"message": "Test successful!", "timestamp": "now"}

import base64

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

@app.post("/solve-problem")
async def solve_problem(request: ImageRequest):
    try:
        print("📥 Received solve request")
        
        # Decode base64 image
        print("🖼️ Decoding base64 image...")
        image_data = base64.b64decode(request.image.split(',')[1])  # Remove data:image/png;base64, prefix
        image = Image.open(io.BytesIO(image_data))
        print(f"✅ Image decoded successfully: {image.size} pixels, mode: {image.mode}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("🔄 Converted image to RGB")
        
        # Create prompt for Gemini
        prompt = """
        Analyze this image and solve any math or problem you see. And consider if any text is there in image follow those instructions then.
        
        Please format your response as clean HTML with inline CSS styling that looks like a modern webpage.
        Use proper HTML structure with:
        - A main heading for the problem type
        - Styled sections for step-by-step solution
        - Highlighted final answer
        - Professional styling with colors, spacing, and typography
        
        Make it look like a beautiful, well-structured webpage with proper CSS styling.
        Include the problem type, step-by-step solution, and final answer.
        
        Use inline CSS for all styling so it renders properly.
        """
        
        # Get response from Gemini
        print("🤖 Sending request to Gemini AI...")
        response = model.generate_content([prompt, image])
        response_text = response.text.strip()
        print(f"✅ Received response from Gemini: {len(response_text)} characters")
        
        # Extract information using regex
        def extract_problem_type(text):
            # Look for problem type indicators
            type_patterns = [
                r'(?i)(algebra|arithmetic|geometry|calculus|trigonometry|statistics)',
                r'(?i)(math|mathematical|equation|formula)',
                r'(?i)(addition|subtraction|multiplication|division)',
                r'(?i)(linear|quadratic|polynomial)'
            ]
            for pattern in type_patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).lower()
            return "math problem"
        
        def extract_steps(text):
            # Extract numbered steps or bullet points
            step_patterns = [
                r'(?i)step\s*\d+[:\.]?\s*([^\n]+)',
                r'(?i)\d+[:\.\)]\s*([^\n]+)',
                r'(?i)•\s*([^\n]+)',
                r'(?i)-\s*([^\n]+)'
            ]
            
            steps = []
            for pattern in step_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    steps.extend(matches)
                    break
            
            # If no structured steps found, split by sentences
            if not steps:
                sentences = re.split(r'[.!?]+', text)
                steps = [s.strip() for s in sentences if len(s.strip()) > 10][:5]
            
            return steps[:5]  # Limit to 5 steps
        
        def extract_final_answer(text):
            # Look for final answer patterns
            answer_patterns = [
                r'(?i)(?:final\s*answer|answer|result|solution)[:\s]*([^\n\.]+)',
                r'(?i)(?:therefore|thus|so)[,\s]*([^\n\.]+)',
                r'(?i)(?:equals?|=)\s*([^\n\.]+)',
                r'(?i)(?:x\s*=|y\s*=|z\s*=)\s*([^\n\.]+)'
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()
            
            # If no specific answer found, take the last meaningful sentence
            sentences = re.split(r'[.!?]+', text)
            for sentence in reversed(sentences):
                if len(sentence.strip()) > 5:
                    return sentence.strip()
            
            return "See solution above"
        
        # Return HTML response directly
        print("✅ Successfully processed request")
        return {
            "success": True,
            "solution": response_text
        }
    
    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "problem_type": "Error",
            "problem_description": "",
            "solution": "",
            "step_by_step": [],
            "final_answer": "",
            "confidence": "low"
        }

if __name__ == "__main__":
    print("🚀 Starting Smart Canvas Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("🔧 Health Check: http://localhost:8000/health")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

