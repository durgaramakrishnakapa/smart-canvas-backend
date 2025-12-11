"""
Agentic Smart Canvas - Minimal LangGraph + Serper Implementation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from PIL import Image
import io
import base64
import uvicorn
from typing import Dict, Any, List, TypedDict
from dotenv import load_dotenv
import requests

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Agentic Smart Canvas API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Configure APIs with multiple Gemini keys for load balancing
import random

GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"), 
    os.getenv("GEMINI_API_KEY_3")
]

# Filter out None values and ensure we have at least one key
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key]

if not GEMINI_API_KEYS:
    raise ValueError("Please set at least one GEMINI_API_KEY in your .env file")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not SERPER_API_KEY:
    print("⚠️  Warning: SERPER_API_KEY not set. Search functionality will be disabled.")

def get_random_gemini_model():
    """Get a Gemini model with a random API key for load balancing"""
    api_key = random.choice(GEMINI_API_KEYS)
    key_index = GEMINI_API_KEYS.index(api_key) + 1
    print(f"🔑 Using Gemini API Key #{key_index}")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

# Initialize with first key (will be randomized on each use)
genai.configure(api_key=GEMINI_API_KEYS[0])
model = genai.GenerativeModel('gemini-2.5-flash')

# State Definition
class AgentState(TypedDict):
    image_data: Any
    analysis: str
    search_results: Dict[str, Any]
    solution: str
    needs_search: bool
    search_query: str

class ImageRequest(BaseModel):
    image: str

# Serper Search Tool
def search_serper(query: str) -> Dict[str, Any]:
    """Search using Serper API and return structured results"""
    try:
        print(f"🔍 Searching Serper for: {query}")
        response = requests.post(
            "https://google.serper.dev/search",
            headers={'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'},
            json={'q': query, 'num': 1}
        )
        data = response.json()
        
        results = {
            'organic': [],
            'answer_box': None,
            'knowledge_graph': None,
            'query': query
        }
        
        # Extract organic results
        for item in data.get('organic', [])[:3]:
            results['organic'].append({
                'title': item.get('title', ''),
                'snippet': item.get('snippet', ''),
                'link': item.get('link', '')
            })
        
        # Extract answer box if available
        if 'answerBox' in data:
            ab = data['answerBox']
            results['answer_box'] = {
                'answer': ab.get('answer', ab.get('snippet', '')),
                'title': ab.get('title', 'Direct Answer'),
                'source': ab.get('link', '')
            }
        
        # Extract knowledge graph if available
        if 'knowledgeGraph' in data:
            kg = data['knowledgeGraph']
            results['knowledge_graph'] = {
                'title': kg.get('title', ''),
                'description': kg.get('description', ''),
                'source': kg.get('descriptionLink', '')
            }
        
        print(f"✅ Found {len(results['organic'])} organic results")
        
        # Debug: Print search results to see what we're getting
        if results.get('answer_box'):
            print(f"📦 Answer Box: {results['answer_box']['answer']}")
        
        for i, result in enumerate(results['organic'][:2], 1):
            print(f"🔍 Result {i}: {result['title']} - {result['snippet'][:100]}...")
        
        return results
        
    except Exception as e:
        print(f"❌ Serper search failed: {str(e)}")
        return {'organic': [], 'answer_box': None, 'knowledge_graph': None, 'query': query, 'error': str(e)}

# LangGraph Nodes
def analyze_node(state: AgentState) -> AgentState:
    """Analyze image and determine if search is needed"""
    prompt = """Analyze this image carefully and determine:
    1. What type of problem, question, or content do you see?
    2. Can you solve this with your existing knowledge, or do you ABSOLUTELY need current/external information?
    
    ONLY use external search if the problem involves:
    - Current date/time, recent events, or breaking news (after 2024)
    - Real-time data like stock prices, weather, sports scores
    - Very recent scientific discoveries or technological developments
    - Current political events or recent policy changes
    - Live statistics or data that changes frequently
    
    DO NOT search for:
    - Basic math problems (algebra, calculus, geometry, arithmetic)
    - Well-established scientific concepts, formulas, or constants
    - Historical facts or events before 2024
    - General knowledge questions
    - Programming concepts or coding problems
    - Standard academic subjects (physics, chemistry, biology, etc.)
    - Common sense reasoning or logic problems
    
    Be conservative - only search if you're absolutely certain you need current information.
    
    If you need external information, respond with:
    "SEARCH: [specific search query]"
    
    If you can solve it with your existing knowledge, respond with:
    "NO_SEARCH: [brief analysis of what you see]"
    """
    
    try:
        # Use random API key for load balancing
        model = get_random_gemini_model()
        response = model.generate_content([prompt, state["image_data"]])
        result = response.text.strip()
        
        if "SEARCH:" in result:
            search_query = result.split("SEARCH:")[1].strip()
            
            # Additional conservative check - filter out common non-search queries
            search_keywords_to_avoid = [
                "math", "equation", "solve", "calculate", "formula", "algebra", 
                "geometry", "physics", "chemistry", "biology", "programming",
                "code", "algorithm", "logic", "reasoning", "problem", "question",
                "basic", "simple", "elementary", "standard", "common", "general"
            ]
            
            # Check if the search query contains terms that suggest it's basic knowledge
            query_lower = search_query.lower()
            should_avoid_search = any(keyword in query_lower for keyword in search_keywords_to_avoid)
            
            # Also check for time-sensitive keywords that DO need search
            time_sensitive_keywords = [
                "today", "now", "current", "latest", "recent", "2024", "2025",
                "breaking", "news", "live", "real-time", "stock", "price",
                "weather", "score", "election", "poll", "date", "sensex", 
                "nifty", "market", "index", "bse", "nse", "share"
            ]
            
            needs_current_info = any(keyword in query_lower for keyword in time_sensitive_keywords)
            
            # Improve search queries for specific requests
            if "sensex" in query_lower or "bse" in query_lower:
                search_query = "Sensex current value today BSE index price latest"
            elif "nifty" in query_lower or "nse" in query_lower:
                search_query = "Nifty current value today NSE index price latest"
            elif "date" in query_lower or "today" in query_lower:
                search_query = "what is today's date current date December 2025"
            
            # Final decision: search only if it's time-sensitive and not basic knowledge
            if needs_current_info and not should_avoid_search:
                state["needs_search"] = True
                state["search_query"] = search_query
                state["analysis"] = f"Identified need for current information: {search_query}"
            else:
                state["needs_search"] = False
                state["search_query"] = ""
                state["analysis"] = f"Can solve with existing knowledge: {result.replace('SEARCH:', '').strip()}"
        else:
            state["needs_search"] = False
            state["search_query"] = ""
            state["analysis"] = result.replace("NO_SEARCH:", "").strip()
            
    except Exception as e:
        state["analysis"] = f"Analysis failed: {str(e)}"
        state["needs_search"] = False
        state["search_query"] = ""
    
    return state

def search_node(state: AgentState) -> AgentState:
    """Search for external information"""
    if state["needs_search"] and state["search_query"]:
        search_results = search_serper(state["search_query"])
        state["search_results"] = search_results
    else:
        state["search_results"] = {}
    return state

def solve_node(state: AgentState) -> AgentState:
    """Solve the problem with enhanced context"""
    
    # Build context with search results
    context_parts = [f"Analysis: {state['analysis']}"]
    
    if state["search_results"] and state["search_results"].get('organic'):
        context_parts.append("\n=== CURRENT EXTERNAL INFORMATION (USE THIS DATA) ===")
        
        # Add answer box if available - this is usually the most accurate
        if state["search_results"].get('answer_box'):
            ab = state["search_results"]['answer_box']
            context_parts.append(f"CURRENT ANSWER: {ab['answer']}")
            context_parts.append(f"Source: {ab.get('source', 'Direct search result')}")
        
        # Add knowledge graph if available
        if state["search_results"].get('knowledge_graph'):
            kg = state["search_results"]['knowledge_graph']
            context_parts.append(f"VERIFIED INFO: {kg['title']} - {kg['description']}")
        
        # Add organic results with emphasis on current data
        context_parts.append("\nLATEST SEARCH RESULTS:")
        for i, result in enumerate(state["search_results"]['organic'][:3], 1):
            context_parts.append(f"{i}. {result['title']}: {result['snippet']}")
        
        context_parts.append("\n=== END EXTERNAL INFORMATION ===")
        context_parts.append("IMPORTANT: Use the above external information as your primary source of truth, especially for dates and current information.")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Solve the problem in this image with clear explanations and proper formatting.

CRITICAL: If external information is provided below, use that as your primary source of truth.

Context:
{context}

RESPONSE GUIDELINES:
- Don't describe what you see in the image (avoid "The image shows...")
- Focus on solving the problem with clear explanations
- Show your work and reasoning process
- Use structured format with proper sections

RESPONSE STRUCTURE:
- For SIMPLE questions: Brief explanation + direct answer
- For COMPLEX problems: Step-by-step solution with clear reasoning
- For CURRENT DATA queries: Use external information + explain the source

INSTRUCTIONS:
1. Use external information as definitive answer when provided
2. For current data (prices, dates), rely on search results not training data
3. Show the solution process clearly
4. Provide neat, organized explanations

Format as clean HTML with professional styling:
- Main heading for the problem type
- Clear sections for step-by-step solutions
- Highlighted final answer
- Use dark text colors (#111827, #374151, #1f2937) for readability
- Use light backgrounds (#ffffff, #f9fafb, #f3f4f6) for containers
- Professional spacing and typography
- Use inline CSS for all styling

Create a beautiful, well-structured solution with proper explanations."""
    
    try:
        # Use random API key for load balancing
        model = get_random_gemini_model()
        response = model.generate_content([prompt, state["image_data"]])
        state["solution"] = response.text.strip()
    except Exception as e:
        state["solution"] = f"<div style='color: red; padding: 20px;'>❌ Solution failed: {str(e)}</div>"
    
    return state

# Conditional routing
def should_search(state: AgentState) -> str:
    return "search" if state["needs_search"] else "solve"

# Create LangGraph
def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("search", search_node)
    workflow.add_node("solve", solve_node)
    
    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges("analyze", should_search, {"search": "search", "solve": "solve"})
    workflow.add_edge("search", "solve")
    workflow.add_edge("solve", END)
    
    return workflow.compile()

# Global graph instance
graph = create_graph()

def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image"""
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert('RGB') if image.mode != 'RGB' else image

def format_html_response(state: AgentState) -> str:
    """Format response as HTML - return clean AI-generated HTML directly"""
    
    # Always return the AI-generated solution directly without extra wrapper
    # This ensures clean output without metadata headers
    return state["solution"].strip()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "LangGraph Smart Canvas API", "search": "Serper enabled"}

@app.get("/health")
async def health():
    return {"status": "healthy", "langgraph": True, "serper": bool(SERPER_API_KEY)}

@app.post("/solve-problem")
async def solve_problem(request: ImageRequest):
    try:
        # Decode image
        image = decode_image(request.image)
        
        # Initialize state
        initial_state = AgentState(
            image_data=image,
            analysis="",
            search_results={},
            solution="",
            needs_search=False,
            search_query=""
        )
        
        # Execute LangGraph
        final_state = graph.invoke(initial_state)
        
        return {
            "success": True,
            "solution": format_html_response(final_state),
            "agent_metadata": {
                "search_used": bool(final_state["search_results"].get('organic')),
                "search_query": final_state.get("search_query", ""),
                "framework": "LangGraph + Serper"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "solution": f"<div style='color: red; padding: 20px;'>❌ Error: {str(e)}</div>"
        }


import uvicorn
    
    # Get port from environment (for Render deployment)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    print("🚀 LangGraph + Serper Smart Canvas")
    print("📊 Flow: smart_analyze_solve → [search] → solve")
    print(f"📡 Server starting on 0.0.0.0:{port}")
    print(f"🔑 Loaded {len(GEMINI_API_KEYS)} Gemini API keys")
    print(f"🔍 Serper API: {'✅ Configured' if SERPER_API_KEY else '❌ Not configured'}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
