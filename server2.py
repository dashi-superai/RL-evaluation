from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional, List
import logging
from transformers import TextIteratorStreamer
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Inference API",
    description="API for Large Language Model inference using Transformers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_NAME = "./models2"  # Change to your model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables
model = None
tokenizer = None

# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_length: Optional[int] = Field(100, description="Maximum length of generated text")
    temperature: Optional[float] = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: Optional[int] = Field(50, ge=0, description="Top-k sampling")
    num_return_sequences: Optional[int] = Field(1, ge=1, le=5, description="Number of sequences to return")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    repetition_penalty: Optional[float] = Field(1.0, ge=1.0, le=2.0, description="Repetition penalty")

class GenerateResponse(BaseModel):
    generated_text: List[str]
    prompt: str
    model: str

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_length: Optional[int] = Field(512, description="Maximum length")
    temperature: Optional[float] = Field(0.7, ge=0.1, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    top_k: Optional[float] = Field(40, description="")

class ChatResponse(BaseModel):
    response: str
    model: str

def load_model():
    """Load the LLM model and tokenizer"""
    global model, tokenizer
    
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info(f"Using device: {DEVICE}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            # low_cpu_mem_usage=True
        )
        
        if DEVICE == "cpu":
            model = model.to(DEVICE)
        
        model = torch.compile(model)
        model.eval()
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Inference API is running",
        "model": MODEL_NAME,
        "device": DEVICE,
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "model_name": MODEL_NAME,
        "device": DEVICE
    }

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "parameters": sum(p.numel() for p in model.parameters()),
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt
    """
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                num_return_sequences=request.num_return_sequences,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode outputs
        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return GenerateResponse(
            generated_text=generated_texts,
            prompt=request.prompt,
            model=MODEL_NAME
        )
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with conversation history
    """
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Format messages into a prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        
        prompt += "Assistant:"
        
        logger.info(f"Chat request with {len(request.messages)} messages")
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(DEVICE)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new response
        response = full_response[len(prompt):].strip()
        
        return ChatResponse(
            response=response,
            model=MODEL_NAME
        )
        
    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tokenize")
async def tokenize(text: str):
    """Tokenize input text"""
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    return {
        "text": text,
        "tokens": tokens,
        "token_count": len(tokens),
        "decoded": decoded
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5002,
        log_level="debug"
    )