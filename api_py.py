"""
FastAPI REST API for Servicing Intelligence
Provides endpoints for classification, summarization, and complete pipeline
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import logging

from inference import get_pipeline
from utils import load_config

# Initialize FastAPI app
app = FastAPI(
    title="Servicing Intelligence API",
    description="NLP API for ticket classification and summarization",
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

# Load configuration
config = load_config()
api_config = config.get('api', {})

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Request/Response Models
class TextInput(BaseModel):
    """Input model for text"""
    text: str = Field(..., description="Input text to process", min_length=1)
    
    class Config:
        schema_extra = {
            "example": {
                "text": "My internet connection keeps dropping every few minutes."
            }
        }


class ClassificationResponse(BaseModel):
    """Response model for classification"""
    category: str = Field(..., description="Predicted category")
    confidence: float = Field(..., description="Prediction confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "category": "technical_issue",
                "confidence": 0.92
            }
        }


class SummarizationResponse(BaseModel):
    """Response model for summarization"""
    summary: str = Field(..., description="Generated summary")
    
    class Config:
        schema_extra = {
            "example": {
                "summary": "Internet connection drops frequently"
            }
        }


class PipelineResponse(BaseModel):
    """Response model for complete pipeline"""
    category: Optional[str] = Field(None, description="Predicted category")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    summary: Optional[str] = Field(None, description="Generated summary")
    original_text: str = Field(..., description="Original input text")
    
    class Config:
        schema_extra = {
            "example": {
                "category": "technical_issue",
                "confidence": 0.92,
                "summary": "Internet connection drops frequently",
                "original_text": "My internet connection keeps dropping."
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


# Initialize pipeline on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        logger.info("Initializing inference pipeline...")
        get_pipeline(config)
        logger.info("✓ Inference pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Servicing Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "classify": "/classify",
            "summarize": "/summarize",
            "pipeline": "/pipeline"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    
    Returns service health status
    """
    try:
        # Check if pipeline is initialized
        pipeline = get_pipeline(config)
        
        return HealthResponse(
            status="healthy",
            message="Service is running and models are loaded"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy: Models not loaded"
        )


@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify(input_data: TextInput):
    """
    Classify ticket text into categories
    
    - **text**: Ticket text to classify
    
    Returns predicted category and confidence score
    """
    try:
        logger.info(f"Classification request received")
        
        pipeline = get_pipeline(config)
        category, confidence = pipeline.classify_text(input_data.text)
        
        logger.info(f"Classification result: {category} ({confidence:.3f})")
        
        return ClassificationResponse(
            category=category,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@app.post("/summarize", response_model=SummarizationResponse, tags=["Summarization"])
async def summarize(input_data: TextInput):
    """
    Generate summary for ticket text
    
    - **text**: Ticket text to summarize
    
    Returns generated summary
    """
    try:
        logger.info(f"Summarization request received")
        
        pipeline = get_pipeline(config)
        summary = pipeline.summarize_text(input_data.text)
        
        logger.info(f"Summarization result: {summary}")
        
        return SummarizationResponse(summary=summary)
    
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@app.post("/pipeline", response_model=PipelineResponse, tags=["Pipeline"])
async def complete_pipeline(input_data: TextInput):
    """
    Complete pipeline: classify and summarize text
    
    - **text**: Ticket text to process
    
    Returns classification and summary results
    """
    try:
        logger.info(f"Pipeline request received")
        
        pipeline = get_pipeline(config)
        result = pipeline.pipeline(input_data.text)
        
        logger.info(f"Pipeline result: {result.get('category')} / {result.get('summary')}")
        
        return PipelineResponse(**result)
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}"
        )


@app.get("/categories", tags=["Information"])
async def get_categories():
    """
    Get list of available ticket categories
    
    Returns list of categories the classifier can predict
    """
    try:
        categories = config.get('categories', [])
        return {
            "categories": categories,
            "count": len(categories)
        }
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch categories"
        )


@app.get("/model-info", tags=["Information"])
async def get_model_info():
    """
    Get information about loaded models
    
    Returns model names and configurations
    """
    return {
        "classifier": {
            "model": config['classifier']['model_name'],
            "num_labels": config['classifier']['num_labels'],
            "max_length": config['classifier']['max_length']
        },
        "summarizer": {
            "model": config['summarizer']['model_name'],
            "max_source_length": config['summarizer']['max_source_length'],
            "max_target_length": config['summarizer']['max_target_length']
        }
    }


def main():
    """Run the API server"""
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    reload = api_config.get('reload', False)
    log_level = api_config.get('log_level', 'info')
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


if __name__ == "__main__":
    main()
