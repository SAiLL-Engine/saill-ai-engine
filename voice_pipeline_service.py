"""
SAiLL AI Engine - Voice Pipeline Integration Service
==================================================

FastAPI service that integrates the AI engine with the voice pipeline.
Provides REST endpoints for real-time conversation processing.

Phase 1: OpenAI API integration (1000-2000ms response time)
Phase 2: Local Llama integration (180-350ms response time)
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
import uuid
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import SAiLL AI engine components
from engines.manager import AIEngineManager, create_ai_engine_manager
from engines import ConversationType, ConversationContext
from industry_intelligence import IndustryIntelligenceFactory
from config import load_configuration, validate_configuration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SAiLL AI Engine - Voice Pipeline Service",
    description="AI conversation engine for the SAiLL voice pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI engine manager instance
ai_engine_manager: Optional[AIEngineManager] = None


# Pydantic models for API
class ConversationRequest(BaseModel):
    """Request model for conversation generation"""
    text: str = Field(..., description="Input text from customer")
    session_id: str = Field(..., description="Voice pipeline session ID")
    client_id: str = Field(..., description="Client company ID")
    customer_id: Optional[str] = Field(None, description="Customer ID")
    campaign_id: Optional[str] = Field(None, description="Campaign ID")
    conversation_type: str = Field("client_customer", description="Type of conversation")
    preferred_engine: Optional[str] = Field(None, description="Preferred AI engine")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class ConversationResponse(BaseModel):
    """Response model for conversation generation"""
    text: str = Field(..., description="Generated response text")
    session_id: str = Field(..., description="Voice pipeline session ID")
    processing_time: float = Field(..., description="Processing time in seconds")
    engine_name: str = Field(..., description="AI engine used")
    confidence: float = Field(..., description="Response confidence score")
    conversation_state: Dict[str, Any] = Field(default_factory=dict, description="Updated conversation state")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    manager_healthy: bool
    engines: Dict[str, Any]
    timestamp: float


class ConfigResponse(BaseModel):
    """Response model for configuration"""
    default_engine: str
    available_engines: List[str]
    failover_enabled: bool
    industry_intelligence_enabled: bool


# Dependency for getting AI engine manager
def get_ai_engine_manager() -> AIEngineManager:
    """Get the global AI engine manager instance"""
    if ai_engine_manager is None:
        raise HTTPException(
            status_code=503,
            detail="AI engine manager not initialized"
        )
    return ai_engine_manager


@app.on_event("startup")
async def startup_event():
    """Initialize the AI engine manager on startup"""
    global ai_engine_manager
    
    logger.info("Starting SAiLL AI Engine Voice Pipeline Service...")
    
    try:
        # Load environment variables
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            logger.info("Loaded environment variables")
        
        # Load configuration
        config = load_configuration()
        
        # Phase 1: Ensure OpenAI is the default engine
        if "engines" not in config:
            config["engines"] = {}
        
        config["engines"]["default_engine"] = "openai"
        config["engines"]["failover_enabled"] = True
        
        # Validate configuration
        validation_errors = validate_configuration(config)
        if validation_errors:
            logger.error(f"Configuration validation failed: {validation_errors}")
            raise Exception(f"Configuration validation failed: {validation_errors}")
        
        # Initialize industry intelligence
        industry_intelligence = IndustryIntelligenceFactory.create_service(
            service_type=config.get("industry_intelligence", {}).get("type", "mock"),
            config=config.get("industry_intelligence", {})
        )
        
        # Initialize AI engine manager
        ai_engine_manager = await create_ai_engine_manager(
            engine_configs=config.get("engines", {}),
            industry_intelligence_service=industry_intelligence
        )
        
        # Health check
        health_status = await ai_engine_manager.health_check()
        logger.info(f"Engine health status: {health_status}")
        
        if not health_status["manager_healthy"]:
            raise Exception("AI engine manager is not healthy")
        
        logger.info("AI Engine Voice Pipeline Service started successfully")
        logger.info(f"Available engines: {ai_engine_manager.list_available_engines()}")
        
    except Exception as e:
        logger.error(f"Failed to start AI engine service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global ai_engine_manager
    
    logger.info("Shutting down AI Engine Voice Pipeline Service...")
    
    if ai_engine_manager:
        await ai_engine_manager.cleanup()
        ai_engine_manager = None
    
    logger.info("AI Engine Service shutdown complete")


@app.get("/health", response_model=HealthResponse)
async def health_check(manager: AIEngineManager = Depends(get_ai_engine_manager)):
    """Health check endpoint"""
    try:
        health_status = await manager.health_check()
        return HealthResponse(
            status="healthy" if health_status["manager_healthy"] else "unhealthy",
            manager_healthy=health_status["manager_healthy"],
            engines=health_status["engines"],
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config", response_model=ConfigResponse)
async def get_config(manager: AIEngineManager = Depends(get_ai_engine_manager)):
    """Get current configuration"""
    try:
        available_engines = manager.list_available_engines()
        engine_names = [engine["name"] for engine in available_engines]
        
        return ConfigResponse(
            default_engine=manager._engine_configs.get("default_engine", "openai"),
            available_engines=engine_names,
            failover_enabled=manager._engine_configs.get("failover_enabled", False),
            industry_intelligence_enabled=manager._industry_intelligence_service is not None
        )
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engines")
async def list_engines(manager: AIEngineManager = Depends(get_ai_engine_manager)):
    """List available engines and their status"""
    try:
        return {
            "engines": manager.list_available_engines(),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to list engines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation", response_model=ConversationResponse)
async def generate_conversation_response(
    request: ConversationRequest,
    manager: AIEngineManager = Depends(get_ai_engine_manager)
):
    """
    Generate AI conversation response for voice pipeline
    
    This is the main endpoint used by the voice pipeline for real-time conversation.
    """
    start_time = time.time()
    
    try:
        logger.debug(f"Processing conversation request for session {request.session_id}")
        
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Input text cannot be empty"
            )
        
        # Create conversation context
        conversation_context = ConversationContext(
            client_id=request.client_id,
            customer_id=request.customer_id or "anonymous",
            campaign_id=request.campaign_id or "default"
        )
        
        # Add session context
        context_data = conversation_context.get_context_for_ai()
        context_data.update(request.context)
        context_data["session_id"] = request.session_id
        
        # Prepare conversation input
        conversation_input = {
            "text": request.text,
            "timestamp": time.time(),
            "session_id": request.session_id
        }
        
        # Default client configuration for Phase 1
        client_config = {
            "company_name": "SAiLL Demo Client",
            "primary_industry": "technology",
            "products_services": "AI sales automation",
            "subscription_tier": "enterprise",
            "voice_pipeline_mode": True
        }
        
        # Determine conversation type
        try:
            conv_type = ConversationType[request.conversation_type.upper()]
        except KeyError:
            conv_type = ConversationType.CLIENT_CUSTOMER
        
        # Generate response
        response = await manager.generate_response(
            conversation_input=conversation_input,
            conversation_context=context_data,
            client_config=client_config,
            conversation_type=conv_type,
            preferred_engine=request.preferred_engine
        )
        
        processing_time = time.time() - start_time
        
        logger.debug(
            f"Generated response for session {request.session_id}: "
            f"{len(response.get('text', ''))} chars in {processing_time:.3f}s"
        )
        
        return ConversationResponse(
            text=response.get("text", ""),
            session_id=request.session_id,
            processing_time=processing_time,
            engine_name=response.get("engine_name", "unknown"),
            confidence=response.get("confidence", 0.9),
            conversation_state=response.get("conversation_state", {}),
            metadata={
                "input_length": len(request.text),
                "output_length": len(response.get("text", "")),
                "response_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "client_id": request.client_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Conversation generation failed for session {request.session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversation generation failed: {str(e)}"
        )


@app.post("/conversation/stream")
async def stream_conversation_response():
    """
    Streaming conversation response endpoint
    
    This will be implemented in Phase 2 for real-time streaming responses.
    """
    raise HTTPException(
        status_code=501,
        detail="Streaming conversation not yet implemented (Phase 2 feature)"
    )


# Voice pipeline specific endpoints
@app.post("/voice-pipeline/conversation")
async def voice_pipeline_conversation(
    text: str,
    session_id: str,
    client_id: str = "demo_client",
    customer_id: Optional[str] = None,
    manager: AIEngineManager = Depends(get_ai_engine_manager)
):
    """
    Simplified voice pipeline conversation endpoint
    
    Used directly by the voice gateway for streamlined integration.
    """
    try:
        request = ConversationRequest(
            text=text,
            session_id=session_id,
            client_id=client_id,
            customer_id=customer_id,
            conversation_type="client_customer"
        )
        
        response = await generate_conversation_response(request, manager)
        
        return {
            "session_id": session_id,
            "response_text": response.text,
            "processing_time": response.processing_time,
            "engine_used": response.engine_name,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Voice pipeline conversation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/metrics")
async def get_metrics(manager: AIEngineManager = Depends(get_ai_engine_manager)):
    """Get performance metrics"""
    try:
        # Get engine metrics
        metrics = {}
        
        for engine_info in manager.list_available_engines():
            engine_name = engine_info["name"]
            if engine_info["initialized"]:
                # Engine-specific metrics would be gathered here
                metrics[engine_name] = {
                    "status": "healthy" if engine_info["healthy"] else "unhealthy",
                    "initialized": engine_info["initialized"],
                    "last_used": engine_info.get("last_used", None)
                }
        
        return {
            "engines": metrics,
            "system": {
                "uptime": time.time(),
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "voice_pipeline_service:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )
