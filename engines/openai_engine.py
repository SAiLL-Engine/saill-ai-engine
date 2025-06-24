"""
OpenAI Engine Implementation
Phase 1 AI engine using OpenAI GPT-4 API with streaming responses
"""

import os
import openai
import asyncio
import json
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime
import logging

from . import AIEngineInterface, ConversationType, EngineStatus


class OpenAIEngine(AIEngineInterface):
    """OpenAI GPT-4 implementation for Phase 1 validation and rapid deployment"""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.model_name = "gpt-4o"
        self.max_tokens = 150
        self.temperature = 0.7
        self.stream = True
        
        self.industry_intelligence = None
        self.performance_monitor = None
        
        # OpenAI specific configuration
        self.rate_limit_manager = None
        self.cost_tracker = {
            "total_tokens_used": 0,
            "estimated_cost": 0.0,
            "requests_count": 0
        }
    
    async def initialize(
        self, 
        engine_config: Dict[str, Any],
        industry_intelligence_service: Any,
        performance_monitor: Any
    ) -> None:
        """Initialize OpenAI engine with API key and configuration"""
        
        try:
            # Set OpenAI API key
            api_key = engine_config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided in config or environment")
            
            # Initialize OpenAI client
            self.client = openai.AsyncOpenAI(api_key=api_key)
            
            # Configure model parameters
            self.model_name = engine_config.get("model_name", "gpt-4o")
            self.max_tokens = engine_config.get("max_tokens", 150)
            self.temperature = engine_config.get("temperature", 0.7)
            self.stream = engine_config.get("stream", True)
            
            # Store dependencies
            self.industry_intelligence = industry_intelligence_service
            self.performance_monitor = performance_monitor
            
            # Initialize rate limiting
            self.rate_limit_manager = OpenAIRateLimitManager(
                max_requests_per_minute=engine_config.get("rate_limit_rpm", 500),
                max_tokens_per_minute=engine_config.get("rate_limit_tpm", 30000)
            )
            
            # Test connection
            await self._test_connection()
            
            self.status = EngineStatus.READY
            self.logger.info(f"✅ OpenAI Engine initialized with model: {self.model_name}")
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            self.logger.error(f"❌ Failed to initialize OpenAI Engine: {e}")
            raise
    
    async def generate_response(
        self,
        conversation_input: Dict[str, Any],
        conversation_context: Dict[str, Any],
        client_config: Dict[str, Any],
        conversation_type: ConversationType = ConversationType.CLIENT_CUSTOMER
    ) -> Dict[str, Any]:
        """Generate response using OpenAI GPT-4 with industry context enhancement"""
        
        start_time = datetime.now()
        
        try:
            self.status = EngineStatus.PROCESSING
            
            # Apply rate limiting
            await self.rate_limit_manager.acquire()
            
            # Get industry context enhancement
            industry_context = await self._get_industry_context(
                conversation_input, conversation_context, client_config, conversation_type
            )
            
            # Build prompt with industry intelligence
            system_prompt = await self._build_system_prompt(
                client_config, conversation_type, industry_context
            )
            
            # Build conversation messages
            messages = await self._build_conversation_messages(
                system_prompt, conversation_input, conversation_context
            )
            
            # Generate response with OpenAI
            if self.stream:
                response_text, token_usage = await self._generate_streaming_response(messages)
            else:
                response_text, token_usage = await self._generate_direct_response(messages)
            
            # Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update cost tracking
            self._update_cost_tracking(token_usage)
            
            # Update performance monitoring
            self.update_performance_stats(processing_time)
            if self.performance_monitor:
                self.performance_monitor.record_engine_performance(
                    engine_name="openai",
                    response_time=processing_time,
                    conversation_type=conversation_type,
                    success=True
                )
            
            self.status = EngineStatus.READY
            
            return {
                "text": response_text,
                "generation_time_ms": processing_time,
                "token_usage": token_usage,
                "model_used": self.model_name,
                "industry_context_applied": bool(industry_context),
                "conversation_type": conversation_type.value,
                "engine_name": "openai",
                "timestamp": datetime.now().isoformat(),
                "cost_estimate": self._calculate_request_cost(token_usage),
                "success": True
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.status = EngineStatus.READY
            
            # Update error tracking
            self.update_performance_stats(processing_time, error_occurred=True)
            if self.performance_monitor:
                self.performance_monitor.record_engine_performance(
                    engine_name="openai",
                    response_time=processing_time,
                    conversation_type=conversation_type,
                    success=False
                )
            
            self.logger.error(f"OpenAI generation error: {e}")
            
            return {
                "text": "I apologize, but I'm experiencing technical difficulties. Please try again.",
                "generation_time_ms": processing_time,
                "error": str(e),
                "engine_name": "openai",
                "success": False,
                "fallback_response": True,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_industry_context(
        self, 
        conversation_input: Dict[str, Any], 
        conversation_context: Dict[str, Any],
        client_config: Dict[str, Any],
        conversation_type: ConversationType
    ) -> Dict[str, Any]:
        """Retrieve industry-specific context for conversation enhancement"""
        
        if not self.industry_intelligence:
            return {}
        
        try:
            # Get industry from client configuration
            industry = client_config.get("primary_industry", "general")
            
            # Get relevant industry context based on conversation
            context_request = {
                "industry": industry,
                "conversation_type": conversation_type.value,
                "client_id": conversation_context.get("client_id"),
                "user_input": conversation_input.get("text", ""),
                "conversation_history": conversation_context.get("conversation_history", [])
            }
            
            # Retrieve context from industry intelligence service
            industry_context = await self.industry_intelligence.get_context(context_request)
            
            return industry_context
            
        except Exception as e:
            self.logger.warning(f"Failed to retrieve industry context: {e}")
            return {}
    
    async def _build_system_prompt(
        self, 
        client_config: Dict[str, Any], 
        conversation_type: ConversationType,
        industry_context: Dict[str, Any]
    ) -> str:
        """Build system prompt with client and industry-specific instructions"""
        
        base_prompt = """You are an AI sales assistant for {company_name}. Your role is to have natural, helpful conversations with customers about {products_services}.

Key Guidelines:
- Be conversational and natural, like a knowledgeable human representative
- Listen carefully and respond to customer needs and questions
- Provide helpful information about products and services
- Build rapport and trust through genuine interaction
- Keep responses concise but informative (1-3 sentences)
- If interrupted, acknowledge gracefully and adapt to the new direction
"""
        
        # Customize based on conversation type
        if conversation_type == ConversationType.META_SALES_BRAIN:
            base_prompt = """You are an AI assistant specializing in sales automation and AI conversation systems. Your role is to help potential clients understand how AI can transform their sales operations.

Key Guidelines:
- Focus on AI sales automation benefits and ROI
- Demonstrate expertise in conversation AI and CRM integration
- Address concerns about AI replacing human interaction
- Provide specific examples of successful AI implementations
- Keep responses professional but approachable (1-3 sentences)
"""
        
        # Add industry-specific context
        industry_enhancement = ""
        if industry_context:
            industry_name = industry_context.get("industry_name", "")
            industry_insights = industry_context.get("key_insights", [])
            
            if industry_insights:
                industry_enhancement = f"\n\nIndustry Context ({industry_name}):\n"
                for insight in industry_insights[:3]:  # Limit to top 3 insights
                    industry_enhancement += f"- {insight}\n"
        
        # Format with client information
        company_name = client_config.get("company_name", "our company")
        products_services = client_config.get("products_services", "our solutions")
        
        return base_prompt.format(
            company_name=company_name,
            products_services=products_services
        ) + industry_enhancement
    
    async def _build_conversation_messages(
        self, 
        system_prompt: str, 
        conversation_input: Dict[str, Any],
        conversation_context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Build message array for OpenAI API including conversation history"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 5 interactions to stay within token limits)
        conversation_history = conversation_context.get("conversation_history", [])
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
        for interaction in recent_history:
            messages.append({"role": "user", "content": interaction["user_input"]})
            messages.append({"role": "assistant", "content": interaction["ai_response"]})
        
        # Add current user input
        current_input = conversation_input.get("text", "")
        if current_input:
            messages.append({"role": "user", "content": current_input})
        
        return messages
    
    async def _generate_streaming_response(self, messages: List[Dict[str, str]]) -> tuple[str, Dict[str, int]]:
        """Generate response using OpenAI streaming for faster perceived response time"""
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            response_text = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response_text += chunk.choices[0].delta.content
            
            # Estimate token usage (actual usage not available in streaming)
            estimated_tokens = {
                "prompt_tokens": sum(len(msg["content"].split()) for msg in messages) * 1.3,  # Rough estimate
                "completion_tokens": len(response_text.split()) * 1.3,
                "total_tokens": 0
            }
            estimated_tokens["total_tokens"] = estimated_tokens["prompt_tokens"] + estimated_tokens["completion_tokens"]
            
            return response_text.strip(), estimated_tokens
            
        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise
    
    async def _generate_direct_response(self, messages: List[Dict[str, str]]) -> tuple[str, Dict[str, int]]:
        """Generate response using direct OpenAI API call"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            response_text = response.choices[0].message.content
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return response_text.strip(), token_usage
            
        except Exception as e:
            self.logger.error(f"Direct generation failed: {e}")
            raise
    
    def _update_cost_tracking(self, token_usage: Dict[str, int]):
        """Update internal cost tracking based on token usage"""
        
        self.cost_tracker["total_tokens_used"] += token_usage.get("total_tokens", 0)
        self.cost_tracker["requests_count"] += 1
        
        # Calculate estimated cost (GPT-4 pricing: ~$0.03/1K prompt tokens, ~$0.06/1K completion tokens)
        prompt_cost = (token_usage.get("prompt_tokens", 0) / 1000) * 0.03
        completion_cost = (token_usage.get("completion_tokens", 0) / 1000) * 0.06
        
        self.cost_tracker["estimated_cost"] += prompt_cost + completion_cost
    
    def _calculate_request_cost(self, token_usage: Dict[str, int]) -> float:
        """Calculate cost for current request"""
        
        prompt_cost = (token_usage.get("prompt_tokens", 0) / 1000) * 0.03
        completion_cost = (token_usage.get("completion_tokens", 0) / 1000) * 0.06
        
        return prompt_cost + completion_cost
    
    async def _test_connection(self):
        """Test OpenAI API connection"""
        
        try:
            test_response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            if not test_response.choices:
                raise Exception("No response from OpenAI API")
                
        except Exception as e:
            raise Exception(f"OpenAI API connection test failed: {e}")
    
    async def _perform_health_check(self) -> None:
        """OpenAI specific health check implementation"""
        
        try:
            # Test API connectivity
            await self._test_connection()
            
            # Check rate limiting status
            if self.rate_limit_manager:
                rate_status = self.rate_limit_manager.get_status()
                if rate_status["requests_available"] < 10:
                    self.logger.warning("OpenAI rate limit approaching")
            
        except Exception as e:
            raise Exception(f"OpenAI health check failed: {e}")
    
    async def cleanup(self) -> None:
        """Clean up OpenAI engine resources"""
        
        self.status = EngineStatus.MAINTENANCE
        
        if self.client:
            await self.client.close()
        
        self.logger.info("OpenAI Engine cleaned up")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary"""
        
        return {
            "total_requests": self.cost_tracker["requests_count"],
            "total_tokens_used": self.cost_tracker["total_tokens_used"],
            "estimated_total_cost": self.cost_tracker["estimated_cost"],
            "average_cost_per_request": (
                self.cost_tracker["estimated_cost"] / max(self.cost_tracker["requests_count"], 1)
            ),
            "average_tokens_per_request": (
                self.cost_tracker["total_tokens_used"] / max(self.cost_tracker["requests_count"], 1)
            )
        }


class OpenAIRateLimitManager:
    """Manage OpenAI API rate limits to prevent throttling"""
    
    def __init__(self, max_requests_per_minute: int = 500, max_tokens_per_minute: int = 30000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        
        self.request_timestamps = []
        self.token_usage_timeline = []
        
        self.logger = logging.getLogger("OpenAIRateLimitManager")
    
    async def acquire(self, estimated_tokens: int = 100):
        """Acquire permission to make API request, respecting rate limits"""
        
        current_time = datetime.now()
        
        # Clean old entries (older than 1 minute)
        self._clean_old_entries(current_time)
        
        # Check request rate limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_timestamps[0]).total_seconds()
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                self._clean_old_entries(datetime.now())
        
        # Check token rate limit
        current_token_usage = sum(entry["tokens"] for entry in self.token_usage_timeline)
        if current_token_usage + estimated_tokens > self.max_tokens_per_minute:
            sleep_time = 60 - (current_time - self.token_usage_timeline[0]["timestamp"]).total_seconds()
            if sleep_time > 0:
                self.logger.info(f"Token rate limit reached, sleeping for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                self._clean_old_entries(datetime.now())
        
        # Record this request
        self.request_timestamps.append(current_time)
        self.token_usage_timeline.append({
            "timestamp": current_time,
            "tokens": estimated_tokens
        })
    
    def _clean_old_entries(self, current_time: datetime):
        """Remove entries older than 1 minute"""
        
        cutoff_time = current_time.timestamp() - 60
        
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if ts.timestamp() > cutoff_time
        ]
        
        self.token_usage_timeline = [
            entry for entry in self.token_usage_timeline 
            if entry["timestamp"].timestamp() > cutoff_time
        ]
    
    def get_status(self) -> Dict[str, int]:
        """Get current rate limit status"""
        
        current_time = datetime.now()
        self._clean_old_entries(current_time)
        
        current_token_usage = sum(entry["tokens"] for entry in self.token_usage_timeline)
        
        return {
            "requests_used": len(self.request_timestamps),
            "requests_available": max(0, self.max_requests_per_minute - len(self.request_timestamps)),
            "tokens_used": current_token_usage,
            "tokens_available": max(0, self.max_tokens_per_minute - current_token_usage),
            "requests_limit": self.max_requests_per_minute,
            "tokens_limit": self.max_tokens_per_minute
        }
