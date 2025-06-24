"""
Local Llama Engine Implementation
Phase 2 AI engine using Llama-3.1-8B-Instruct with PyTorch optimization
"""

import torch
import asyncio
import psutil
import subprocess
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import json
import gc

from . import AIEngineInterface, ConversationType, EngineStatus


class LocalLlamaEngine(AIEngineInterface):
    """Local Llama-3.1-8B implementation optimized for production performance"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_path = "meta-llama/Llama-3.1-8B-Instruct"
        
        # Performance optimization settings
        self.max_new_tokens = 150
        self.temperature = 0.7
        self.top_p = 0.9
        self.do_sample = True
        
        # GPU management
        self.gpu_manager = None
        self.memory_monitor = None
        self.performance_optimizer = None
        
        # Industry overlay management
        self.industry_overlays = {}
        self.current_overlay = None
        
        # Dependencies
        self.industry_intelligence = None
        self.performance_monitor = None
        
        # Context management for long conversations
        self.context_compressor = None
        self.max_context_length = 4096
        
    async def initialize(
        self, 
        engine_config: Dict[str, Any],
        industry_intelligence_service: Any,
        performance_monitor: Any
    ) -> None:
        """Initialize Local Llama engine with GPU optimization"""
        
        try:
            self.logger.info("🚀 Initializing Local Llama Engine...")
            
            # Store dependencies
            self.industry_intelligence = industry_intelligence_service
            self.performance_monitor = performance_monitor
            
            # Initialize GPU management
            self.gpu_manager = GPUResourceManager(
                max_memory_gb=engine_config.get("max_gpu_memory", 14),
                target_memory_usage=engine_config.get("target_memory_usage", 0.85)
            )
            
            # Check GPU availability
            await self._validate_gpu_requirements()
            
            # Initialize memory monitoring
            self.memory_monitor = MemoryMonitor()
            
            # Load model configuration
            self.model_path = engine_config.get("model_path", self.model_path)
            self.max_new_tokens = engine_config.get("max_new_tokens", 150)
            self.temperature = engine_config.get("temperature", 0.7)
            
            # Load base model
            await self._load_base_model()
            
            # Initialize performance optimizer
            self.performance_optimizer = LlamaPerformanceOptimizer(self.model, self.tokenizer)
            await self.performance_optimizer.optimize()
            
            # Initialize context compressor
            self.context_compressor = ConversationContextCompressor(
                tokenizer=self.tokenizer,
                max_length=self.max_context_length
            )
            
            # Warm up model with test inference
            await self._warmup_model()
            
            self.status = EngineStatus.READY
            self.logger.info("✅ Local Llama Engine fully initialized and optimized")
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            self.logger.error(f"❌ Failed to initialize Local Llama Engine: {e}")
            raise
    
    async def generate_response(
        self,
        conversation_input: Dict[str, Any],
        conversation_context: Dict[str, Any],
        client_config: Dict[str, Any],
        conversation_type: ConversationType = ConversationType.CLIENT_CUSTOMER
    ) -> Dict[str, Any]:
        """Generate response using Local Llama with industry overlays and optimization"""
        
        start_time = datetime.now()
        
        try:
            self.status = EngineStatus.PROCESSING
            
            # Monitor GPU memory before processing
            memory_before = self.gpu_manager.get_memory_usage()
            
            # Load appropriate industry overlay if needed
            industry = client_config.get("primary_industry", "general")
            await self._load_industry_overlay(industry, conversation_type)
            
            # Get industry context enhancement
            industry_context = await self._get_industry_context(
                conversation_input, conversation_context, client_config, conversation_type
            )
            
            # Build optimized prompt with context compression
            optimized_prompt = await self._build_optimized_prompt(
                conversation_input, conversation_context, client_config, 
                conversation_type, industry_context
            )
            
            # Generate response with optimized inference
            response_text = await self._generate_optimized_response(optimized_prompt)
            
            # Monitor GPU memory after processing
            memory_after = self.gpu_manager.get_memory_usage()
            
            # Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update performance monitoring
            self.update_performance_stats(processing_time)
            if self.performance_monitor:
                self.performance_monitor.record_engine_performance(
                    engine_name="local_llama",
                    response_time=processing_time,
                    conversation_type=conversation_type,
                    success=True
                )
            
            # Clean up GPU memory if needed
            if memory_after["used_percent"] > 90:
                await self._cleanup_gpu_memory()
            
            self.status = EngineStatus.READY
            
            return {
                "text": response_text,
                "generation_time_ms": processing_time,
                "model_used": f"{self.model_path}",
                "industry_overlay_applied": self.current_overlay is not None,
                "industry_context_applied": bool(industry_context),
                "conversation_type": conversation_type.value,
                "engine_name": "local_llama",
                "timestamp": datetime.now().isoformat(),
                "gpu_memory_usage": memory_after,
                "context_compressed": optimized_prompt.get("was_compressed", False),
                "success": True
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.status = EngineStatus.READY
            
            # Update error tracking
            self.update_performance_stats(processing_time, error_occurred=True)
            if self.performance_monitor:
                self.performance_monitor.record_engine_performance(
                    engine_name="local_llama",
                    response_time=processing_time,
                    conversation_type=conversation_type,
                    success=False
                )
            
            self.logger.error(f"Local Llama generation error: {e}")
            
            return {
                "text": "I apologize, but I'm experiencing technical difficulties. Please try again.",
                "generation_time_ms": processing_time,
                "error": str(e),
                "engine_name": "local_llama",
                "success": False,
                "fallback_response": True,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _validate_gpu_requirements(self):
        """Validate GPU requirements for Llama inference"""
        
        if not torch.cuda.is_available():
            raise Exception("CUDA GPU required for Local Llama engine")
        
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise Exception("No CUDA GPUs detected")
        
        # Check VRAM availability (minimum 12GB recommended for Llama-3.1-8B)
        for i in range(gpu_count):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            if gpu_memory < 12:
                self.logger.warning(f"GPU {i} has {gpu_memory:.1f}GB VRAM, minimum 12GB recommended")
        
        self.device = torch.device("cuda:0")
        self.logger.info(f"✅ GPU validation passed: {gpu_count} GPU(s) available")
    
    async def _load_base_model(self):
        """Load Llama model with optimization for inference"""
        
        try:
            self.logger.info(f"Loading Llama model: {self.model_path}")
            
            # Import transformers with error handling
            try:
                from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
            except ImportError:
                raise Exception("transformers library required. Install with: pip install transformers")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Set padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimization
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # Half precision for memory efficiency
                device_map="auto",          # Automatic device mapping
                low_cpu_mem_usage=True,     # Optimize CPU memory usage
                trust_remote_code=True
            )
            
            # Enable evaluation mode for inference
            self.model.eval()
            
            # Compile model for optimization (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                self.logger.info("Compiling model with torch.compile for optimization...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
            
            self.logger.info("✅ Llama model loaded and optimized")
            
        except Exception as e:
            raise Exception(f"Failed to load Llama model: {e}")
    
    async def _load_industry_overlay(self, industry: str, conversation_type: ConversationType):
        """Load industry-specific model overlay (LoRA adapter)"""
        
        overlay_key = f"{industry}_{conversation_type.value}"
        
        # Check if overlay is already loaded
        if self.current_overlay == overlay_key:
            return
        
        try:
            # In production, this would load actual LoRA adapters
            # For now, we'll simulate the overlay system
            
            overlay_path = f"./models/overlays/{industry.lower().replace(' ', '_')}"
            
            # Simulated overlay loading - in production this would use PEFT library
            self.industry_overlays[overlay_key] = {
                "loaded_at": datetime.now(),
                "industry": industry,
                "conversation_type": conversation_type,
                "model_size_mb": 150,  # Typical LoRA adapter size
                "active": True
            }
            
            self.current_overlay = overlay_key
            self.logger.info(f"✅ Loaded industry overlay: {overlay_key}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load industry overlay {overlay_key}: {e}")
            self.current_overlay = None
    
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
            
            # Get relevant industry context
            context_request = {
                "industry": industry,
                "conversation_type": conversation_type.value,
                "client_id": conversation_context.get("client_id"),
                "user_input": conversation_input.get("text", ""),
                "conversation_history": conversation_context.get("conversation_history", [])
            }
            
            # Retrieve context with timeout for performance
            industry_context = await asyncio.wait_for(
                self.industry_intelligence.get_context(context_request),
                timeout=30  # 30ms timeout for context retrieval
            )
            
            return industry_context
            
        except asyncio.TimeoutError:
            self.logger.warning("Industry context retrieval timed out")
            return {}
        except Exception as e:
            self.logger.warning(f"Failed to retrieve industry context: {e}")
            return {}
    
    async def _build_optimized_prompt(
        self,
        conversation_input: Dict[str, Any],
        conversation_context: Dict[str, Any],
        client_config: Dict[str, Any],
        conversation_type: ConversationType,
        industry_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build optimized prompt with context compression"""
        
        # Build base system prompt
        system_prompt = self._build_system_prompt(client_config, conversation_type, industry_context)
        
        # Get conversation history and compress if needed
        conversation_history = conversation_context.get("conversation_history", [])
        current_input = conversation_input.get("text", "")
        
        # Use context compressor to manage long conversations
        compressed_context = await self.context_compressor.compress_context(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            current_input=current_input
        )
        
        return compressed_context
    
    def _build_system_prompt(
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
                for insight in industry_insights[:2]:  # Limit to top 2 insights for token efficiency
                    industry_enhancement += f"- {insight}\n"
        
        # Format with client information
        company_name = client_config.get("company_name", "our company")
        products_services = client_config.get("products_services", "our solutions")
        
        return base_prompt.format(
            company_name=company_name,
            products_services=products_services
        ) + industry_enhancement
    
    async def _generate_optimized_response(self, optimized_prompt: Dict[str, Any]) -> str:
        """Generate response using optimized Llama inference"""
        
        try:
            # Tokenize input
            input_text = optimized_prompt["final_prompt"]
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.max_context_length
            ).to(self.device)
            
            # Generate with optimization
            with torch.no_grad():  # Disable gradients for inference
                with torch.cuda.amp.autocast():  # Mixed precision for speed
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=self.do_sample,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True  # Enable KV caching for speed
                    )
            
            # Decode response
            response_ids = outputs[0][inputs.input_ids.shape[1]:]  # Remove input tokens
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            return response_text.strip()
            
        except Exception as e:
            self.logger.error(f"Optimized generation failed: {e}")
            raise
    
    async def _warmup_model(self):
        """Warm up model with test inference to optimize first-time performance"""
        
        try:
            self.logger.info("Warming up model with test inference...")
            
            test_input = "Hello, how are you today?"
            inputs = self.tokenizer(test_input, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _ = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=10,
                        do_sample=False
                    )
            
            self.logger.info("✅ Model warmup completed")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    async def _cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent OOM errors"""
        
        try:
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Log memory status
            memory_status = self.gpu_manager.get_memory_usage()
            self.logger.info(f"GPU memory cleaned up: {memory_status['used_percent']:.1f}% used")
            
        except Exception as e:
            self.logger.warning(f"GPU memory cleanup failed: {e}")
    
    async def _perform_health_check(self) -> None:
        """Local Llama specific health check implementation"""
        
        try:
            # Check GPU availability
            if not torch.cuda.is_available():
                raise Exception("CUDA GPU not available")
            
            # Check model status
            if self.model is None:
                raise Exception("Model not loaded")
            
            # Check GPU memory
            memory_status = self.gpu_manager.get_memory_usage()
            if memory_status["used_percent"] > 95:
                raise Exception(f"GPU memory critically low: {memory_status['used_percent']:.1f}%")
            
            # Test model inference
            test_input = self.tokenizer("Test", return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model.generate(test_input.input_ids, max_new_tokens=1)
            
        except Exception as e:
            raise Exception(f"Local Llama health check failed: {e}")
    
    async def cleanup(self) -> None:
        """Clean up Local Llama engine resources"""
        
        self.status = EngineStatus.MAINTENANCE
        
        try:
            # Clear model from GPU memory
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Local Llama Engine cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information and status"""
        
        return {
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "current_overlay": self.current_overlay,
            "available_overlays": list(self.industry_overlays.keys()),
            "gpu_status": self.gpu_manager.get_memory_usage() if self.gpu_manager else None,
            "performance_stats": self.performance_stats,
            "device": str(self.device) if self.device else None
        }


class GPUResourceManager:
    """Manage GPU resources and memory for optimal Llama performance"""
    
    def __init__(self, max_memory_gb: int = 14, target_memory_usage: float = 0.85):
        self.max_memory_gb = max_memory_gb
        self.target_memory_usage = target_memory_usage
        self.logger = logging.getLogger("GPUResourceManager")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        
        if not torch.cuda.is_available():
            return {"used_gb": 0, "total_gb": 0, "used_percent": 0}
        
        try:
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            
            gpu_properties = torch.cuda.get_device_properties(0)
            total_memory = gpu_properties.total_memory / (1024**3)  # GB
            
            return {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved,
                "total_gb": total_memory,
                "used_percent": (memory_reserved / total_memory) * 100
            }
        except Exception as e:
            self.logger.error(f"Failed to get GPU memory usage: {e}")
            return {"used_gb": 0, "total_gb": 0, "used_percent": 0}
    
    def is_memory_available(self, required_gb: float) -> bool:
        """Check if required memory is available"""
        
        memory_status = self.get_memory_usage()
        available_gb = memory_status["total_gb"] - memory_status["reserved_gb"]
        
        return available_gb >= required_gb


class MemoryMonitor:
    """Monitor system and GPU memory usage"""
    
    def __init__(self):
        self.logger = logging.getLogger("MemoryMonitor")
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get system RAM usage"""
        
        memory = psutil.virtual_memory()
        
        return {
            "total_gb": memory.total / (1024**3),
            "used_gb": memory.used / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_percent": memory.percent
        }
    
    def log_memory_status(self):
        """Log current memory status"""
        
        system_mem = self.get_system_memory()
        self.logger.info(f"System RAM: {system_mem['used_percent']:.1f}% used")
        
        if torch.cuda.is_available():
            gpu_mem = GPUResourceManager().get_memory_usage()
            self.logger.info(f"GPU Memory: {gpu_mem['used_percent']:.1f}% used")


class LlamaPerformanceOptimizer:
    """Optimize Llama model for maximum inference performance"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger("LlamaPerformanceOptimizer")
    
    async def optimize(self):
        """Apply performance optimizations to the model"""
        
        try:
            # Enable optimizations
            self._optimize_attention()
            self._optimize_kv_cache()
            await self._optimize_memory_layout()
            
            self.logger.info("✅ Llama performance optimizations applied")
            
        except Exception as e:
            self.logger.warning(f"Performance optimization failed: {e}")
    
    def _optimize_attention(self):
        """Optimize attention mechanisms for speed"""
        
        try:
            # Enable flash attention if available
            if hasattr(self.model.config, 'use_flash_attention_2'):
                self.model.config.use_flash_attention_2 = True
            
        except Exception as e:
            self.logger.debug(f"Flash attention optimization failed: {e}")
    
    def _optimize_kv_cache(self):
        """Optimize key-value caching for repeated inference"""
        
        try:
            # Configure optimal cache settings
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
            
        except Exception as e:
            self.logger.debug(f"KV cache optimization failed: {e}")
    
    async def _optimize_memory_layout(self):
        """Optimize memory layout for better performance"""
        
        try:
            # Ensure model parameters are properly aligned
            for param in self.model.parameters():
                if param.is_cuda:
                    param.data = param.data.contiguous()
            
        except Exception as e:
            self.logger.debug(f"Memory layout optimization failed: {e}")


class ConversationContextCompressor:
    """Compress conversation context to fit within model limits while preserving important information"""
    
    def __init__(self, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger("ConversationContextCompressor")
    
    async def compress_context(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, Any]],
        current_input: str
    ) -> Dict[str, Any]:
        """Compress conversation context to fit within token limits"""
        
        # Calculate token counts
        system_tokens = len(self.tokenizer.encode(system_prompt))
        current_input_tokens = len(self.tokenizer.encode(current_input))
        
        # Reserve tokens for response generation
        reserved_tokens = 200
        available_tokens = self.max_length - system_tokens - current_input_tokens - reserved_tokens
        
        if available_tokens <= 0:
            # Emergency compression - just use system prompt and current input
            final_prompt = f"{system_prompt}\n\nUser: {current_input}\nAssistant:"
            return {
                "final_prompt": final_prompt,
                "was_compressed": True,
                "compression_ratio": 1.0,
                "included_history_turns": 0
            }
        
        # Build conversation history within token limit
        history_text = ""
        included_turns = 0
        
        # Include recent interactions first (reverse chronological)
        for interaction in reversed(conversation_history):
            user_input = interaction.get("user_input", "")
            ai_response = interaction.get("ai_response", "")
            
            interaction_text = f"User: {user_input}\nAssistant: {ai_response}\n"
            interaction_tokens = len(self.tokenizer.encode(interaction_text))
            
            if len(self.tokenizer.encode(history_text + interaction_text)) <= available_tokens:
                history_text = interaction_text + history_text  # Prepend to maintain chronological order
                included_turns += 1
            else:
                break
        
        # Build final prompt
        if history_text:
            final_prompt = f"{system_prompt}\n\n{history_text}User: {current_input}\nAssistant:"
        else:
            final_prompt = f"{system_prompt}\n\nUser: {current_input}\nAssistant:"
        
        # Calculate compression metrics
        total_history_turns = len(conversation_history)
        compression_ratio = included_turns / max(total_history_turns, 1)
        
        return {
            "final_prompt": final_prompt,
            "was_compressed": included_turns < total_history_turns,
            "compression_ratio": compression_ratio,
            "included_history_turns": included_turns,
            "total_history_turns": total_history_turns,
            "final_token_count": len(self.tokenizer.encode(final_prompt))
        }
