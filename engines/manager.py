"""
Engine Factory and Registration System
Creates and manages AI engine instances with automatic registration
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Type, List
from datetime import datetime

from . import AIEngineInterface, EngineFactory, ConversationType, PerformanceMonitor
from .openai_engine import OpenAIEngine
from .local_llama_engine import LocalLlamaEngine  
from .hybrid_engine import HybridIntelligentEngine


class AIEngineManager:
    """Central manager for AI engine lifecycle and operations"""
    
    def __init__(self):
        self.engines: Dict[str, AIEngineInterface] = {}
        self.engine_configs: Dict[str, Dict[str, Any]] = {}
        self.performance_monitor = PerformanceMonitor()
        self.industry_intelligence = None
        self.logger = logging.getLogger("AIEngineManager")
        
        # Register available engines
        self._register_engines()
        
        # Manager state
        self.initialized = False
        self.default_engine = None
        
    def _register_engines(self):
        """Register all available AI engine implementations"""
        
        # Register core engines
        EngineFactory.register_engine("openai", OpenAIEngine)
        EngineFactory.register_engine("local_llama", LocalLlamaEngine)
        EngineFactory.register_engine("hybrid", HybridIntelligentEngine)
        
        self.logger.info("✅ Registered AI engines: openai, local_llama, hybrid")
    
    async def initialize(
        self, 
        engine_configs: Dict[str, Dict[str, Any]],
        industry_intelligence_service: Any = None
    ) -> None:
        """Initialize AI engine manager with configurations"""
        
        try:
            self.industry_intelligence = industry_intelligence_service
            self.engine_configs = engine_configs
            
            # Initialize engines based on configuration
            initialization_tasks = []
            
            for engine_name, config in engine_configs.items():
                if config.get("enabled", True):
                    task = self._initialize_engine(engine_name, config)
                    initialization_tasks.append(task)
            
            # Initialize engines concurrently
            if initialization_tasks:
                results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
                
                # Check initialization results
                for i, result in enumerate(results):
                    engine_name = list(engine_configs.keys())[i]
                    if isinstance(result, Exception):
                        self.logger.error(f"Failed to initialize {engine_name}: {result}")
                    else:
                        self.logger.info(f"✅ Successfully initialized {engine_name}")
            
            # Set default engine
            self._set_default_engine()
            
            self.initialized = True
            self.logger.info(f"🎯 AI Engine Manager initialized with {len(self.engines)} engines")
            
        except Exception as e:
            self.logger.error(f"❌ AI Engine Manager initialization failed: {e}")
            raise
    
    async def _initialize_engine(self, engine_name: str, config: Dict[str, Any]) -> None:
        """Initialize a specific AI engine"""
        
        try:
            # Create engine instance
            engine = await EngineFactory.create_engine(
                engine_name=engine_name,
                engine_config=config,
                industry_intelligence_service=self.industry_intelligence,
                performance_monitor=self.performance_monitor
            )
            
            # Store engine instance
            self.engines[engine_name] = engine
            
            self.logger.info(f"✅ Engine {engine_name} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize engine {engine_name}: {e}")
            raise
    
    def _set_default_engine(self):
        """Set default engine based on availability and preferences"""
        
        # Priority order: hybrid -> local_llama -> openai
        engine_priority = ["hybrid", "local_llama", "openai"]
        
        for engine_name in engine_priority:
            if engine_name in self.engines:
                self.default_engine = engine_name
                self.logger.info(f"🎯 Default engine set to: {engine_name}")
                break
        
        if not self.default_engine and self.engines:
            # Fallback to any available engine
            self.default_engine = list(self.engines.keys())[0]
            self.logger.warning(f"⚠️ Fallback default engine set to: {self.default_engine}")
    
    async def generate_response(
        self,
        conversation_input: Dict[str, Any],
        conversation_context: Dict[str, Any],
        client_config: Dict[str, Any],
        conversation_type: ConversationType = ConversationType.CLIENT_CUSTOMER,
        preferred_engine: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using specified or default engine"""
        
        if not self.initialized:
            raise Exception("AI Engine Manager not initialized")
        
        # Determine engine to use
        engine_name = preferred_engine or self.default_engine
        
        if not engine_name or engine_name not in self.engines:
            available_engines = list(self.engines.keys())
            raise Exception(f"Engine {engine_name} not available. Available: {available_engines}")
        
        # Get engine instance
        engine = self.engines[engine_name]
        
        try:
            # Generate response
            response = await engine.generate_response(
                conversation_input=conversation_input,
                conversation_context=conversation_context,
                client_config=client_config,
                conversation_type=conversation_type
            )
            
            # Add manager metadata
            response["manager_metadata"] = {
                "engine_used": engine_name,
                "was_preferred": preferred_engine is not None,
                "available_engines": list(self.engines.keys()),
                "manager_version": "1.0.0"
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed on {engine_name}: {e}")
            
            # Attempt fallback to default engine if different
            if preferred_engine and preferred_engine != self.default_engine:
                self.logger.info(f"Attempting fallback to default engine: {self.default_engine}")
                
                try:
                    fallback_response = await self.engines[self.default_engine].generate_response(
                        conversation_input=conversation_input,
                        conversation_context=conversation_context,
                        client_config=client_config,
                        conversation_type=conversation_type
                    )
                    
                    fallback_response["manager_metadata"] = {
                        "engine_used": self.default_engine,
                        "was_fallback": True,
                        "original_engine": preferred_engine,
                        "fallback_reason": str(e)
                    }
                    
                    return fallback_response
                    
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed: {fallback_error}")
            
            # Both primary and fallback failed
            raise Exception(f"All engines failed. Primary error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on all engines"""
        
        health_status = {
            "manager_healthy": True,
            "engines": {},
            "summary": {
                "total_engines": len(self.engines),
                "healthy_engines": 0,
                "unhealthy_engines": 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Check each engine
        for engine_name, engine in self.engines.items():
            try:
                engine_healthy = await engine.health_check()
                engine_status = engine.get_status()
                
                health_status["engines"][engine_name] = {
                    "healthy": engine_healthy,
                    "status": engine_status,
                    "last_check": datetime.now().isoformat()
                }
                
                if engine_healthy:
                    health_status["summary"]["healthy_engines"] += 1
                else:
                    health_status["summary"]["unhealthy_engines"] += 1
                    
            except Exception as e:
                health_status["engines"][engine_name] = {
                    "healthy": False,
                    "error": str(e),
                    "last_check": datetime.now().isoformat()
                }
                health_status["summary"]["unhealthy_engines"] += 1
        
        # Overall manager health
        if health_status["summary"]["healthy_engines"] == 0:
            health_status["manager_healthy"] = False
        
        return health_status
    
    def get_engine_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for all engines"""
        
        summary = {
            "performance_monitor": self.performance_monitor.get_performance_summary(),
            "engine_details": {},
            "recommendations": []
        }
        
        # Get individual engine performance
        for engine_name, engine in self.engines.items():
            engine_stats = engine.get_status()
            
            summary["engine_details"][engine_name] = {
                "status": engine_stats,
                "type": type(engine).__name__
            }
            
            # Add engine-specific information
            if hasattr(engine, 'get_cost_summary'):  # OpenAI engine
                summary["engine_details"][engine_name]["cost_summary"] = engine.get_cost_summary()
            
            if hasattr(engine, 'get_model_info'):  # Local Llama engine
                summary["engine_details"][engine_name]["model_info"] = engine.get_model_info()
            
            if hasattr(engine, 'get_routing_summary'):  # Hybrid engine
                summary["engine_details"][engine_name]["routing_summary"] = engine.get_routing_summary()
        
        return summary
    
    def list_available_engines(self) -> List[Dict[str, Any]]:
        """List all available engines with their capabilities"""
        
        engines_info = []
        
        for engine_name in EngineFactory.list_available_engines():
            engine_info = {
                "name": engine_name,
                "initialized": engine_name in self.engines,
                "is_default": engine_name == self.default_engine,
                "capabilities": self._get_engine_capabilities(engine_name)
            }
            
            if engine_name in self.engines:
                engine_info["status"] = self.engines[engine_name].get_status()
            
            engines_info.append(engine_info)
        
        return engines_info
    
    def _get_engine_capabilities(self, engine_name: str) -> Dict[str, Any]:
        """Get capabilities for a specific engine type"""
        
        capabilities = {
            "openai": {
                "response_time": "1-2 seconds",
                "concurrent_capacity": "10-15 conversations",
                "cost_model": "per-token",
                "specialization": "complex conversations",
                "advantages": ["high quality", "broad knowledge", "reliable"]
            },
            "local_llama": {
                "response_time": "180-350ms",
                "concurrent_capacity": "60-80 conversations", 
                "cost_model": "hardware only",
                "specialization": "high-performance local inference",
                "advantages": ["fast", "cost effective", "private", "customizable"]
            },
            "hybrid": {
                "response_time": "180ms-2s (adaptive)",
                "concurrent_capacity": "60-80 conversations",
                "cost_model": "optimized routing",
                "specialization": "intelligent engine selection",
                "advantages": ["optimal performance", "automatic failover", "cost optimization"]
            }
        }
        
        return capabilities.get(engine_name, {"description": "Unknown engine type"})
    
    async def reload_engine(self, engine_name: str, new_config: Dict[str, Any]) -> bool:
        """Reload a specific engine with new configuration"""
        
        try:
            # Clean up existing engine
            if engine_name in self.engines:
                await self.engines[engine_name].cleanup()
                del self.engines[engine_name]
            
            # Initialize with new configuration
            await self._initialize_engine(engine_name, new_config)
            
            # Update stored configuration
            self.engine_configs[engine_name] = new_config
            
            # Update default engine if needed
            if engine_name == self.default_engine or not self.default_engine:
                self._set_default_engine()
            
            self.logger.info(f"✅ Successfully reloaded engine: {engine_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reload engine {engine_name}: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up all engines and manager resources"""
        
        self.logger.info("🧹 Cleaning up AI Engine Manager...")
        
        cleanup_tasks = []
        for engine_name, engine in self.engines.items():
            cleanup_tasks.append(engine.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.engines.clear()
        self.initialized = False
        
        self.logger.info("✅ AI Engine Manager cleanup completed")


# Convenience function for creating engine manager
async def create_ai_engine_manager(
    engine_configs: Dict[str, Dict[str, Any]],
    industry_intelligence_service: Any = None
) -> AIEngineManager:
    """Create and initialize AI engine manager"""
    
    manager = AIEngineManager()
    await manager.initialize(engine_configs, industry_intelligence_service)
    return manager


# Configuration validation
class EngineConfigValidator:
    """Validate AI engine configurations"""
    
    @staticmethod
    def validate_openai_config(config: Dict[str, Any]) -> List[str]:
        """Validate OpenAI engine configuration"""
        
        errors = []
        
        # Required fields
        if not config.get("openai_api_key") and not os.getenv("OPENAI_API_KEY"):
            errors.append("OpenAI API key required")
        
        # Optional validation
        if "model_name" in config and not config["model_name"].startswith("gpt"):
            errors.append("Invalid OpenAI model name")
        
        if "max_tokens" in config and (config["max_tokens"] < 1 or config["max_tokens"] > 4000):
            errors.append("max_tokens must be between 1 and 4000")
        
        return errors
    
    @staticmethod
    def validate_llama_config(config: Dict[str, Any]) -> List[str]:
        """Validate Local Llama engine configuration"""
        
        errors = []
        
        # GPU requirements
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA GPU required for Local Llama engine")
        
        # Memory requirements
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            min_memory = config.get("min_gpu_memory", 12)
            if gpu_memory < min_memory:
                errors.append(f"Insufficient GPU memory: {gpu_memory:.1f}GB < {min_memory}GB required")
        
        return errors
    
    @staticmethod
    def validate_hybrid_config(config: Dict[str, Any]) -> List[str]:
        """Validate Hybrid engine configuration"""
        
        errors = []
        
        # Check that at least one sub-engine is configured
        openai_config = config.get("openai_config", {})
        llama_config = config.get("llama_config", {})
        
        if not openai_config and not llama_config:
            errors.append("Hybrid engine requires at least OpenAI or Llama configuration")
        
        # Validate sub-configurations
        if openai_config:
            errors.extend(EngineConfigValidator.validate_openai_config(openai_config))
        
        if llama_config:
            errors.extend(EngineConfigValidator.validate_llama_config(llama_config))
        
        return errors
    
    @staticmethod
    def validate_all_configs(engine_configs: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Validate all engine configurations"""
        
        validation_results = {}
        
        for engine_name, config in engine_configs.items():
            if engine_name == "openai":
                validation_results[engine_name] = EngineConfigValidator.validate_openai_config(config)
            elif engine_name == "local_llama":
                validation_results[engine_name] = EngineConfigValidator.validate_llama_config(config)
            elif engine_name == "hybrid":
                validation_results[engine_name] = EngineConfigValidator.validate_hybrid_config(config)
            else:
                validation_results[engine_name] = [f"Unknown engine type: {engine_name}"]
        
        return validation_results


# Export main classes
__all__ = [
    'AIEngineManager',
    'create_ai_engine_manager', 
    'EngineConfigValidator'
]
