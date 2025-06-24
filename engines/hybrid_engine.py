"""
Hybrid Intelligent Engine Implementation
Intelligent routing between OpenAI and Local Llama engines based on context and performance
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import random

from . import AIEngineInterface, ConversationType, EngineStatus
from .openai_engine import OpenAIEngine
from .local_llama_engine import LocalLlamaEngine


class HybridIntelligentEngine(AIEngineInterface):
    """Intelligent routing between OpenAI and Local Llama based on context and performance"""
    
    def __init__(self):
        super().__init__()
        self.openai_engine = None
        self.local_llama_engine = None
        self.routing_intelligence = None
        self.performance_monitor = None
        
        # Engine health monitoring
        self.engine_health = {
            "openai": {"available": False, "last_check": None, "response_time": None},
            "local_llama": {"available": False, "last_check": None, "response_time": None}
        }
        
        # Routing configuration
        self.routing_config = {
            "prefer_local": True,  # Prefer local engine when available
            "failover_enabled": True,  # Enable automatic failover
            "load_balancing": False,  # Load balance between engines
            "quality_threshold": 4.0,  # Minimum quality score for engine selection
            "latency_threshold": 1000  # Maximum acceptable latency (ms)
        }
        
        # Performance tracking
        self.routing_stats = {
            "total_requests": 0,
            "openai_requests": 0,
            "local_llama_requests": 0,
            "failover_events": 0,
            "average_decision_time": 0
        }
    
    async def initialize(
        self, 
        engine_config: Dict[str, Any],
        industry_intelligence_service: Any,
        performance_monitor: Any
    ) -> None:
        """Initialize hybrid engine with both implementations"""
        
        try:
            self.performance_monitor = performance_monitor
            self.routing_intelligence = IntelligentRoutingEngine()
            
            # Update routing configuration
            self.routing_config.update(engine_config.get("routing_config", {}))
            
            # Initialize OpenAI engine
            try:
                self.openai_engine = OpenAIEngine()
                await self.openai_engine.initialize(
                    engine_config.get("openai_config", {}),
                    industry_intelligence_service,
                    performance_monitor
                )
                self.engine_health["openai"]["available"] = True
                self.logger.info("✅ OpenAI engine initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ OpenAI engine initialization failed: {e}")
                self.engine_health["openai"]["available"] = False
            
            # Initialize Local Llama engine
            try:
                self.local_llama_engine = LocalLlamaEngine()
                await self.local_llama_engine.initialize(
                    engine_config.get("llama_config", {}),
                    industry_intelligence_service,
                    performance_monitor
                )
                self.engine_health["local_llama"]["available"] = True
                self.logger.info("✅ Local Llama engine initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Local Llama engine initialization failed: {e}")
                self.engine_health["local_llama"]["available"] = False
            
            # Validate at least one engine is available
            if not any(engine["available"] for engine in self.engine_health.values()):
                raise Exception("No AI engines available - both OpenAI and Local Llama failed to initialize")
            
            self.status = EngineStatus.READY
            self.logger.info("✅ Hybrid Intelligent Engine initialized with available engines")
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            self.logger.error(f"❌ Failed to initialize Hybrid Engine: {e}")
            raise
    
    async def generate_response(
        self,
        conversation_input: Dict[str, Any],
        conversation_context: Dict[str, Any],
        client_config: Dict[str, Any],
        conversation_type: ConversationType = ConversationType.CLIENT_CUSTOMER
    ) -> Dict[str, Any]:
        """Intelligently route to optimal AI engine and handle failover"""
        
        start_time = datetime.now()
        
        try:
            self.status = EngineStatus.PROCESSING
            
            # Perform routing decision
            routing_decision = await self._make_routing_decision(
                conversation_input, conversation_context, client_config, conversation_type
            )
            
            selected_engine = routing_decision["selected_engine"]
            reasoning = routing_decision["reasoning"]
            
            self.logger.debug(f"Routing decision: {selected_engine} - {reasoning}")
            
            # Attempt primary engine
            try:
                response = await self._execute_on_engine(
                    selected_engine, conversation_input, conversation_context, 
                    client_config, conversation_type
                )
                
                # Update routing statistics
                self._update_routing_stats(selected_engine, success=True)
                
                # Add routing metadata to response
                response["routing_decision"] = {
                    "selected_engine": selected_engine,
                    "reasoning": reasoning,
                    "decision_time_ms": routing_decision["decision_time_ms"],
                    "was_failover": False
                }
                
                self.status = EngineStatus.READY
                return response
                
            except Exception as primary_error:
                self.logger.warning(f"Primary engine {selected_engine} failed: {primary_error}")
                
                # Attempt failover if enabled
                if self.routing_config["failover_enabled"]:
                    fallback_engine = "local_llama" if selected_engine == "openai" else "openai"
                    
                    if self.engine_health[fallback_engine]["available"]:
                        try:
                            self.logger.info(f"Attempting failover to {fallback_engine}")
                            
                            response = await self._execute_on_engine(
                                fallback_engine, conversation_input, conversation_context,
                                client_config, conversation_type
                            )
                            
                            # Update failover statistics
                            self._update_routing_stats(fallback_engine, success=True, was_failover=True)
                            
                            # Add failover metadata
                            response["routing_decision"] = {
                                "selected_engine": fallback_engine,
                                "reasoning": f"Failover from {selected_engine}: {str(primary_error)}",
                                "decision_time_ms": routing_decision["decision_time_ms"],
                                "was_failover": True,
                                "primary_engine_error": str(primary_error)
                            }
                            
                            self.status = EngineStatus.READY
                            return response
                            
                        except Exception as fallback_error:
                            self.logger.error(f"Failover engine {fallback_engine} also failed: {fallback_error}")
                
                # Both engines failed - return error response
                raise Exception(f"All available engines failed. Primary: {primary_error}")
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.status = EngineStatus.READY
            
            self._update_routing_stats(selected_engine if 'selected_engine' in locals() else "unknown", 
                                     success=False)
            
            self.logger.error(f"Hybrid engine generation failed: {e}")
            
            return {
                "text": "I apologize, but I'm experiencing technical difficulties. Please try again.",
                "generation_time_ms": processing_time,
                "error": str(e),
                "engine_name": "hybrid_intelligent",
                "success": False,
                "fallback_response": True,
                "timestamp": datetime.now().isoformat(),
                "routing_decision": {
                    "selected_engine": "none",
                    "reasoning": "Engine selection failed",
                    "was_failover": False,
                    "error": str(e)
                }
            }
    
    async def _make_routing_decision(
        self,
        conversation_input: Dict[str, Any],
        conversation_context: Dict[str, Any],
        client_config: Dict[str, Any],
        conversation_type: ConversationType
    ) -> Dict[str, Any]:
        """Make intelligent routing decision based on multiple factors"""
        
        decision_start = datetime.now()
        
        # Check engine availability
        available_engines = [
            engine for engine, health in self.engine_health.items() 
            if health["available"]
        ]
        
        if not available_engines:
            raise Exception("No engines available for routing")
        
        if len(available_engines) == 1:
            decision_time = (datetime.now() - decision_start).total_seconds() * 1000
            return {
                "selected_engine": available_engines[0],
                "reasoning": "Only engine available",
                "decision_time_ms": decision_time,
                "confidence": 1.0
            }
        
        # Multi-factor decision making
        decision_factors = await self._analyze_decision_factors(
            conversation_input, conversation_context, client_config, conversation_type
        )
        
        # Apply routing logic
        selected_engine, reasoning, confidence = await self._apply_routing_logic(
            decision_factors, available_engines
        )
        
        decision_time = (datetime.now() - decision_start).total_seconds() * 1000
        
        return {
            "selected_engine": selected_engine,
            "reasoning": reasoning,
            "decision_time_ms": decision_time,
            "confidence": confidence,
            "factors": decision_factors
        }
    
    async def _analyze_decision_factors(
        self,
        conversation_input: Dict[str, Any],
        conversation_context: Dict[str, Any],
        client_config: Dict[str, Any],
        conversation_type: ConversationType
    ) -> Dict[str, Any]:
        """Analyze factors that influence routing decision"""
        
        factors = {
            "conversation_complexity": self._assess_conversation_complexity(conversation_input, conversation_context),
            "latency_requirement": self._assess_latency_requirement(conversation_type, client_config),
            "industry_specialization": self._assess_industry_specialization(client_config),
            "current_load": await self._assess_current_load(),
            "cost_sensitivity": self._assess_cost_sensitivity(client_config),
            "quality_requirement": self._assess_quality_requirement(conversation_type, client_config)
        }
        
        return factors
    
    def _assess_conversation_complexity(
        self, 
        conversation_input: Dict[str, Any], 
        conversation_context: Dict[str, Any]
    ) -> float:
        """Assess conversation complexity (0.0 = simple, 1.0 = complex)"""
        
        complexity_score = 0.0
        
        # Input length complexity
        input_text = conversation_input.get("text", "")
        if len(input_text) > 200:
            complexity_score += 0.3
        elif len(input_text) > 100:
            complexity_score += 0.2
        
        # Conversation history length
        history = conversation_context.get("conversation_history", [])
        if len(history) > 10:
            complexity_score += 0.3
        elif len(history) > 5:
            complexity_score += 0.2
        
        # Technical terms detection (simplified)
        technical_terms = ["API", "integration", "technical", "specification", "algorithm", "optimization"]
        if any(term.lower() in input_text.lower() for term in technical_terms):
            complexity_score += 0.2
        
        return min(complexity_score, 1.0)
    
    def _assess_latency_requirement(
        self, 
        conversation_type: ConversationType, 
        client_config: Dict[str, Any]
    ) -> float:
        """Assess latency requirement (0.0 = relaxed, 1.0 = critical)"""
        
        # Meta-Sales conversations need faster response
        if conversation_type == ConversationType.META_SALES_BRAIN:
            return 0.9
        
        # Real-time customer conversations need fast response
        if conversation_type == ConversationType.CLIENT_CUSTOMER:
            return 0.8
        
        # Other conversation types more relaxed
        return 0.5
    
    def _assess_industry_specialization(self, client_config: Dict[str, Any]) -> float:
        """Assess need for industry specialization (0.0 = general, 1.0 = specialized)"""
        
        industry = client_config.get("primary_industry", "general")
        
        # Industries that benefit from specialization
        specialized_industries = [
            "healthcare", "financial services", "legal services", 
            "real estate", "automotive", "technology"
        ]
        
        if industry.lower() in [ind.lower() for ind in specialized_industries]:
            return 0.8
        
        return 0.3
    
    async def _assess_current_load(self) -> Dict[str, float]:
        """Assess current load on each engine"""
        
        load_stats = {
            "openai": 0.5,  # Default moderate load
            "local_llama": 0.5
        }
        
        # Check OpenAI engine load
        if self.openai_engine and hasattr(self.openai_engine, 'rate_limit_manager'):
            rate_status = self.openai_engine.rate_limit_manager.get_status()
            openai_load = 1.0 - (rate_status["requests_available"] / rate_status["requests_limit"])
            load_stats["openai"] = max(0.0, min(1.0, openai_load))
        
        # Check Local Llama GPU usage
        if self.local_llama_engine and hasattr(self.local_llama_engine, 'gpu_manager'):
            gpu_usage = self.local_llama_engine.gpu_manager.get_memory_usage()
            load_stats["local_llama"] = gpu_usage["used_percent"] / 100.0
        
        return load_stats
    
    def _assess_cost_sensitivity(self, client_config: Dict[str, Any]) -> float:
        """Assess cost sensitivity (0.0 = cost insensitive, 1.0 = highly cost sensitive)"""
        
        # Check client tier or cost preferences
        client_tier = client_config.get("subscription_tier", "standard")
        
        if client_tier.lower() in ["basic", "economy"]:
            return 0.9  # Highly cost sensitive
        elif client_tier.lower() in ["premium", "enterprise"]:
            return 0.2  # Less cost sensitive
        
        return 0.6  # Moderate cost sensitivity
    
    def _assess_quality_requirement(
        self, 
        conversation_type: ConversationType, 
        client_config: Dict[str, Any]
    ) -> float:
        """Assess quality requirement (0.0 = basic, 1.0 = highest quality)"""
        
        # Meta-Sales requires highest quality
        if conversation_type == ConversationType.META_SALES_BRAIN:
            return 0.95
        
        # Enterprise clients expect higher quality
        client_tier = client_config.get("subscription_tier", "standard")
        if client_tier.lower() in ["premium", "enterprise"]:
            return 0.85
        
        return 0.7  # Standard quality
    
    async def _apply_routing_logic(
        self, 
        factors: Dict[str, Any], 
        available_engines: List[str]
    ) -> tuple[str, str, float]:
        """Apply routing logic based on analyzed factors"""
        
        scores = {}
        
        for engine in available_engines:
            score = 0.0
            reasoning_parts = []
            
            if engine == "local_llama":
                # Local Llama strengths
                score += (1.0 - factors["cost_sensitivity"]) * 0.3  # Cost efficiency
                if factors["cost_sensitivity"] > 0.7:
                    reasoning_parts.append("cost efficiency")
                
                # Performance advantage
                if factors["latency_requirement"] > 0.7:
                    score += 0.4
                    reasoning_parts.append("low latency")
                
                # Load consideration
                current_load = factors["current_load"].get("local_llama", 0.5)
                score += (1.0 - current_load) * 0.2
                if current_load < 0.5:
                    reasoning_parts.append("low GPU usage")
                
                # Preference for local
                if self.routing_config["prefer_local"]:
                    score += 0.1
                    reasoning_parts.append("local preference")
            
            elif engine == "openai":
                # OpenAI strengths
                score += factors["conversation_complexity"] * 0.3  # Better for complex conversations
                if factors["conversation_complexity"] > 0.6:
                    reasoning_parts.append("conversation complexity")
                
                # Quality advantage
                score += factors["quality_requirement"] * 0.3
                if factors["quality_requirement"] > 0.8:
                    reasoning_parts.append("quality requirement")
                
                # Load consideration
                current_load = factors["current_load"].get("openai", 0.5)
                score += (1.0 - current_load) * 0.2
                if current_load < 0.5:
                    reasoning_parts.append("low API usage")
                
                # Industry specialization
                score += factors["industry_specialization"] * 0.2
                if factors["industry_specialization"] > 0.7:
                    reasoning_parts.append("industry specialization")
            
            scores[engine] = {
                "score": score,
                "reasoning": ", ".join(reasoning_parts) if reasoning_parts else "default scoring"
            }
        
        # Select engine with highest score
        best_engine = max(scores.keys(), key=lambda x: scores[x]["score"])
        best_score = scores[best_engine]["score"]
        reasoning = scores[best_engine]["reasoning"]
        
        # Calculate confidence based on score difference
        score_values = [s["score"] for s in scores.values()]
        if len(score_values) > 1:
            confidence = (best_score - min(score_values)) / max(1.0, max(score_values))
        else:
            confidence = 1.0
        
        return best_engine, reasoning, confidence
    
    async def _execute_on_engine(
        self,
        engine_name: str,
        conversation_input: Dict[str, Any],
        conversation_context: Dict[str, Any],
        client_config: Dict[str, Any],
        conversation_type: ConversationType
    ) -> Dict[str, Any]:
        """Execute request on specified engine"""
        
        if engine_name == "openai" and self.openai_engine:
            return await self.openai_engine.generate_response(
                conversation_input, conversation_context, client_config, conversation_type
            )
        elif engine_name == "local_llama" and self.local_llama_engine:
            return await self.local_llama_engine.generate_response(
                conversation_input, conversation_context, client_config, conversation_type
            )
        else:
            raise Exception(f"Engine {engine_name} not available")
    
    def _update_routing_stats(self, engine_name: str, success: bool, was_failover: bool = False):
        """Update routing statistics"""
        
        self.routing_stats["total_requests"] += 1
        
        if success:
            if engine_name == "openai":
                self.routing_stats["openai_requests"] += 1
            elif engine_name == "local_llama":
                self.routing_stats["local_llama_requests"] += 1
        
        if was_failover:
            self.routing_stats["failover_events"] += 1
    
    async def _perform_health_check(self) -> None:
        """Hybrid engine health check implementation"""
        
        health_issues = []
        
        # Check individual engines
        if self.openai_engine:
            try:
                openai_healthy = await self.openai_engine.health_check()
                self.engine_health["openai"]["available"] = openai_healthy
                self.engine_health["openai"]["last_check"] = datetime.now()
            except Exception as e:
                self.engine_health["openai"]["available"] = False
                health_issues.append(f"OpenAI engine: {e}")
        
        if self.local_llama_engine:
            try:
                llama_healthy = await self.local_llama_engine.health_check()
                self.engine_health["local_llama"]["available"] = llama_healthy
                self.engine_health["local_llama"]["last_check"] = datetime.now()
            except Exception as e:
                self.engine_health["local_llama"]["available"] = False
                health_issues.append(f"Local Llama engine: {e}")
        
        # Ensure at least one engine is healthy
        if not any(engine["available"] for engine in self.engine_health.values()):
            raise Exception(f"No engines available. Issues: {'; '.join(health_issues)}")
    
    async def cleanup(self) -> None:
        """Clean up hybrid engine resources"""
        
        self.status = EngineStatus.MAINTENANCE
        
        cleanup_tasks = []
        
        if self.openai_engine:
            cleanup_tasks.append(self.openai_engine.cleanup())
        
        if self.local_llama_engine:
            cleanup_tasks.append(self.local_llama_engine.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.logger.info("Hybrid Intelligent Engine cleaned up")
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """Get comprehensive routing and performance summary"""
        
        return {
            "routing_stats": self.routing_stats,
            "engine_health": self.engine_health,
            "routing_config": self.routing_config,
            "available_engines": [
                engine for engine, health in self.engine_health.items() 
                if health["available"]
            ],
            "performance_summary": {
                "total_requests": self.routing_stats["total_requests"],
                "openai_percentage": (
                    (self.routing_stats["openai_requests"] / max(self.routing_stats["total_requests"], 1)) * 100
                ),
                "local_llama_percentage": (
                    (self.routing_stats["local_llama_requests"] / max(self.routing_stats["total_requests"], 1)) * 100
                ),
                "failover_rate": (
                    (self.routing_stats["failover_events"] / max(self.routing_stats["total_requests"], 1)) * 100
                )
            }
        }


class IntelligentRoutingEngine:
    """Advanced routing intelligence with machine learning capabilities"""
    
    def __init__(self):
        self.routing_history = []
        self.performance_patterns = {}
        self.logger = logging.getLogger("IntelligentRoutingEngine")
    
    def record_routing_outcome(
        self, 
        decision_factors: Dict[str, Any], 
        selected_engine: str, 
        performance_metrics: Dict[str, Any]
    ):
        """Record routing decision outcome for learning"""
        
        outcome = {
            "timestamp": datetime.now(),
            "factors": decision_factors,
            "selected_engine": selected_engine,
            "performance": performance_metrics,
            "success": performance_metrics.get("success", False)
        }
        
        self.routing_history.append(outcome)
        
        # Keep only recent history (last 1000 decisions)
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def get_routing_recommendations(self, current_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Get intelligent routing recommendations based on historical patterns"""
        
        # Simplified pattern matching - in production this would use ML
        recommendations = {
            "recommended_engine": "local_llama",  # Default preference
            "confidence": 0.5,
            "reasoning": "Default recommendation"
        }
        
        # Analyze recent successful patterns
        recent_successes = [
            outcome for outcome in self.routing_history[-100:]
            if outcome["success"] and outcome["performance"].get("generation_time_ms", 0) < 1000
        ]
        
        if recent_successes:
            # Find most successful engine for similar contexts
            engine_success_rates = {}
            for outcome in recent_successes:
                engine = outcome["selected_engine"]
                if engine not in engine_success_rates:
                    engine_success_rates[engine] = {"successes": 0, "total": 0}
                engine_success_rates[engine]["successes"] += 1
                engine_success_rates[engine]["total"] += 1
            
            # Calculate success rates
            best_engine = None
            best_rate = 0
            for engine, stats in engine_success_rates.items():
                rate = stats["successes"] / stats["total"]
                if rate > best_rate:
                    best_rate = rate
                    best_engine = engine
            
            if best_engine and best_rate > 0.7:
                recommendations["recommended_engine"] = best_engine
                recommendations["confidence"] = best_rate
                recommendations["reasoning"] = f"Historical success rate: {best_rate:.1%}"
        
        return recommendations
