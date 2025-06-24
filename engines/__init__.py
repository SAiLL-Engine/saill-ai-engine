"""
SAiLL AI Engine - Modular Intelligence System
Core engine interfaces and base classes for swappable AI implementations
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import logging

class ConversationType(Enum):
    """Define conversation types for different use cases"""
    CLIENT_CUSTOMER = "client_customer"
    META_SALES_BRAIN = "meta_sales_brain"
    INTERNAL_COACHING = "internal_coaching"
    INDUSTRY_RESEARCH = "industry_research"

class EngineStatus(Enum):
    """Engine operational status"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class AIEngineInterface(ABC):
    """Base interface for all AI engine implementations"""
    
    def __init__(self):
        self.status = EngineStatus.INITIALIZING
        self.logger = logging.getLogger(self.__class__.__name__)
        self.performance_stats = {
            "total_requests": 0,
            "total_processing_time": 0,
            "average_response_time": 0,
            "error_count": 0,
            "last_request_time": None
        }
    
    @abstractmethod
    async def initialize(
        self, 
        engine_config: Dict[str, Any],
        industry_intelligence_service: Any,
        performance_monitor: Any
    ) -> None:
        """Initialize the AI engine with configuration and dependencies"""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        conversation_input: Dict[str, Any],
        conversation_context: Dict[str, Any],
        client_config: Dict[str, Any],
        conversation_type: ConversationType = ConversationType.CLIENT_CUSTOMER
    ) -> Dict[str, Any]:
        """Generate AI response with industry context and performance tracking"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources and connections"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status and performance metrics"""
        return {
            "status": self.status.value,
            "performance_stats": self.performance_stats,
            "last_health_check": datetime.now().isoformat()
        }
    
    async def health_check(self) -> bool:
        """Perform engine health check"""
        try:
            # Basic connectivity and resource check
            await self._perform_health_check()
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    @abstractmethod
    async def _perform_health_check(self) -> None:
        """Engine-specific health check implementation"""
        pass
    
    def update_performance_stats(self, processing_time: float, error_occurred: bool = False):
        """Update internal performance tracking"""
        self.performance_stats["total_requests"] += 1
        self.performance_stats["last_request_time"] = datetime.now().isoformat()
        
        if error_occurred:
            self.performance_stats["error_count"] += 1
        else:
            self.performance_stats["total_processing_time"] += processing_time
            self.performance_stats["average_response_time"] = (
                self.performance_stats["total_processing_time"] / 
                (self.performance_stats["total_requests"] - self.performance_stats["error_count"])
            )

class EngineFactory:
    """Factory for creating and managing AI engine instances"""
    
    _engines = {}
    
    @classmethod
    def register_engine(cls, engine_name: str, engine_class: type):
        """Register an AI engine implementation"""
        cls._engines[engine_name] = engine_class
    
    @classmethod
    async def create_engine(
        cls, 
        engine_name: str, 
        engine_config: Dict[str, Any],
        industry_intelligence_service: Any,
        performance_monitor: Any
    ) -> AIEngineInterface:
        """Create and initialize an AI engine instance"""
        
        if engine_name not in cls._engines:
            raise ValueError(f"Unknown engine: {engine_name}. Available: {list(cls._engines.keys())}")
        
        engine_class = cls._engines[engine_name]
        engine_instance = engine_class()
        
        await engine_instance.initialize(
            engine_config, 
            industry_intelligence_service, 
            performance_monitor
        )
        
        return engine_instance
    
    @classmethod
    def list_available_engines(cls) -> List[str]:
        """Get list of registered engine implementations"""
        return list(cls._engines.keys())

class ConversationContext:
    """Manage conversation context and state across interactions"""
    
    def __init__(self, client_id: str, customer_id: str, campaign_id: str):
        self.client_id = client_id
        self.customer_id = customer_id  
        self.campaign_id = campaign_id
        self.conversation_id = f"{client_id}_{customer_id}_{int(datetime.now().timestamp())}"
        
        self.conversation_history = []
        self.context_metadata = {
            "started_at": datetime.now().isoformat(),
            "last_interaction": None,
            "turn_count": 0,
            "sentiment_score": 0.0,
            "engagement_level": "unknown"
        }
        
        self.industry_context = {}
        self.client_specific_context = {}
    
    def add_interaction(self, user_input: str, ai_response: str, metadata: Dict[str, Any] = None):
        """Add interaction to conversation history"""
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "turn": self.context_metadata["turn_count"] + 1,
            "user_input": user_input,
            "ai_response": ai_response,
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(interaction)
        self.context_metadata["turn_count"] += 1
        self.context_metadata["last_interaction"] = datetime.now().isoformat()
    
    def get_context_for_ai(self, max_history: int = 5) -> Dict[str, Any]:
        """Get formatted context for AI processing"""
        
        recent_history = self.conversation_history[-max_history:] if max_history else self.conversation_history
        
        return {
            "conversation_id": self.conversation_id,
            "client_id": self.client_id,
            "customer_id": self.customer_id,
            "campaign_id": self.campaign_id,
            "conversation_metadata": self.context_metadata,
            "conversation_history": recent_history,
            "industry_context": self.industry_context,
            "client_context": self.client_specific_context
        }
    
    def update_industry_context(self, industry_data: Dict[str, Any]):
        """Update industry-specific context"""
        self.industry_context.update(industry_data)
    
    def update_client_context(self, client_data: Dict[str, Any]):
        """Update client-specific context"""
        self.client_specific_context.update(client_data)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Generate conversation summary for analytics"""
        return {
            "conversation_id": self.conversation_id,
            "duration_minutes": self._calculate_duration(),
            "total_turns": self.context_metadata["turn_count"],
            "final_sentiment": self.context_metadata.get("sentiment_score", 0.0),
            "engagement_level": self.context_metadata.get("engagement_level", "unknown"),
            "topics_discussed": self._extract_topics(),
            "outcome": self.context_metadata.get("conversation_outcome", "ongoing")
        }
    
    def _calculate_duration(self) -> float:
        """Calculate conversation duration in minutes"""
        if not self.context_metadata["last_interaction"]:
            return 0.0
        
        start_time = datetime.fromisoformat(self.context_metadata["started_at"])
        end_time = datetime.fromisoformat(self.context_metadata["last_interaction"])
        
        return (end_time - start_time).total_seconds() / 60.0
    
    def _extract_topics(self) -> List[str]:
        """Extract key topics from conversation history"""
        # Simplified topic extraction - in production this would use NLP
        topics = set()
        for interaction in self.conversation_history:
            # Basic keyword extraction
            user_words = interaction["user_input"].lower().split()
            ai_words = interaction["ai_response"].lower().split()
            
            for word in user_words + ai_words:
                if len(word) > 5 and word.isalpha():  # Simple filter for meaningful words
                    topics.add(word)
        
        return list(topics)[:10]  # Return top 10 topics

# Performance monitoring and analytics
class PerformanceMonitor:
    """Monitor AI engine performance and conversation quality"""
    
    def __init__(self):
        self.metrics = {
            "engine_performance": {},
            "conversation_quality": {},
            "system_health": {},
            "error_tracking": {}
        }
        self.logger = logging.getLogger("PerformanceMonitor")
    
    def record_engine_performance(
        self, 
        engine_name: str, 
        response_time: float, 
        conversation_type: ConversationType,
        success: bool = True
    ):
        """Record engine performance metrics"""
        
        if engine_name not in self.metrics["engine_performance"]:
            self.metrics["engine_performance"][engine_name] = {
                "total_requests": 0,
                "total_response_time": 0,
                "average_response_time": 0,
                "success_rate": 0,
                "by_conversation_type": {}
            }
        
        engine_stats = self.metrics["engine_performance"][engine_name]
        engine_stats["total_requests"] += 1
        
        if success:
            engine_stats["total_response_time"] += response_time
            engine_stats["average_response_time"] = (
                engine_stats["total_response_time"] / engine_stats["total_requests"]
            )
        
        # Track by conversation type
        conv_type = conversation_type.value
        if conv_type not in engine_stats["by_conversation_type"]:
            engine_stats["by_conversation_type"][conv_type] = {
                "requests": 0,
                "avg_response_time": 0,
                "total_time": 0
            }
        
        type_stats = engine_stats["by_conversation_type"][conv_type]
        type_stats["requests"] += 1
        if success:
            type_stats["total_time"] += response_time
            type_stats["avg_response_time"] = type_stats["total_time"] / type_stats["requests"]
    
    def record_conversation_quality(
        self, 
        conversation_id: str, 
        quality_score: float,
        sentiment_score: float,
        engagement_metrics: Dict[str, Any]
    ):
        """Record conversation quality metrics"""
        
        self.metrics["conversation_quality"][conversation_id] = {
            "quality_score": quality_score,
            "sentiment_score": sentiment_score,
            "engagement_metrics": engagement_metrics,
            "recorded_at": datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "summary_generated_at": datetime.now().isoformat(),
            "engine_performance": self.metrics["engine_performance"],
            "conversation_quality": self._calculate_quality_averages(),
            "system_health": self.metrics["system_health"],
            "recommendations": self._generate_recommendations()
        }
    
    def _calculate_quality_averages(self) -> Dict[str, float]:
        """Calculate average conversation quality metrics"""
        quality_scores = [
            conv["quality_score"] 
            for conv in self.metrics["conversation_quality"].values()
        ]
        
        sentiment_scores = [
            conv["sentiment_score"] 
            for conv in self.metrics["conversation_quality"].values()
        ]
        
        if not quality_scores:
            return {"average_quality": 0.0, "average_sentiment": 0.0}
        
        return {
            "average_quality": sum(quality_scores) / len(quality_scores),
            "average_sentiment": sum(sentiment_scores) / len(sentiment_scores),
            "total_conversations": len(quality_scores)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Analyze engine performance
        for engine_name, stats in self.metrics["engine_performance"].items():
            avg_time = stats.get("average_response_time", 0)
            
            if avg_time > 2000:  # 2 seconds
                recommendations.append(
                    f"Engine {engine_name} response time ({avg_time:.0f}ms) exceeds target. Consider optimization."
                )
            elif avg_time > 1000:  # 1 second
                recommendations.append(
                    f"Engine {engine_name} response time ({avg_time:.0f}ms) approaching limit. Monitor closely."
                )
        
        # Analyze conversation quality
        quality_avg = self._calculate_quality_averages().get("average_quality", 0)
        if quality_avg < 4.0:
            recommendations.append(
                f"Average conversation quality ({quality_avg:.1f}/5.0) below target. Review conversation flows."
            )
        
        if not recommendations:
            recommendations.append("All systems performing within acceptable parameters.")
        
        return recommendations

# Export main classes
__all__ = [
    'AIEngineInterface',
    'ConversationType', 
    'EngineStatus',
    'EngineFactory',
    'ConversationContext',
    'PerformanceMonitor'
]
