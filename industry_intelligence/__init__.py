"""
Industry Intelligence Integration Interface
Provides real-time industry context and knowledge enhancement for AI conversations
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


class IndustryIntelligenceInterface(ABC):
    """Base interface for industry intelligence services"""
    
    @abstractmethod
    async def get_context(self, context_request: Dict[str, Any]) -> Dict[str, Any]:
        """Get industry-specific context for conversation enhancement"""
        pass
    
    @abstractmethod
    async def get_industry_insights(self, industry: str) -> List[Dict[str, Any]]:
        """Get key insights for a specific industry"""
        pass
    
    @abstractmethod
    async def search_industry_knowledge(self, query: str, industry: str) -> List[Dict[str, Any]]:
        """Search industry-specific knowledge base"""
        pass


class MockIndustryIntelligence(IndustryIntelligenceInterface):
    """Mock implementation of industry intelligence for development and testing"""
    
    def __init__(self):
        self.logger = logging.getLogger("MockIndustryIntelligence")
        self.mock_data = self._initialize_mock_data()
        
    def _initialize_mock_data(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock industry data for testing"""
        
        return {
            "healthcare": {
                "industry_name": "Healthcare & Medical Services",
                "key_insights": [
                    "Patient care quality is the top priority for healthcare decisions",
                    "Compliance with HIPAA and medical regulations is critical",
                    "Digital health solutions are rapidly expanding market",
                    "Cost reduction while maintaining quality is a key challenge"
                ],
                "common_terms": ["HIPAA", "EMR", "patient care", "medical devices", "telehealth"],
                "typical_concerns": ["compliance", "patient safety", "cost management", "technology adoption"],
                "decision_factors": ["regulatory compliance", "patient outcomes", "cost effectiveness", "integration capabilities"]
            },
            "financial_services": {
                "industry_name": "Financial Services & Banking",
                "key_insights": [
                    "Regulatory compliance and security are paramount concerns",
                    "Digital transformation is reshaping customer expectations",
                    "Fraud prevention and risk management are critical capabilities",
                    "Personalized financial services drive customer loyalty"
                ],
                "common_terms": ["compliance", "KYC", "AML", "fintech", "digital banking"],
                "typical_concerns": ["security", "regulation", "customer experience", "digital innovation"],
                "decision_factors": ["regulatory approval", "security standards", "integration complexity", "customer impact"]
            },
            "retail": {
                "industry_name": "Retail & E-commerce",
                "key_insights": [
                    "Customer experience and convenience drive purchasing decisions",
                    "Omnichannel integration is essential for competitive advantage",
                    "Data analytics and personalization increase conversion rates",
                    "Supply chain efficiency impacts profitability and customer satisfaction"
                ],
                "common_terms": ["omnichannel", "customer journey", "conversion", "inventory", "personalization"],
                "typical_concerns": ["customer retention", "inventory management", "competition", "profit margins"],
                "decision_factors": ["customer impact", "ROI", "implementation speed", "scalability"]
            },
            "technology": {
                "industry_name": "Technology & Software",
                "key_insights": [
                    "Innovation speed and technical excellence are competitive advantages",
                    "Scalability and performance are critical for growth",
                    "Developer experience and API quality impact adoption",
                    "Security and reliability are table stakes for enterprise adoption"
                ],
                "common_terms": ["scalability", "API", "cloud native", "DevOps", "microservices"],
                "typical_concerns": ["technical debt", "scalability", "security", "developer productivity"],
                "decision_factors": ["technical fit", "scalability", "performance", "integration ease"]
            },
            "manufacturing": {
                "industry_name": "Manufacturing & Industrial",
                "key_insights": [
                    "Operational efficiency and cost reduction drive technology adoption",
                    "Industry 4.0 and IoT integration are transforming operations",
                    "Quality control and compliance are non-negotiable requirements",
                    "Predictive maintenance reduces downtime and costs"
                ],
                "common_terms": ["Industry 4.0", "IoT", "predictive maintenance", "OEE", "lean manufacturing"],
                "typical_concerns": ["operational efficiency", "quality control", "cost reduction", "downtime"],
                "decision_factors": ["ROI", "operational impact", "integration complexity", "reliability"]
            }
        }
    
    async def get_context(self, context_request: Dict[str, Any]) -> Dict[str, Any]:
        """Get industry-specific context for conversation enhancement"""
        
        try:
            industry = context_request.get("industry", "general").lower()
            conversation_type = context_request.get("conversation_type", "client_customer")
            user_input = context_request.get("user_input", "")
            
            # Get base industry data
            industry_data = self.mock_data.get(industry, {})
            
            if not industry_data:
                return {"industry_context_available": False}
            
            # Build context response
            context = {
                "industry_context_available": True,
                "industry_name": industry_data.get("industry_name", industry),
                "key_insights": industry_data.get("key_insights", [])[:3],  # Top 3 insights
                "relevant_terms": self._get_relevant_terms(user_input, industry_data),
                "conversation_guidance": self._get_conversation_guidance(conversation_type, industry_data),
                "context_retrieval_time_ms": 25  # Simulated fast retrieval
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get industry context: {e}")
            return {"industry_context_available": False, "error": str(e)}
    
    async def get_industry_insights(self, industry: str) -> List[Dict[str, Any]]:
        """Get key insights for a specific industry"""
        
        industry_data = self.mock_data.get(industry.lower(), {})
        
        if not industry_data:
            return []
        
        insights = []
        for insight in industry_data.get("key_insights", []):
            insights.append({
                "insight": insight,
                "relevance_score": 0.8,
                "source": "market_research",
                "last_updated": datetime.now().isoformat()
            })
        
        return insights
    
    async def search_industry_knowledge(self, query: str, industry: str) -> List[Dict[str, Any]]:
        """Search industry-specific knowledge base"""
        
        industry_data = self.mock_data.get(industry.lower(), {})
        
        if not industry_data:
            return []
        
        # Simple keyword matching for mock implementation
        results = []
        query_lower = query.lower()
        
        # Search in insights
        for insight in industry_data.get("key_insights", []):
            if any(word in insight.lower() for word in query_lower.split()):
                results.append({
                    "type": "insight",
                    "content": insight,
                    "relevance_score": 0.7,
                    "source": "industry_analysis"
                })
        
        # Search in terms
        for term in industry_data.get("common_terms", []):
            if query_lower in term.lower() or term.lower() in query_lower:
                results.append({
                    "type": "term_definition",
                    "content": f"{term} is a key concept in {industry}",
                    "relevance_score": 0.6,
                    "source": "terminology_database"
                })
        
        return results[:5]  # Return top 5 results
    
    def _get_relevant_terms(self, user_input: str, industry_data: Dict[str, Any]) -> List[str]:
        """Get industry terms relevant to user input"""
        
        user_input_lower = user_input.lower()
        common_terms = industry_data.get("common_terms", [])
        
        relevant_terms = []
        for term in common_terms:
            if term.lower() in user_input_lower or any(word in term.lower() for word in user_input_lower.split()):
                relevant_terms.append(term)
        
        return relevant_terms[:3]  # Return top 3 relevant terms
    
    def _get_conversation_guidance(self, conversation_type: str, industry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get conversation guidance based on type and industry"""
        
        base_guidance = {
            "key_concerns": industry_data.get("typical_concerns", []),
            "decision_factors": industry_data.get("decision_factors", []),
            "communication_style": "professional"
        }
        
        if conversation_type == "meta_sales_brain":
            base_guidance.update({
                "focus_areas": ["AI automation benefits", "ROI demonstration", "competitive advantages"],
                "communication_style": "consultative",
                "key_messages": ["proven results", "industry expertise", "scalable solutions"]
            })
        
        return base_guidance


class CachedIndustryIntelligence:
    """Wrapper that adds caching to industry intelligence service"""
    
    def __init__(self, base_service: IndustryIntelligenceInterface, cache_ttl_minutes: int = 30):
        self.base_service = base_service
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("CachedIndustryIntelligence")
    
    async def get_context(self, context_request: Dict[str, Any]) -> Dict[str, Any]:
        """Get context with caching"""
        
        # Create cache key
        cache_key = self._create_cache_key("context", context_request)
        
        # Check cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            cached_result["from_cache"] = True
            return cached_result
        
        # Get from base service
        result = await self.base_service.get_context(context_request)
        
        # Cache result
        self._store_in_cache(cache_key, result)
        result["from_cache"] = False
        
        return result
    
    async def get_industry_insights(self, industry: str) -> List[Dict[str, Any]]:
        """Get insights with caching"""
        
        cache_key = self._create_cache_key("insights", {"industry": industry})
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        result = await self.base_service.get_industry_insights(industry)
        self._store_in_cache(cache_key, result)
        
        return result
    
    async def search_industry_knowledge(self, query: str, industry: str) -> List[Dict[str, Any]]:
        """Search with caching"""
        
        cache_key = self._create_cache_key("search", {"query": query, "industry": industry})
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        result = await self.base_service.search_industry_knowledge(query, industry)
        self._store_in_cache(cache_key, result)
        
        return result
    
    def _create_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Create cache key from operation and parameters"""
        
        import hashlib
        import json
        
        # Create deterministic key
        params_str = json.dumps(params, sort_keys=True)
        key_input = f"{operation}:{params_str}"
        
        return hashlib.md5(key_input.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        
        if cache_key not in self.cache:
            return None
        
        cached_item = self.cache[cache_key]
        
        # Check expiration
        if datetime.now() - cached_item["cached_at"] > self.cache_ttl:
            del self.cache[cache_key]
            return None
        
        return cached_item["data"]
    
    def _store_in_cache(self, cache_key: str, data: Any):
        """Store item in cache"""
        
        self.cache[cache_key] = {
            "data": data,
            "cached_at": datetime.now()
        }
        
        # Simple cache size management
        if len(self.cache) > 1000:
            # Remove oldest 100 items
            sorted_items = sorted(
                self.cache.items(), 
                key=lambda x: x[1]["cached_at"]
            )
            
            for key, _ in sorted_items[:100]:
                del self.cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        return {
            "cache_size": len(self.cache),
            "cache_ttl_minutes": self.cache_ttl.total_seconds() / 60,
            "cache_hit_rate": getattr(self, '_hit_rate', 0.0)  # Would track in production
        }


class IndustryIntelligenceFactory:
    """Factory for creating industry intelligence services"""
    
    @staticmethod
    def create_service(
        service_type: str = "mock", 
        config: Dict[str, Any] = None
    ) -> IndustryIntelligenceInterface:
        """Create industry intelligence service"""
        
        config = config or {}
        
        if service_type == "mock":
            base_service = MockIndustryIntelligence()
        else:
            raise ValueError(f"Unknown service type: {service_type}")
        
        # Add caching if requested
        if config.get("enable_caching", True):
            cache_ttl = config.get("cache_ttl_minutes", 30)
            return CachedIndustryIntelligence(base_service, cache_ttl)
        
        return base_service


# Production-ready placeholder for real industry intelligence
class ProductionIndustryIntelligence(IndustryIntelligenceInterface):
    """Production implementation placeholder for real industry intelligence service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ProductionIndustryIntelligence")
        
        # In production, this would connect to:
        # - APIFY data collection service
        # - Industry knowledge databases
        # - Real-time market data feeds
        # - Client-specific knowledge bases
        
    async def get_context(self, context_request: Dict[str, Any]) -> Dict[str, Any]:
        """Production implementation would query real industry databases"""
        
        # Placeholder for production implementation
        self.logger.warning("Production industry intelligence not implemented - using mock data")
        
        mock_service = MockIndustryIntelligence()
        return await mock_service.get_context(context_request)
    
    async def get_industry_insights(self, industry: str) -> List[Dict[str, Any]]:
        """Production implementation would provide real insights"""
        
        mock_service = MockIndustryIntelligence()
        return await mock_service.get_industry_insights(industry)
    
    async def search_industry_knowledge(self, query: str, industry: str) -> List[Dict[str, Any]]:
        """Production implementation would search real knowledge base"""
        
        mock_service = MockIndustryIntelligence()
        return await mock_service.search_industry_knowledge(query, industry)


# Export main classes
__all__ = [
    'IndustryIntelligenceInterface',
    'MockIndustryIntelligence',
    'CachedIndustryIntelligence',
    'IndustryIntelligenceFactory',
    'ProductionIndustryIntelligence'
]
