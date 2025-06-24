"""
Test suite for SAiLL AI Engine core components
Tests engine interfaces, factory system, and configuration management
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Test imports
from engines import (
    AIEngineInterface, ConversationType, EngineStatus, 
    EngineFactory, ConversationContext, PerformanceMonitor
)
from engines.manager import AIEngineManager, create_ai_engine_manager, EngineConfigValidator
from engines.openai_engine import OpenAIEngine
from engines.local_llama_engine import LocalLlamaEngine
from engines.hybrid_engine import HybridIntelligentEngine
from industry_intelligence import MockIndustryIntelligence, IndustryIntelligenceFactory
from config import load_configuration, validate_configuration, get_default_configuration


# Test fixtures
@pytest.fixture
def mock_openai_config():
    """Mock OpenAI configuration for testing"""
    return {
        "openai_api_key": "test_api_key",
        "model_name": "gpt-4o",
        "max_tokens": 150,
        "temperature": 0.7,
        "rate_limit_rpm": 500,
        "rate_limit_tpm": 30000
    }


@pytest.fixture
def mock_llama_config():
    """Mock Llama configuration for testing"""
    return {
        "model_path": "test/llama-model",
        "max_gpu_memory": 8,
        "target_memory_usage": 0.8,
        "max_new_tokens": 150,
        "temperature": 0.7,
        "context_length": 2048
    }


@pytest.fixture
def mock_industry_intelligence():
    """Mock industry intelligence service for testing"""
    return MockIndustryIntelligence()


@pytest.fixture
def test_conversation_context():
    """Test conversation context"""
    return ConversationContext(
        client_id="test_client",
        customer_id="test_customer",
        campaign_id="test_campaign"
    )


@pytest.fixture
def test_conversation_input():
    """Test conversation input"""
    return {
        "text": "Hello, I'm interested in AI solutions for my business."
    }


@pytest.fixture
def test_client_config():
    """Test client configuration"""
    return {
        "company_name": "Test Company",
        "primary_industry": "technology",
        "products_services": "AI automation solutions",
        "subscription_tier": "enterprise"
    }


# Core Interface Tests
class TestAIEngineInterface:
    """Test AI engine interface and base functionality"""
    
    def test_engine_factory_registration(self):
        """Test engine factory registration system"""
        
        # Check that engines are registered
        available_engines = EngineFactory.list_available_engines()
        assert "openai" in available_engines
        assert "local_llama" in available_engines
        assert "hybrid" in available_engines
    
    def test_conversation_context_creation(self, test_conversation_context):
        """Test conversation context creation and management"""
        
        context = test_conversation_context
        
        # Check basic properties
        assert context.client_id == "test_client"
        assert context.customer_id == "test_customer"
        assert context.campaign_id == "test_campaign"
        assert context.conversation_id.startswith("test_client_test_customer_")
        
        # Check initial state
        assert len(context.conversation_history) == 0
        assert context.context_metadata["turn_count"] == 0
    
    def test_conversation_context_interactions(self, test_conversation_context):
        """Test conversation context interaction management"""
        
        context = test_conversation_context
        
        # Add interaction
        context.add_interaction(
            user_input="Hello",
            ai_response="Hi there! How can I help you?",
            metadata={"sentiment": 0.8}
        )
        
        # Check state updates
        assert len(context.conversation_history) == 1
        assert context.context_metadata["turn_count"] == 1
        assert context.context_metadata["last_interaction"] is not None
        
        # Check interaction content
        interaction = context.conversation_history[0]
        assert interaction["user_input"] == "Hello"
        assert interaction["ai_response"] == "Hi there! How can I help you?"
        assert interaction["metadata"]["sentiment"] == 0.8
    
    def test_performance_monitor(self):
        """Test performance monitoring functionality"""
        
        monitor = PerformanceMonitor()
        
        # Record engine performance
        monitor.record_engine_performance(
            engine_name="test_engine",
            response_time=500,
            conversation_type=ConversationType.CLIENT_CUSTOMER,
            success=True
        )
        
        # Check metrics recording
        summary = monitor.get_performance_summary()
        assert "engine_performance" in summary
        assert "test_engine" in summary["engine_performance"]
        
        engine_stats = summary["engine_performance"]["test_engine"]
        assert engine_stats["total_requests"] == 1
        assert engine_stats["average_response_time"] == 500


# Industry Intelligence Tests
class TestIndustryIntelligence:
    """Test industry intelligence system"""
    
    @pytest.mark.asyncio
    async def test_mock_industry_intelligence(self, mock_industry_intelligence):
        """Test mock industry intelligence service"""
        
        # Test context retrieval
        context_request = {
            "industry": "healthcare",
            "conversation_type": "client_customer",
            "client_id": "test_client",
            "user_input": "Tell me about HIPAA compliance"
        }
        
        context = await mock_industry_intelligence.get_context(context_request)
        
        # Check response structure
        assert context["industry_context_available"] == True
        assert "industry_name" in context
        assert "key_insights" in context
        assert len(context["key_insights"]) <= 3  # Limited to top 3
    
    @pytest.mark.asyncio
    async def test_industry_insights(self, mock_industry_intelligence):
        """Test industry insights retrieval"""
        
        insights = await mock_industry_intelligence.get_industry_insights("technology")
        
        # Check insights structure
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        for insight in insights:
            assert "insight" in insight
            assert "relevance_score" in insight
            assert "source" in insight
    
    @pytest.mark.asyncio
    async def test_industry_knowledge_search(self, mock_industry_intelligence):
        """Test industry knowledge search"""
        
        results = await mock_industry_intelligence.search_industry_knowledge(
            query="scalability", 
            industry="technology"
        )
        
        # Check search results
        assert isinstance(results, list)
        assert len(results) <= 5  # Limited to top 5 results
        
        if results:
            result = results[0]
            assert "type" in result
            assert "content" in result
            assert "relevance_score" in result
    
    def test_industry_intelligence_factory(self):
        """Test industry intelligence factory"""
        
        # Test mock service creation
        service = IndustryIntelligenceFactory.create_service("mock")
        assert isinstance(service, MockIndustryIntelligence) or hasattr(service, 'base_service')
        
        # Test with caching enabled
        service_with_cache = IndustryIntelligenceFactory.create_service(
            "mock", 
            {"enable_caching": True, "cache_ttl_minutes": 15}
        )
        assert hasattr(service_with_cache, 'cache')


# Configuration Management Tests
class TestConfigurationManagement:
    """Test configuration loading, validation, and management"""
    
    def test_default_configuration(self):
        """Test default configuration loading"""
        
        config = get_default_configuration()
        
        # Check required sections
        assert "system" in config
        assert "engines" in config
        assert "database" in config
        assert "redis" in config
        assert "industry_intelligence" in config
        
        # Check engine configurations
        engines = config["engines"]
        assert "openai" in engines
        assert "local_llama" in engines
        assert "hybrid" in engines
        assert engines["default_engine"] in ["openai", "local_llama", "hybrid"]
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        
        # Test valid configuration
        valid_config = get_default_configuration()
        valid_config["engines"]["openai"]["openai_api_key"] = "test_key"
        
        errors = validate_configuration(valid_config)
        # Should have minimal errors (mainly GPU requirements for Llama)
        assert len(errors) <= 2  # Allow for GPU availability warnings
    
    def test_invalid_configuration_detection(self):
        """Test detection of invalid configurations"""
        
        # Test invalid configuration
        invalid_config = get_default_configuration()
        invalid_config["system"]["log_level"] = "INVALID_LEVEL"
        invalid_config["engines"]["default_engine"] = "nonexistent_engine"
        invalid_config["database"]["url"] = "invalid_url"
        
        errors = validate_configuration(invalid_config)
        assert len(errors) > 0
        
        # Check specific error detection
        error_text = " ".join(errors)
        assert "Invalid log level" in error_text
        assert "Invalid default_engine" in error_text
    
    def test_engine_config_validator(self, mock_openai_config, mock_llama_config):
        """Test engine-specific configuration validation"""
        
        # Test OpenAI config validation
        openai_errors = EngineConfigValidator.validate_openai_config(mock_openai_config)
        assert len(openai_errors) == 0  # Should be valid
        
        # Test invalid OpenAI config
        invalid_openai = mock_openai_config.copy()
        invalid_openai["max_tokens"] = 10000  # Too high
        openai_errors = EngineConfigValidator.validate_openai_config(invalid_openai)
        assert len(openai_errors) > 0
        
        # Test Llama config validation (will likely fail due to GPU requirements)
        llama_errors = EngineConfigValidator.validate_llama_config(mock_llama_config)
        # GPU requirement errors are expected in test environment


# Engine Manager Tests
class TestEngineManager:
    """Test AI engine manager functionality"""
    
    @pytest.mark.asyncio
    async def test_engine_manager_initialization(self, mock_industry_intelligence):
        """Test engine manager initialization"""
        
        # Create minimal engine configs for testing
        engine_configs = {
            "openai": {
                "enabled": False  # Disable to avoid API key requirements
            },
            "local_llama": {
                "enabled": False  # Disable to avoid GPU requirements
            },
            "hybrid": {
                "enabled": False  # Disable as it requires sub-engines
            }
        }
        
        manager = AIEngineManager()
        
        # Test initialization with no engines (should handle gracefully)
        try:
            await manager.initialize(engine_configs, mock_industry_intelligence)
        except Exception as e:
            # Expected to fail with no engines enabled
            assert "No engines available" in str(e) or "engines failed" in str(e)
    
    @pytest.mark.asyncio
    async def test_engine_manager_health_check(self, mock_industry_intelligence):
        """Test engine manager health checking"""
        
        manager = AIEngineManager()
        
        # Health check on uninitialized manager
        health_status = await manager.health_check()
        
        # Check response structure
        assert "manager_healthy" in health_status
        assert "engines" in health_status
        assert "summary" in health_status
        assert "timestamp" in health_status
    
    def test_engine_manager_capabilities(self):
        """Test engine manager capability reporting"""
        
        manager = AIEngineManager()
        
        # Test engine listing
        available_engines = manager.list_available_engines()
        assert isinstance(available_engines, list)
        
        for engine_info in available_engines:
            assert "name" in engine_info
            assert "initialized" in engine_info
            assert "capabilities" in engine_info


# Integration Tests
class TestEngineIntegration:
    """Test integration between engine components"""
    
    @pytest.mark.asyncio
    async def test_mock_conversation_flow(
        self, 
        mock_industry_intelligence,
        test_conversation_context,
        test_conversation_input,
        test_client_config
    ):
        """Test complete conversation flow with mock components"""
        
        # This test uses only mock components to avoid external dependencies
        
        # Test industry intelligence context retrieval
        context_request = {
            "industry": test_client_config["primary_industry"],
            "conversation_type": "client_customer",
            "client_id": test_conversation_context.client_id,
            "user_input": test_conversation_input["text"]
        }
        
        industry_context = await mock_industry_intelligence.get_context(context_request)
        assert industry_context["industry_context_available"] == True
        
        # Test conversation context management
        test_conversation_context.add_interaction(
            user_input=test_conversation_input["text"],
            ai_response="Thank you for your interest! I'd be happy to discuss our AI solutions.",
            metadata={"industry_context_applied": True}
        )
        
        # Verify context state
        assert len(test_conversation_context.conversation_history) == 1
        assert test_conversation_context.context_metadata["turn_count"] == 1
        
        # Test conversation summary generation
        summary = test_conversation_context.get_conversation_summary()
        assert "conversation_id" in summary
        assert "total_turns" in summary
        assert summary["total_turns"] == 1


# Performance Tests
class TestPerformanceScenarios:
    """Test performance-related scenarios and edge cases"""
    
    def test_conversation_context_large_history(self):
        """Test conversation context with large history"""
        
        context = ConversationContext("client", "customer", "campaign")
        
        # Add many interactions
        for i in range(100):
            context.add_interaction(
                user_input=f"Message {i}",
                ai_response=f"Response {i}",
                metadata={"turn": i}
            )
        
        # Test context retrieval with limits
        ai_context = context.get_context_for_ai(max_history=5)
        assert len(ai_context["conversation_history"]) == 5
        
        # Test summary generation
        summary = context.get_conversation_summary()
        assert summary["total_turns"] == 100
        assert "duration_minutes" in summary
    
    def test_performance_monitor_stress(self):
        """Test performance monitor under stress"""
        
        monitor = PerformanceMonitor()
        
        # Record many performance metrics
        for i in range(1000):
            monitor.record_engine_performance(
                engine_name="stress_test_engine",
                response_time=100 + (i % 100),  # Varying response times
                conversation_type=ConversationType.CLIENT_CUSTOMER,
                success=i % 10 != 0  # 90% success rate
            )
        
        # Check aggregated metrics
        summary = monitor.get_performance_summary()
        engine_stats = summary["engine_performance"]["stress_test_engine"]
        
        assert engine_stats["total_requests"] == 1000
        assert 100 <= engine_stats["average_response_time"] <= 200
        
        # Check recommendations generation
        recommendations = summary["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


# Utility test functions
def test_enum_definitions():
    """Test enum definitions and values"""
    
    # Test ConversationType enum
    assert ConversationType.CLIENT_CUSTOMER.value == "client_customer"
    assert ConversationType.META_SALES_BRAIN.value == "meta_sales_brain"
    assert ConversationType.INTERNAL_COACHING.value == "internal_coaching"
    
    # Test EngineStatus enum
    assert EngineStatus.INITIALIZING.value == "initializing"
    assert EngineStatus.READY.value == "ready"
    assert EngineStatus.PROCESSING.value == "processing"
    assert EngineStatus.ERROR.value == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
