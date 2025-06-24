# Test Module Initialization
"""
SAiLL AI Engine Test Suite
Comprehensive testing for all AI engine components
"""

# Test configuration
import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure pytest for async testing
pytest_plugins = ['pytest_asyncio']

def pytest_configure(config):
    """Configure pytest settings"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as asyncio test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "openai: mark test as requiring OpenAI API"
    )

# Test fixtures available to all tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Export test utilities
__all__ = [
    'pytest_configure'
]
