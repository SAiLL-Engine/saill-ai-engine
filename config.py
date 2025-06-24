"""
Configuration Management System
Handles loading, validation, and management of SAiLL AI Engine configuration
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

from engines.manager import EngineConfigValidator


class ConfigurationError(Exception):
    """Configuration-related errors"""
    pass


def load_configuration(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file and environment variables"""
    
    # Start with default configuration
    config = get_default_configuration()
    
    # Load from configuration file if provided
    if config_file:
        file_config = load_config_file(config_file)
        config = merge_configurations(config, file_config)
    
    # Override with environment variables
    env_config = load_environment_configuration()
    config = merge_configurations(config, env_config)
    
    return config


def get_default_configuration() -> Dict[str, Any]:
    """Get default configuration values"""
    
    return {
        "system": {
            "environment": "development",
            "debug_mode": False,
            "log_level": "INFO",
            "health_check_interval": 300,
            "enable_performance_monitoring": True,
            "enable_prometheus_metrics": False
        },
        "engines": {
            "default_engine": "hybrid",
            "failover_enabled": True,
            "prefer_local_engine": True,
            "openai": {
                "enabled": True,
                "model_name": "gpt-4o",
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": True,
                "rate_limit_rpm": 500,
                "rate_limit_tpm": 30000
            },
            "local_llama": {
                "enabled": True,
                "model_path": "meta-llama/Llama-3.1-8B-Instruct",
                "max_gpu_memory": 14,
                "target_memory_usage": 0.85,
                "max_new_tokens": 150,
                "temperature": 0.7,
                "context_length": 4096
            },
            "hybrid": {
                "enabled": True,
                "routing_config": {
                    "prefer_local": True,
                    "failover_enabled": True,
                    "load_balancing": False,
                    "quality_threshold": 4.0,
                    "latency_threshold": 1000
                },
                "openai_config": {},  # Will inherit from engines.openai
                "llama_config": {}    # Will inherit from engines.local_llama
            }
        },
        "database": {
            "url": "postgresql://localhost:5432/saill_platform",
            "pool_size": 20,
            "max_overflow": 30,
            "echo": False
        },
        "redis": {
            "url": "redis://localhost:6379/0",
            "max_connections": 50,
            "socket_timeout": 30,
            "socket_connect_timeout": 30
        },
        "industry_intelligence": {
            "type": "mock",
            "enable_caching": True,
            "cache_ttl_minutes": 30,
            "timeout_seconds": 30
        },
        "performance": {
            "target_response_time_ms": 1000,
            "openai_target_response_time_ms": 2000,
            "llama_target_response_time_ms": 350,
            "enable_conversation_analytics": True,
            "enable_cost_tracking": True,
            "enable_quality_monitoring": True
        },
        "security": {
            "api_key_header": "X-API-Key",
            "allowed_origins": ["http://localhost:3000"],
            "enable_cors": True,
            "enable_conversation_encryption": False
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "worker_class": "uvicorn.workers.UvicornWorker",
            "worker_connections": 1000,
            "max_requests": 10000,
            "max_requests_jitter": 1000
        }
    }


def load_config_file(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {config_path.suffix}")
    
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration file {config_file}: {e}")


def load_environment_configuration() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    
    config = {"engines": {"openai": {}, "local_llama": {}, "hybrid": {}}}
    
    # System configuration
    if os.getenv("ENVIRONMENT"):
        config.setdefault("system", {})["environment"] = os.getenv("ENVIRONMENT")
    if os.getenv("DEBUG_MODE"):
        config.setdefault("system", {})["debug_mode"] = os.getenv("DEBUG_MODE").lower() == "true"
    if os.getenv("LOG_LEVEL"):
        config.setdefault("system", {})["log_level"] = os.getenv("LOG_LEVEL")
    
    # Engine configuration
    if os.getenv("DEFAULT_ENGINE"):
        config["engines"]["default_engine"] = os.getenv("DEFAULT_ENGINE")
    if os.getenv("ENABLE_FAILOVER"):
        config["engines"]["failover_enabled"] = os.getenv("ENABLE_FAILOVER").lower() == "true"
    if os.getenv("PREFER_LOCAL_ENGINE"):
        config["engines"]["prefer_local_engine"] = os.getenv("PREFER_LOCAL_ENGINE").lower() == "true"
    
    # OpenAI configuration
    openai_config = config["engines"]["openai"]
    if os.getenv("OPENAI_API_KEY"):
        openai_config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_MODEL"):
        openai_config["model_name"] = os.getenv("OPENAI_MODEL")
    if os.getenv("OPENAI_MAX_TOKENS"):
        openai_config["max_tokens"] = int(os.getenv("OPENAI_MAX_TOKENS"))
    if os.getenv("OPENAI_TEMPERATURE"):
        openai_config["temperature"] = float(os.getenv("OPENAI_TEMPERATURE"))
    if os.getenv("OPENAI_RATE_LIMIT_RPM"):
        openai_config["rate_limit_rpm"] = int(os.getenv("OPENAI_RATE_LIMIT_RPM"))
    if os.getenv("OPENAI_RATE_LIMIT_TPM"):
        openai_config["rate_limit_tpm"] = int(os.getenv("OPENAI_RATE_LIMIT_TPM"))
    
    # Local Llama configuration
    llama_config = config["engines"]["local_llama"]
    if os.getenv("LLAMA_MODEL_PATH"):
        llama_config["model_path"] = os.getenv("LLAMA_MODEL_PATH")
    if os.getenv("LLAMA_MAX_GPU_MEMORY"):
        llama_config["max_gpu_memory"] = int(os.getenv("LLAMA_MAX_GPU_MEMORY"))
    if os.getenv("LLAMA_TARGET_MEMORY_USAGE"):
        llama_config["target_memory_usage"] = float(os.getenv("LLAMA_TARGET_MEMORY_USAGE"))
    if os.getenv("LLAMA_MAX_NEW_TOKENS"):
        llama_config["max_new_tokens"] = int(os.getenv("LLAMA_MAX_NEW_TOKENS"))
    if os.getenv("LLAMA_TEMPERATURE"):
        llama_config["temperature"] = float(os.getenv("LLAMA_TEMPERATURE"))
    if os.getenv("LLAMA_CONTEXT_LENGTH"):
        llama_config["context_length"] = int(os.getenv("LLAMA_CONTEXT_LENGTH"))
    
    # Database configuration
    if os.getenv("DATABASE_URL"):
        config.setdefault("database", {})["url"] = os.getenv("DATABASE_URL")
    if os.getenv("DATABASE_POOL_SIZE"):
        config.setdefault("database", {})["pool_size"] = int(os.getenv("DATABASE_POOL_SIZE"))
    if os.getenv("DATABASE_MAX_OVERFLOW"):
        config.setdefault("database", {})["max_overflow"] = int(os.getenv("DATABASE_MAX_OVERFLOW"))
    
    # Redis configuration
    if os.getenv("REDIS_URL"):
        config.setdefault("redis", {})["url"] = os.getenv("REDIS_URL")
    if os.getenv("REDIS_MAX_CONNECTIONS"):
        config.setdefault("redis", {})["max_connections"] = int(os.getenv("REDIS_MAX_CONNECTIONS"))
    
    # Industry intelligence configuration
    if os.getenv("INDUSTRY_INTELLIGENCE_TYPE"):
        config.setdefault("industry_intelligence", {})["type"] = os.getenv("INDUSTRY_INTELLIGENCE_TYPE")
    if os.getenv("INDUSTRY_CACHE_TTL_MINUTES"):
        config.setdefault("industry_intelligence", {})["cache_ttl_minutes"] = int(os.getenv("INDUSTRY_CACHE_TTL_MINUTES"))
    if os.getenv("INDUSTRY_INTELLIGENCE_TIMEOUT"):
        config.setdefault("industry_intelligence", {})["timeout_seconds"] = int(os.getenv("INDUSTRY_INTELLIGENCE_TIMEOUT"))
    
    # Performance configuration
    if os.getenv("TARGET_RESPONSE_TIME_MS"):
        config.setdefault("performance", {})["target_response_time_ms"] = int(os.getenv("TARGET_RESPONSE_TIME_MS"))
    if os.getenv("OPENAI_TARGET_RESPONSE_TIME_MS"):
        config.setdefault("performance", {})["openai_target_response_time_ms"] = int(os.getenv("OPENAI_TARGET_RESPONSE_TIME_MS"))
    if os.getenv("LLAMA_TARGET_RESPONSE_TIME_MS"):
        config.setdefault("performance", {})["llama_target_response_time_ms"] = int(os.getenv("LLAMA_TARGET_RESPONSE_TIME_MS"))
    
    # Server configuration
    if os.getenv("HOST"):
        config.setdefault("server", {})["host"] = os.getenv("HOST")
    if os.getenv("PORT"):
        config.setdefault("server", {})["port"] = int(os.getenv("PORT"))
    if os.getenv("WORKERS"):
        config.setdefault("server", {})["workers"] = int(os.getenv("WORKERS"))
    
    return config


def merge_configurations(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries recursively"""
    
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configurations(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_configuration(config: Dict[str, Any]) -> List[str]:
    """Validate complete configuration and return list of errors"""
    
    errors = []
    
    try:
        # Validate system configuration
        system_errors = validate_system_configuration(config.get("system", {}))
        errors.extend(system_errors)
        
        # Validate engine configurations
        engines_config = config.get("engines", {})
        
        # Validate individual engine configurations
        engine_validation_results = EngineConfigValidator.validate_all_configs({
            "openai": engines_config.get("openai", {}),
            "local_llama": engines_config.get("local_llama", {}),
            "hybrid": engines_config.get("hybrid", {})
        })
        
        for engine_name, engine_errors in engine_validation_results.items():
            for error in engine_errors:
                errors.append(f"{engine_name}: {error}")
        
        # Validate database configuration
        database_errors = validate_database_configuration(config.get("database", {}))
        errors.extend(database_errors)
        
        # Validate performance configuration
        performance_errors = validate_performance_configuration(config.get("performance", {}))
        errors.extend(performance_errors)
        
        # Validate cross-system dependencies
        dependency_errors = validate_system_dependencies(config)
        errors.extend(dependency_errors)
        
    except Exception as e:
        errors.append(f"Configuration validation failed: {e}")
    
    return errors


def validate_system_configuration(system_config: Dict[str, Any]) -> List[str]:
    """Validate system configuration"""
    
    errors = []
    
    # Validate log level
    log_level = system_config.get("log_level", "INFO")
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_log_levels:
        errors.append(f"Invalid log level: {log_level}. Must be one of: {valid_log_levels}")
    
    # Validate environment
    environment = system_config.get("environment", "development")
    valid_environments = ["development", "testing", "staging", "production"]
    if environment not in valid_environments:
        errors.append(f"Invalid environment: {environment}. Must be one of: {valid_environments}")
    
    # Validate health check interval
    health_check_interval = system_config.get("health_check_interval", 300)
    if not isinstance(health_check_interval, int) or health_check_interval < 30:
        errors.append("health_check_interval must be an integer >= 30 seconds")
    
    return errors


def validate_database_configuration(database_config: Dict[str, Any]) -> List[str]:
    """Validate database configuration"""
    
    errors = []
    
    # Validate database URL
    database_url = database_config.get("url")
    if not database_url:
        errors.append("Database URL is required")
    elif not database_url.startswith(("postgresql://", "postgres://")):
        errors.append("Database URL must be a PostgreSQL connection string")
    
    # Validate pool settings
    pool_size = database_config.get("pool_size", 20)
    if not isinstance(pool_size, int) or pool_size < 1:
        errors.append("Database pool_size must be a positive integer")
    
    max_overflow = database_config.get("max_overflow", 30)
    if not isinstance(max_overflow, int) or max_overflow < 0:
        errors.append("Database max_overflow must be a non-negative integer")
    
    return errors


def validate_performance_configuration(performance_config: Dict[str, Any]) -> List[str]:
    """Validate performance configuration"""
    
    errors = []
    
    # Validate response time targets
    target_response_time = performance_config.get("target_response_time_ms", 1000)
    if not isinstance(target_response_time, int) or target_response_time < 100:
        errors.append("target_response_time_ms must be >= 100ms")
    
    openai_target = performance_config.get("openai_target_response_time_ms", 2000)
    if not isinstance(openai_target, int) or openai_target < 500:
        errors.append("openai_target_response_time_ms must be >= 500ms")
    
    llama_target = performance_config.get("llama_target_response_time_ms", 350)
    if not isinstance(llama_target, int) or llama_target < 100:
        errors.append("llama_target_response_time_ms must be >= 100ms")
    
    return errors


def validate_system_dependencies(config: Dict[str, Any]) -> List[str]:
    """Validate dependencies between system components"""
    
    errors = []
    
    # Check if default engine is enabled
    engines_config = config.get("engines", {})
    default_engine = engines_config.get("default_engine", "hybrid")
    
    if default_engine not in ["openai", "local_llama", "hybrid"]:
        errors.append(f"Invalid default_engine: {default_engine}")
    else:
        engine_config = engines_config.get(default_engine, {})
        if not engine_config.get("enabled", True):
            errors.append(f"Default engine '{default_engine}' is disabled")
    
    # Check hybrid engine dependencies
    if default_engine == "hybrid" or engines_config.get("hybrid", {}).get("enabled", True):
        hybrid_config = engines_config.get("hybrid", {})
        openai_config = engines_config.get("openai", {})
        llama_config = engines_config.get("local_llama", {})
        
        if not openai_config.get("enabled", True) and not llama_config.get("enabled", True):
            errors.append("Hybrid engine requires at least one of OpenAI or Local Llama to be enabled")
    
    # Check industry intelligence configuration
    industry_config = config.get("industry_intelligence", {})
    intelligence_type = industry_config.get("type", "mock")
    if intelligence_type not in ["mock", "production"]:
        errors.append(f"Invalid industry_intelligence type: {intelligence_type}")
    
    return errors


def save_configuration(config: Dict[str, Any], config_file: str, format: str = "yaml") -> None:
    """Save configuration to file"""
    
    config_path = Path(config_file)
    
    try:
        with open(config_path, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(config, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported format: {format}")
    
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration to {config_file}: {e}")


def get_engine_config(config: Dict[str, Any], engine_name: str) -> Dict[str, Any]:
    """Get configuration for a specific engine with inheritance"""
    
    engines_config = config.get("engines", {})
    engine_config = engines_config.get(engine_name, {}).copy()
    
    # For hybrid engine, inherit from base engine configurations
    if engine_name == "hybrid":
        # Inherit OpenAI configuration
        openai_base = engines_config.get("openai", {})
        openai_override = engine_config.get("openai_config", {})
        engine_config["openai_config"] = merge_configurations(openai_base, openai_override)
        
        # Inherit Llama configuration  
        llama_base = engines_config.get("local_llama", {})
        llama_override = engine_config.get("llama_config", {})
        engine_config["llama_config"] = merge_configurations(llama_base, llama_override)
    
    return engine_config


# Export main functions
__all__ = [
    'load_configuration',
    'get_default_configuration',
    'validate_configuration',
    'save_configuration',
    'get_engine_config',
    'ConfigurationError'
]
