"""
SAiLL AI Engine - Main Application Entry Point
Provides CLI interface and application startup for the AI engine system
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from engines.manager import AIEngineManager, create_ai_engine_manager, EngineConfigValidator
from engines import ConversationType, ConversationContext
from industry_intelligence import IndustryIntelligenceFactory
from config import load_configuration, validate_configuration

# Initialize CLI app and console
app = typer.Typer(help="SAiLL AI Engine - Modular Intelligence System")
console = Console()


@app.command()
def start(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    env_file: Optional[str] = typer.Option(".env", "--env", "-e", help="Environment file path"),
    engine: Optional[str] = typer.Option(None, "--engine", help="Preferred engine (openai, local_llama, hybrid)"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port")
):
    """Start the SAiLL AI Engine server"""
    
    rprint(Panel.fit(
        "[bold blue]SAiLL AI Engine[/bold blue]\n"
        "[dim]Modular Intelligence System[/dim]",
        title="🚀 Starting",
        border_style="blue"
    ))
    
    try:
        # Load environment variables
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
            console.print(f"✅ Loaded environment from: {env_file}")
        
        # Load configuration
        config = load_configuration(config_file)
        
        # Override with CLI parameters
        if engine:
            config["engines"]["default_engine"] = engine
        if debug:
            config["system"]["debug_mode"] = True
            config["system"]["log_level"] = "DEBUG"
        
        # Validate configuration
        validation_errors = validate_configuration(config)
        if validation_errors:
            console.print("❌ Configuration validation failed:")
            for error in validation_errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)
        
        # Start the server
        asyncio.run(start_server(config, port))
        
    except KeyboardInterrupt:
        console.print("\n👋 Shutting down SAiLL AI Engine...")
    except Exception as e:
        console.print(f"❌ Failed to start: {e}")
        raise typer.Exit(1)


@app.command()
def test_engines(
    env_file: Optional[str] = typer.Option(".env", "--env", "-e", help="Environment file path"),
    engine: Optional[str] = typer.Option(None, "--engine", help="Test specific engine")
):
    """Test AI engine functionality"""
    
    rprint(Panel.fit(
        "[bold green]Engine Testing[/bold green]\n"
        "[dim]Testing AI engine functionality[/dim]",
        title="🧪 Testing",
        border_style="green"
    ))
    
    try:
        # Load environment
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
        
        # Run engine tests
        asyncio.run(run_engine_tests(engine))
        
    except Exception as e:
        console.print(f"❌ Testing failed: {e}")
        raise typer.Exit(1)


@app.command()
def validate_config(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    env_file: Optional[str] = typer.Option(".env", "--env", "-e", help="Environment file path")
):
    """Validate engine configuration"""
    
    rprint(Panel.fit(
        "[bold yellow]Configuration Validation[/bold yellow]\n"
        "[dim]Checking configuration and environment[/dim]",
        title="✅ Validating",
        border_style="yellow"
    ))
    
    try:
        # Load environment
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
            console.print(f"✅ Loaded environment from: {env_file}")
        
        # Load and validate configuration
        config = load_configuration(config_file)
        validation_errors = validate_configuration(config)
        
        if validation_errors:
            console.print("❌ Configuration validation failed:")
            for error in validation_errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)
        else:
            console.print("✅ Configuration validation passed")
            
            # Show configuration summary
            show_config_summary(config)
        
    except Exception as e:
        console.print(f"❌ Validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def benchmark(
    env_file: Optional[str] = typer.Option(".env", "--env", "-e", help="Environment file path"),
    engine: Optional[str] = typer.Option(None, "--engine", help="Benchmark specific engine"),
    requests: int = typer.Option(10, "--requests", "-n", help="Number of test requests"),
    concurrent: int = typer.Option(1, "--concurrent", "-c", help="Concurrent requests")
):
    """Benchmark AI engine performance"""
    
    rprint(Panel.fit(
        "[bold magenta]Performance Benchmarking[/bold magenta]\n"
        "[dim]Testing response times and throughput[/dim]",
        title="⚡ Benchmarking",
        border_style="magenta"
    ))
    
    try:
        # Load environment
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
        
        # Run benchmarks
        asyncio.run(run_benchmarks(engine, requests, concurrent))
        
    except Exception as e:
        console.print(f"❌ Benchmarking failed: {e}")
        raise typer.Exit(1)


async def start_server(config: Dict[str, Any], port: int):
    """Start the AI engine server"""
    
    # Setup logging
    setup_logging(config.get("system", {}).get("log_level", "INFO"))
    
    logger = logging.getLogger("SAiLLAIEngine")
    logger.info("Starting SAiLL AI Engine server...")
    
    try:
        # Initialize industry intelligence
        industry_intelligence = IndustryIntelligenceFactory.create_service(
            service_type=config.get("industry_intelligence", {}).get("type", "mock"),
            config=config.get("industry_intelligence", {})
        )
        
        # Initialize AI engine manager
        engine_manager = await create_ai_engine_manager(
            engine_configs=config.get("engines", {}),
            industry_intelligence_service=industry_intelligence
        )
        
        # Health check
        health_status = await engine_manager.health_check()
        console.print("🏥 Engine Health Status:")
        show_health_status(health_status)
        
        if not health_status["manager_healthy"]:
            console.print("❌ Engine manager unhealthy - cannot start server")
            return
        
        # Show server info
        show_server_info(config, port)
        
        # Start HTTP server (placeholder - would integrate with FastAPI in production)
        console.print(f"🚀 Server starting on port {port}...")
        console.print("📱 Available endpoints:")
        console.print("  • GET  /health - Health check")
        console.print("  • POST /generate - Generate AI response")
        console.print("  • GET  /status - Engine status")
        console.print("  • GET  /metrics - Performance metrics")
        
        # Keep server running
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise


async def run_engine_tests(engine_filter: Optional[str] = None):
    """Run comprehensive engine tests"""
    
    # Load configuration
    config = load_configuration()
    
    # Initialize industry intelligence
    industry_intelligence = IndustryIntelligenceFactory.create_service()
    
    # Initialize engine manager
    engine_manager = await create_ai_engine_manager(
        engine_configs=config.get("engines", {}),
        industry_intelligence_service=industry_intelligence
    )
    
    # Get available engines
    available_engines = engine_manager.list_available_engines()
    
    # Filter engines if specified
    if engine_filter:
        available_engines = [e for e in available_engines if e["name"] == engine_filter]
    
    console.print(f"🧪 Testing {len(available_engines)} engines...")
    
    # Test each engine
    for engine_info in available_engines:
        engine_name = engine_info["name"]
        
        if not engine_info["initialized"]:
            console.print(f"⚠️  Skipping {engine_name} (not initialized)")
            continue
        
        console.print(f"\n🔄 Testing {engine_name}...")
        
        try:
            # Create test conversation
            conversation_context = ConversationContext(
                client_id="test_client",
                customer_id="test_customer", 
                campaign_id="test_campaign"
            )
            
            # Test conversation input
            conversation_input = {
                "text": "Hello, I'm interested in your AI solutions. Can you tell me more?"
            }
            
            # Test client configuration
            client_config = {
                "company_name": "Test Company",
                "primary_industry": "technology",
                "products_services": "AI solutions",
                "subscription_tier": "enterprise"
            }
            
            # Generate response
            start_time = asyncio.get_event_loop().time()
            response = await engine_manager.generate_response(
                conversation_input=conversation_input,
                conversation_context=conversation_context.get_context_for_ai(),
                client_config=client_config,
                conversation_type=ConversationType.CLIENT_CUSTOMER,
                preferred_engine=engine_name
            )
            end_time = asyncio.get_event_loop().time()
            
            # Show results
            response_time = (end_time - start_time) * 1000
            
            console.print(f"✅ {engine_name} test passed")
            console.print(f"   Response time: {response_time:.1f}ms")
            console.print(f"   Engine used: {response.get('engine_name', 'unknown')}")
            console.print(f"   Response: {response.get('text', '')[:100]}...")
            
        except Exception as e:
            console.print(f"❌ {engine_name} test failed: {e}")
    
    # Cleanup
    await engine_manager.cleanup()


async def run_benchmarks(engine_filter: Optional[str], num_requests: int, concurrent: int):
    """Run performance benchmarks"""
    
    console.print(f"⚡ Running {num_requests} requests with {concurrent} concurrent...")
    
    # Load configuration and initialize
    config = load_configuration()
    industry_intelligence = IndustryIntelligenceFactory.create_service()
    engine_manager = await create_ai_engine_manager(
        engine_configs=config.get("engines", {}),
        industry_intelligence_service=industry_intelligence
    )
    
    # Prepare test data
    conversation_context = ConversationContext(
        client_id="benchmark_client",
        customer_id="benchmark_customer",
        campaign_id="benchmark_campaign"
    )
    
    conversation_input = {"text": "What are the benefits of AI automation for businesses?"}
    client_config = {
        "company_name": "Benchmark Company",
        "primary_industry": "technology",
        "subscription_tier": "enterprise"
    }
    
    # Run benchmark
    results = []
    semaphore = asyncio.Semaphore(concurrent)
    
    async def make_request():
        async with semaphore:
            start_time = asyncio.get_event_loop().time()
            try:
                response = await engine_manager.generate_response(
                    conversation_input=conversation_input,
                    conversation_context=conversation_context.get_context_for_ai(),
                    client_config=client_config,
                    preferred_engine=engine_filter
                )
                end_time = asyncio.get_event_loop().time()
                return {
                    "success": True,
                    "response_time": (end_time - start_time) * 1000,
                    "engine": response.get("engine_name", "unknown")
                }
            except Exception as e:
                end_time = asyncio.get_event_loop().time()
                return {
                    "success": False,
                    "response_time": (end_time - start_time) * 1000,
                    "error": str(e)
                }
    
    # Execute requests
    tasks = [make_request() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    # Analyze results
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    if successful_results:
        response_times = [r["response_time"] for r in successful_results]
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        # Show results table
        table = Table(title="📊 Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Requests", str(num_requests))
        table.add_row("Successful", str(len(successful_results)))
        table.add_row("Failed", str(len(failed_results)))
        table.add_row("Success Rate", f"{len(successful_results)/num_requests*100:.1f}%")
        table.add_row("Average Response Time", f"{avg_time:.1f}ms")
        table.add_row("Min Response Time", f"{min_time:.1f}ms")
        table.add_row("Max Response Time", f"{max_time:.1f}ms")
        
        console.print(table)
    
    # Cleanup
    await engine_manager.cleanup()


def show_config_summary(config: Dict[str, Any]):
    """Display configuration summary"""
    
    table = Table(title="📋 Configuration Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    
    # Engine configuration
    engines = config.get("engines", {})
    table.add_row("Default Engine", engines.get("default_engine", "not set"))
    table.add_row("Failover Enabled", str(engines.get("failover_enabled", False)))
    
    # System configuration
    system = config.get("system", {})
    table.add_row("Log Level", system.get("log_level", "INFO"))
    table.add_row("Debug Mode", str(system.get("debug_mode", False)))
    
    console.print(table)


def show_health_status(health_status: Dict[str, Any]):
    """Display engine health status"""
    
    table = Table(title="🏥 Engine Health Status")
    table.add_column("Engine", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="dim")
    
    for engine_name, engine_health in health_status.get("engines", {}).items():
        status = "✅ Healthy" if engine_health.get("healthy", False) else "❌ Unhealthy"
        details = engine_health.get("error", "OK")
        table.add_row(engine_name, status, details)
    
    console.print(table)


def show_server_info(config: Dict[str, Any], port: int):
    """Display server information"""
    
    table = Table(title="🚀 Server Information")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Port", str(port))
    table.add_row("Environment", config.get("system", {}).get("environment", "development"))
    table.add_row("Debug Mode", str(config.get("system", {}).get("debug_mode", False)))
    
    console.print(table)


def setup_logging(log_level: str = "INFO"):
    """Setup application logging"""
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("saill_ai_engine.log")
        ]
    )


if __name__ == "__main__":
    app()
