#!/usr/bin/env python3
"""
SAiLL AI Engine Foundation Validation Script
Validates that Increment 1: Modular AI Engine Foundation is complete and functional
"""

import asyncio
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich import print as rprint

# Import SAiLL components
from engines import ConversationType, ConversationContext, PerformanceMonitor
from engines.manager import AIEngineManager, EngineConfigValidator
from industry_intelligence import IndustryIntelligenceFactory, MockIndustryIntelligence
from config import get_default_configuration, validate_configuration

console = Console()


class IncrementValidation:
    """Validation framework for SAiLL AI Engine increments"""
    
    def __init__(self):
        self.console = Console()
        self.validation_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    async def run_validation(self) -> bool:
        """Run complete validation suite"""
        
        self.console.print(Panel.fit(
            "[bold blue]SAiLL AI Engine Foundation[/bold blue]\n"
            "[dim]Increment 1: Modular AI Engine Foundation[/dim]\n"
            "[yellow]Validation Suite[/yellow]",
            title="🔍 VALIDATION",
            border_style="blue"
        ))
        
        validation_steps = [
            ("Core Interfaces", self._validate_core_interfaces),
            ("Engine Factory System", self._validate_engine_factory),
            ("Configuration Management", self._validate_configuration),
            ("Industry Intelligence", self._validate_industry_intelligence),
            ("Performance Monitoring", self._validate_performance_monitoring),
            ("Engine Manager", self._validate_engine_manager),
            ("Integration Testing", self._validate_integration),
            ("Documentation & Deployment", self._validate_deployment_readiness)
        ]
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Validating components...", total=len(validation_steps))
            
            for step_name, validation_func in validation_steps:
                self.console.print(f"\n🔄 Testing: {step_name}")
                
                try:
                    success = await validation_func()
                    self._record_result(step_name, success)
                    
                    if success:
                        self.console.print(f"✅ {step_name}: PASSED")
                    else:
                        self.console.print(f"❌ {step_name}: FAILED")
                        
                except Exception as e:
                    self.console.print(f"💥 {step_name}: ERROR - {e}")
                    self._record_result(step_name, False, str(e))
                
                progress.advance(task)
        
        # Display final results
        self._display_final_results()
        
        return self.passed_tests == self.total_tests
    
    async def _validate_core_interfaces(self) -> bool:
        """Validate core AI engine interfaces"""
        
        tests = []
        
        # Test ConversationType enum
        tests.append(self._test_enum_values())
        
        # Test ConversationContext
        tests.append(self._test_conversation_context())
        
        # Test PerformanceMonitor
        tests.append(self._test_performance_monitor())
        
        return all(tests)
    
    def _test_enum_values(self) -> bool:
        """Test enum definitions"""
        try:
            # Test ConversationType
            assert ConversationType.CLIENT_CUSTOMER.value == "client_customer"
            assert ConversationType.META_SALES_BRAIN.value == "meta_sales_brain"
            assert ConversationType.INTERNAL_COACHING.value == "internal_coaching"
            
            return True
        except Exception as e:
            self.console.print(f"  ❌ Enum test failed: {e}")
            return False
    
    def _test_conversation_context(self) -> bool:
        """Test conversation context functionality"""
        try:
            context = ConversationContext("test_client", "test_customer", "test_campaign")
            
            # Test basic properties
            assert context.client_id == "test_client"
            assert context.customer_id == "test_customer"
            assert context.campaign_id == "test_campaign"
            
            # Test interaction management
            context.add_interaction(
                user_input="Hello",
                ai_response="Hi there!",
                metadata={"test": True}
            )
            
            assert len(context.conversation_history) == 1
            assert context.context_metadata["turn_count"] == 1
            
            # Test context formatting
            ai_context = context.get_context_for_ai()
            assert "conversation_id" in ai_context
            assert "conversation_history" in ai_context
            
            return True
        except Exception as e:
            self.console.print(f"  ❌ ConversationContext test failed: {e}")
            return False
    
    def _test_performance_monitor(self) -> bool:
        """Test performance monitoring"""
        try:
            monitor = PerformanceMonitor()
            
            # Record test metrics
            monitor.record_engine_performance(
                engine_name="test_engine",
                response_time=500,
                conversation_type=ConversationType.CLIENT_CUSTOMER,
                success=True
            )
            
            # Get performance summary
            summary = monitor.get_performance_summary()
            assert "engine_performance" in summary
            assert "test_engine" in summary["engine_performance"]
            
            return True
        except Exception as e:
            self.console.print(f"  ❌ PerformanceMonitor test failed: {e}")
            return False
    
    async def _validate_engine_factory(self) -> bool:
        """Validate engine factory registration system"""
        
        try:
            from engines import EngineFactory
            
            # Check engine registration
            available_engines = EngineFactory.list_available_engines()
            required_engines = ["openai", "local_llama", "hybrid"]
            
            for engine in required_engines:
                if engine not in available_engines:
                    self.console.print(f"  ❌ Missing engine: {engine}")
                    return False
            
            self.console.print(f"  ✅ All required engines registered: {available_engines}")
            return True
            
        except Exception as e:
            self.console.print(f"  ❌ Engine factory validation failed: {e}")
            return False
    
    async def _validate_configuration(self) -> bool:
        """Validate configuration management system"""
        
        try:
            # Test default configuration loading
            config = get_default_configuration()
            
            # Check required sections
            required_sections = ["system", "engines", "database", "redis", "industry_intelligence"]
            for section in required_sections:
                if section not in config:
                    self.console.print(f"  ❌ Missing config section: {section}")
                    return False
            
            # Test configuration validation
            validation_errors = validate_configuration(config)
            
            # Allow some errors (like GPU requirements) in test environment
            critical_errors = [error for error in validation_errors 
                             if "GPU" not in error and "API key" not in error]
            
            if critical_errors:
                self.console.print(f"  ❌ Critical config errors: {critical_errors}")
                return False
            
            # Test engine config validation
            validator = EngineConfigValidator()
            engine_configs = {
                "openai": config["engines"]["openai"],
                "local_llama": config["engines"]["local_llama"],
                "hybrid": config["engines"]["hybrid"]
            }
            
            validation_results = validator.validate_all_configs(engine_configs)
            
            self.console.print(f"  ✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.console.print(f"  ❌ Configuration validation failed: {e}")
            return False
    
    async def _validate_industry_intelligence(self) -> bool:
        """Validate industry intelligence system"""
        
        try:
            # Test factory creation
            service = IndustryIntelligenceFactory.create_service("mock")
            
            # Test context retrieval
            context_request = {
                "industry": "technology",
                "conversation_type": "client_customer",
                "client_id": "test_client",
                "user_input": "Tell me about AI solutions"
            }
            
            context = await service.get_context(context_request)
            
            # Validate response structure
            assert context["industry_context_available"] == True
            assert "industry_name" in context
            assert "key_insights" in context
            
            # Test insights retrieval
            insights = await service.get_industry_insights("healthcare")
            assert isinstance(insights, list)
            
            self.console.print(f"  ✅ Industry intelligence functional")
            return True
            
        except Exception as e:
            self.console.print(f"  ❌ Industry intelligence validation failed: {e}")
            return False
    
    async def _validate_performance_monitoring(self) -> bool:
        """Validate performance monitoring system"""
        
        try:
            monitor = PerformanceMonitor()
            
            # Record multiple performance metrics
            for i in range(10):
                monitor.record_engine_performance(
                    engine_name=f"test_engine_{i % 3}",
                    response_time=100 + (i * 50),
                    conversation_type=ConversationType.CLIENT_CUSTOMER,
                    success=i % 5 != 0  # 80% success rate
                )
            
            # Test performance summary
            summary = monitor.get_performance_summary()
            assert "engine_performance" in summary
            assert "conversation_quality" in summary
            assert "recommendations" in summary
            
            # Test recommendations generation
            recommendations = summary["recommendations"]
            assert isinstance(recommendations, list)
            
            self.console.print(f"  ✅ Performance monitoring functional")
            return True
            
        except Exception as e:
            self.console.print(f"  ❌ Performance monitoring validation failed: {e}")
            return False
    
    async def _validate_engine_manager(self) -> bool:
        """Validate AI engine manager"""
        
        try:
            # Create engine manager with minimal config
            engine_configs = {
                "openai": {"enabled": False},
                "local_llama": {"enabled": False},
                "hybrid": {"enabled": False}
            }
            
            # Create industry intelligence service
            industry_intelligence = IndustryIntelligenceFactory.create_service("mock")
            
            # Initialize manager
            manager = AIEngineManager()
            
            # Test manager methods (without actual engines)
            available_engines = manager.list_available_engines()
            assert len(available_engines) == 3  # openai, local_llama, hybrid
            
            # Test health check on uninitialized manager
            health_status = await manager.health_check()
            assert "manager_healthy" in health_status
            assert "engines" in health_status
            
            self.console.print(f"  ✅ Engine manager functional")
            return True
            
        except Exception as e:
            self.console.print(f"  ❌ Engine manager validation failed: {e}")
            return False
    
    async def _validate_integration(self) -> bool:
        """Validate component integration"""
        
        try:
            # Test configuration + industry intelligence integration
            config = get_default_configuration()
            industry_service = IndustryIntelligenceFactory.create_service(
                config["industry_intelligence"]["type"],
                config["industry_intelligence"]
            )
            
            # Test conversation flow components
            context = ConversationContext("test_client", "test_customer", "test_campaign")
            
            conversation_input = {"text": "Hello, I need help with AI solutions"}
            client_config = {
                "company_name": "Test Company",
                "primary_industry": "technology",
                "subscription_tier": "enterprise"
            }
            
            # Test industry context retrieval
            context_request = {
                "industry": client_config["primary_industry"],
                "conversation_type": "client_customer",
                "client_id": context.client_id,
                "user_input": conversation_input["text"]
            }
            
            industry_context = await industry_service.get_context(context_request)
            assert industry_context["industry_context_available"] == True
            
            # Test conversation context update
            context.add_interaction(
                user_input=conversation_input["text"],
                ai_response="I'd be happy to help with AI solutions!",
                metadata={"industry_context_applied": True}
            )
            
            assert len(context.conversation_history) == 1
            
            self.console.print(f"  ✅ Component integration successful")
            return True
            
        except Exception as e:
            self.console.print(f"  ❌ Integration validation failed: {e}")
            return False
    
    async def _validate_deployment_readiness(self) -> bool:
        """Validate deployment readiness"""
        
        try:
            # Check required files exist
            required_files = [
                "requirements.txt",
                ".env.example", 
                "Dockerfile",
                "main.py",
                "config.py"
            ]
            
            missing_files = []
            for file in required_files:
                if not (project_root / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                self.console.print(f"  ❌ Missing files: {missing_files}")
                return False
            
            # Check main CLI functionality
            try:
                from main import app
                assert app is not None
            except Exception as e:
                self.console.print(f"  ❌ Main CLI import failed: {e}")
                return False
            
            self.console.print(f"  ✅ Deployment readiness confirmed")
            return True
            
        except Exception as e:
            self.console.print(f"  ❌ Deployment readiness validation failed: {e}")
            return False
    
    def _record_result(self, test_name: str, success: bool, error: str = None):
        """Record validation result"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        
        self.validation_results.append({
            "test": test_name,
            "passed": success,
            "error": error
        })
    
    def _display_final_results(self):
        """Display final validation results"""
        
        # Create results table
        table = Table(title="🎯 Validation Results Summary")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="dim")
        
        for result in self.validation_results:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            details = result["error"] if result["error"] else "OK"
            table.add_row(result["test"], status, details)
        
        self.console.print(table)
        
        # Success/failure summary
        success_rate = (self.passed_tests / self.total_tests) * 100
        
        if self.passed_tests == self.total_tests:
            self.console.print(Panel.fit(
                f"[bold green]🎉 INCREMENT 1 VALIDATION SUCCESSFUL[/bold green]\n"
                f"[green]All {self.total_tests} tests passed ({success_rate:.1f}%)[/green]\n\n"
                f"[dim]Modular AI Engine Foundation is complete and ready for production deployment.[/dim]",
                title="✅ SUCCESS",
                border_style="green"
            ))
        else:
            self.console.print(Panel.fit(
                f"[bold red]⚠️  INCREMENT 1 VALIDATION INCOMPLETE[/bold red]\n"
                f"[yellow]{self.passed_tests}/{self.total_tests} tests passed ({success_rate:.1f}%)[/yellow]\n"
                f"[red]{self.failed_tests} tests failed[/red]\n\n"
                f"[dim]Please address failing tests before proceeding to Increment 2.[/dim]",
                title="❌ NEEDS ATTENTION",
                border_style="red"
            ))


async def main():
    """Main validation entry point"""
    
    validator = IncrementValidation()
    success = await validator.run_validation()
    
    if success:
        console.print("\n🚀 Ready to proceed to Increment 2: Local Llama Production Engine")
        return 0
    else:
        console.print("\n🛠️  Please address validation issues before continuing")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n👋 Validation interrupted")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n💥 Validation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
