#!/usr/bin/env python3
"""
qudi MCP Server

Model Context Protocol server for qudi-iqo-modules integration with Claude Code/Desktop.
Enables LLM-driven control of quantum photonics experiments through qudi's modular architecture.

Based on instrMCP pattern from https://github.com/caidish/instrMCP
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# MCP imports - using mcp package structure
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    # Fallback for when MCP is not installed
    print("Warning: MCP package not found. Install with: pip install mcp")
    MCP_AVAILABLE = False

# Safety and logging
from .safety import RunLevel, SafetyChecker
from .tools.instrument_tools import InstrumentTools
from .tools.measurement_tools import MeasurementTools  
from .tools.safety_tools import SafetyTools


class QudiMCPServer:
    """Main qudi MCP Server class"""
    
    def __init__(self):
        self.server = Server("qudi-mcp")
        self.logger = self._setup_logging()
        self.runlevel = RunLevel.DRY_RUN
        self.safety_checker = SafetyChecker()
        
        # Tool modules
        self.instrument_tools = InstrumentTools(self)
        self.measurement_tools = MeasurementTools(self) 
        self.safety_tools = SafetyTools(self)
        
        # State
        self.qudi_station = None
        self.instruments = {}
        self.measurement_state = {}
        
        self._register_handlers()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the MCP server"""
        logger = logging.getLogger("qudi-mcp")
        logger.setLevel(logging.INFO)
        
        # Create handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _register_handlers(self):
        """Register MCP handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List all available MCP tools"""
            tools = []
            
            # Station management tools
            tools.extend([
                Tool(
                    name="station.info",
                    description="Get qudi station configuration and status",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="station.load_config",
                    description="Load qudi station configuration",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "config_path": {
                                "type": "string",
                                "description": "Path to qudi configuration file"
                            }
                        },
                        "required": ["config_path"]
                    }
                )
            ])
            
            # Instrument control tools
            tools.extend([
                Tool(
                    name="instrument.list",
                    description="List available instruments in qudi station",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="instrument.load",
                    description="Load and initialize an instrument",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "instrument_name": {
                                "type": "string", 
                                "description": "Name of instrument to load"
                            }
                        },
                        "required": ["instrument_name"]
                    }
                ),
                Tool(
                    name="instrument.get_parameters", 
                    description="Get instrument parameters and current values",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "instrument_name": {
                                "type": "string",
                                "description": "Name of instrument"
                            }
                        },
                        "required": ["instrument_name"]
                    }
                )
            ])
            
            # Measurement tools
            tools.extend([
                Tool(
                    name="measurement.list_modules",
                    description="List available measurement modules",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="measurement.start",
                    description="Start a measurement with specified parameters",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "Name of measurement module"
                            },
                            "parameters": {
                                "type": "object", 
                                "description": "Measurement parameters"
                            }
                        },
                        "required": ["module_name", "parameters"]
                    }
                ),
                Tool(
                    name="measurement.status",
                    description="Get status of running measurements",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "measurement_id": {
                                "type": "string",
                                "description": "Optional measurement ID to check specific measurement"
                            }
                        }
                    }
                )
            ])
            
            # Safety and control tools
            tools.extend([
                Tool(
                    name="safety.check_interlocks",
                    description="Check all safety interlocks and system status", 
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="safety.set_runlevel",
                    description="Set system runlevel (dry-run, sim, live)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "runlevel": {
                                "type": "string",
                                "enum": ["dry-run", "sim", "live"],
                                "description": "Target runlevel"
                            }
                        },
                        "required": ["runlevel"]
                    }
                ),
                Tool(
                    name="system.emergency_stop",
                    description="Emergency stop all operations and measurements",
                    inputSchema={
                        "type": "object", 
                        "properties": {},
                        "required": []
                    }
                )
            ])
            
            return tools
            
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Optional[Dict[str, Any]]
        ) -> CallToolResult:
            """Handle tool calls"""
            
            try:
                self.logger.info(f"Tool called: {name} with args: {arguments}")
                
                # Route to appropriate tool module
                if name.startswith("station."):
                    result = await self._route_station_tool(name, arguments or {})
                elif name.startswith("instrument."):
                    result = await self.instrument_tools.handle_tool(name, arguments or {})
                elif name.startswith("measurement."):
                    result = await self.measurement_tools.handle_tool(name, arguments or {})
                elif name.startswith("safety.") or name.startswith("system."):
                    result = await self.safety_tools.handle_tool(name, arguments or {})
                else:
                    result = {"error": f"Unknown tool: {name}"}
                    
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
            except Exception as e:
                self.logger.error(f"Tool {name} failed: {e}")
                error_result = {"error": f"Tool execution failed: {str(e)}"}
                return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def _route_station_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route station management tools"""
        
        if name == "station.info":
            return await self._get_station_info()
        elif name == "station.load_config":
            return await self._load_station_config(arguments["config_path"])
        else:
            return {"error": f"Unknown station tool: {name}"}
            
    async def _get_station_info(self) -> Dict[str, Any]:
        """Get station information and status"""
        
        if self.runlevel == RunLevel.DRY_RUN:
            return {
                "status": "dry_run_mode",
                "runlevel": self.runlevel.value,
                "station_loaded": bool(self.qudi_station),
                "instruments_loaded": len(self.instruments),
                "active_measurements": len(self.measurement_state),
                "message": "Running in dry-run mode - no hardware interaction"
            }
        
        # TODO: Implement real station info when qudi is available
        return {
            "status": "simulated", 
            "runlevel": self.runlevel.value,
            "station_loaded": False,
            "instruments_loaded": 0,
            "active_measurements": 0,
            "message": "qudi station not yet integrated"
        }
        
    async def _load_station_config(self, config_path: str) -> Dict[str, Any]:
        """Load qudi station configuration"""
        
        if not os.path.exists(config_path):
            return {"error": f"Configuration file not found: {config_path}"}
            
        if self.runlevel == RunLevel.DRY_RUN:
            return {
                "status": "success",
                "message": f"Dry-run: Would load config from {config_path}",
                "runlevel": self.runlevel.value
            }
            
        # TODO: Implement real config loading
        return {
            "status": "simulated",
            "message": f"Simulated loading of {config_path}",
            "config_path": config_path
        }


# Standalone server runner
async def main():
    """Run the qudi MCP server"""
    if not MCP_AVAILABLE:
        print("ERROR: MCP package not available. Install with: pip install mcp")
        return
        
    server_instance = QudiMCPServer()
    
    # Run server with stdio transport for Claude Desktop
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream, 
            write_stream,
            server_instance.server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())