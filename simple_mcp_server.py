#!/usr/bin/env python3
"""
Simple MCP Server for qudi integration - Claude Desktop compatible

This version works without the full MCP package by implementing a minimal
stdio transport that Claude Desktop can understand.
"""

import json
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_integration.safety import RunLevel, SafetyChecker
from mcp_integration.tools.instrument_tools import InstrumentTools
from mcp_integration.tools.measurement_tools import MeasurementTools
from mcp_integration.tools.safety_tools import SafetyTools


class SimpleMCPServer:
    """Minimal MCP server that works with Claude Desktop"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.runlevel = RunLevel.DRY_RUN
        self.safety_checker = SafetyChecker()
        self.instruments = {}
        self.measurement_state = {}
        
        # Tool modules
        self.instrument_tools = InstrumentTools(self)
        self.measurement_tools = MeasurementTools(self)
        self.safety_tools = SafetyTools(self)
        
    def _setup_logging(self):
        """Setup logging for debugging"""
        logger = logging.getLogger("qudi-mcp-simple")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter('%(asctime)s - qudi-mcp - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def get_tools(self):
        """Return list of available tools"""
        return [
            {
                "name": "station_info",
                "description": "Get qudi station configuration and status",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "instrument_list",
                "description": "List available instruments in qudi station",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "instrument_load",
                "description": "Load and initialize an instrument",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instrument_name": {"type": "string", "description": "Name of instrument to load"}
                    },
                    "required": ["instrument_name"]
                }
            },
            {
                "name": "measurement_list_modules",
                "description": "List available measurement modules",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "measurement_start", 
                "description": "Start a measurement with specified parameters",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "module_name": {"type": "string", "description": "Name of measurement module"},
                        "parameters": {"type": "object", "description": "Measurement parameters"}
                    },
                    "required": ["module_name", "parameters"]
                }
            },
            {
                "name": "safety_check_interlocks",
                "description": "Check all safety interlocks and system status",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "safety_validate_parameter",
                "description": "Validate a parameter value against safety limits", 
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "parameter": {"type": "string", "description": "Parameter name"},
                        "value": {"type": "number", "description": "Parameter value"}
                    },
                    "required": ["parameter", "value"]
                }
            },
            {
                "name": "safety_set_runlevel",
                "description": "Set system runlevel (dry-run, sim, live)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "runlevel": {"type": "string", "enum": ["dry-run", "sim", "live"]},
                        "reason": {"type": "string", "description": "Reason for change"}
                    },
                    "required": ["runlevel"]
                }
            },
            {
                "name": "system_emergency_stop",
                "description": "Emergency stop all operations and measurements",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Reason for emergency stop"}
                    }
                }
            },
            {
                "name": "feedback_submit",
                "description": "Submit feedback about qudi MCP integration - usage suggestions, issues, improvements",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "feedback_type": {"type": "string", "enum": ["bug_report", "feature_request", "usage_improvement", "general"], "description": "Type of feedback"},
                        "message": {"type": "string", "description": "Your feedback message"},
                        "user_context": {"type": "string", "description": "Optional context about what you were trying to do"}
                    },
                    "required": ["feedback_type", "message"]
                }
            }
        ]
        
    async def call_tool(self, name: str, arguments: dict):
        """Call a tool and return results"""
        
        self.logger.info(f"Tool called: {name} with args: {arguments}")
        
        try:
            # Route to appropriate tool module
            if name.startswith("station_"):
                result = await self._route_station_tool(name, arguments)
            elif name.startswith("instrument_"):
                # Convert instrument_list -> instrument.list format
                converted_name = name.replace("instrument_", "instrument.")
                result = await self.instrument_tools.handle_tool(converted_name, arguments)
            elif name.startswith("measurement_"):
                # Convert measurement_list_modules -> measurement.list_modules format
                converted_name = name.replace("measurement_", "measurement.")
                result = await self.measurement_tools.handle_tool(converted_name, arguments)
            elif name.startswith("safety_") or name.startswith("system_"):
                # Convert safety_check_interlocks -> safety.check_interlocks format
                if name.startswith("safety_"):
                    converted_name = name.replace("safety_", "safety.")
                elif name.startswith("system_"):
                    converted_name = name.replace("system_", "system.")
                else:
                    converted_name = name
                result = await self.safety_tools.handle_tool(converted_name, arguments)
            elif name.startswith("feedback_"):
                result = await self._route_feedback_tool(name, arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}
                
            self.logger.info(f"Tool result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Tool {name} failed: {e}", exc_info=True)
            return {"error": f"Tool execution failed: {str(e)}"}
    
    async def _route_station_tool(self, name: str, arguments: dict):
        """Route station management tools"""
        
        if name == "station_info":
            return {
                "status": "operational",
                "runlevel": self.runlevel.value,
                "station_loaded": True,
                "instruments_loaded": len(self.instruments),
                "active_measurements": len(self.measurement_state),
                "safety_status": "active",
                "message": f"qudi MCP integration running in {self.runlevel.value} mode"
            }
        else:
            return {"error": f"Unknown station tool: {name}"}
    
    async def _route_feedback_tool(self, name: str, arguments: dict):
        """Route feedback tools"""
        
        if name == "feedback_submit":
            feedback_type = arguments.get("feedback_type", "general")
            message = arguments.get("message", "")
            user_context = arguments.get("user_context", "")
            
            # Log feedback for analysis
            self.logger.info(f"FEEDBACK RECEIVED - Type: {feedback_type}")
            self.logger.info(f"FEEDBACK MESSAGE: {message}")
            if user_context:
                self.logger.info(f"FEEDBACK CONTEXT: {user_context}")
            
            # TODO: In production, this would create a GitHub issue
            # For now, just acknowledge receipt
            return {
                "status": "received",
                "message": "Thank you for your feedback! It has been logged for review.",
                "feedback_id": f"qudi-mcp-{hash(message) % 10000:04d}",
                "next_steps": "Your feedback will be reviewed and may result in GitHub issues or improvements to the qudi MCP integration."
            }
        else:
            return {"error": f"Unknown feedback tool: {name}"}
    
    async def handle_message(self, message):
        """Handle incoming MCP messages"""
        
        try:
            # Get ID, default to 0 if None
            msg_id = message.get("id", 0)
            method = message.get("method", "unknown")
            
            # Log all incoming messages for user interaction tracking
            self.logger.info(f"USER INTERACTION - Method: {method}, ID: {msg_id}")
            if method == "tools/call":
                params = message.get("params", {})
                tool_name = params.get("name", "unknown")
                self.logger.info(f"USER TOOL CALL - Tool: {tool_name}, Args: {params.get('arguments', {})}")
            
            # Parse incoming message
            if message.get("method") == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": self.get_tools()
                    }
                }
            elif message.get("method") == "tools/call":
                params = message.get("params", {})
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                result = await self.call_tool(tool_name, tool_args)
                
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    }
                }
            elif message.get("method") == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "qudi-mcp",
                            "version": "0.1.0"
                        }
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {message.get('method')}"
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Message handling failed: {e}", exc_info=True)
            msg_id = message.get("id", 0) if message else 0
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def run(self):
        """Run the MCP server on stdio"""
        
        self.logger.info("Starting qudi MCP server on stdio")
        print("qudi MCP server running on stdio", file=sys.stderr)
        
        # Read from stdin and write to stdout
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                # Parse JSON message
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON: {e}")
                    continue
                
                # Handle message
                response = await self.handle_message(message)
                
                # Send response
                if response:
                    print(json.dumps(response), flush=True)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Server error: {e}", exc_info=True)
                
        self.logger.info("qudi MCP server shutting down")


async def main():
    """Main entry point"""
    server = SimpleMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())