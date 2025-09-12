"""
Safety and system control tools for qudi MCP integration

Handles safety interlocks, emergency controls, and system monitoring
for quantum photonics experiments.
"""

import logging
from typing import Dict, List, Any, Optional
from ..safety import RunLevel


class SafetyTools:
    """Safety and emergency control tools"""
    
    def __init__(self, server):
        self.server = server
        self.logger = logging.getLogger("qudi-mcp.safety")
        
    async def handle_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route safety tool calls"""
        
        tool_map = {
            "safety.check_interlocks": self._check_interlocks,
            "safety.set_runlevel": self._set_runlevel,
            "safety.get_status": self._get_safety_status,
            "safety.validate_parameter": self._validate_parameter,
            "system.emergency_stop": self._emergency_stop,
            "system.reset_emergency": self._reset_emergency,
            "system.get_limits": self._get_limits
        }
        
        if name not in tool_map:
            return {"error": f"Unknown safety tool: {name}"}
            
        return await tool_map[name](arguments)
        
    async def _check_interlocks(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Check all safety interlocks"""
        
        self.logger.info("Checking safety interlocks")
        
        interlock_results = self.server.safety_checker.check_all_interlocks()
        
        return {
            "status": "success",
            "runlevel": self.server.runlevel.value,
            "interlocks": interlock_results,
            "timestamp": self._get_timestamp(),
            "message": f"Interlock check completed - {interlock_results['overall_status']}"
        }
        
    async def _set_runlevel(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set system runlevel with safety checks"""
        
        target_runlevel = args.get("runlevel")
        reason = args.get("reason", "User request")
        
        if not target_runlevel:
            return {"error": "runlevel is required"}
            
        try:
            target_enum = RunLevel(target_runlevel)
        except ValueError:
            return {"error": f"Invalid runlevel: {target_runlevel}. Valid options: dry-run, sim, live"}
            
        self.logger.info(f"Runlevel change requested: {self.server.runlevel.value} -> {target_runlevel}")
        
        # Use safety checker to validate and perform the change
        result = self.server.safety_checker.request_runlevel_change(target_enum, reason)
        
        if result["success"]:
            self.server.runlevel = target_enum
            
        return {
            "status": "success" if result["success"] else "failed",
            "previous_runlevel": self.server.runlevel.value,
            "current_runlevel": self.server.runlevel.value,
            "reason": reason,
            "message": result["message"],
            "timestamp": self._get_timestamp()
        }
        
    async def _get_safety_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive safety status"""
        
        return {
            "status": "success",
            "safety_status": self.server.safety_checker.get_safety_status(),
            "timestamp": self._get_timestamp()
        }
        
    async def _validate_parameter(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a parameter value against safety limits"""
        
        parameter = args.get("parameter")
        value = args.get("value")
        
        if parameter is None or value is None:
            return {"error": "parameter and value are required"}
            
        try:
            value = float(value)
        except (ValueError, TypeError):
            return {"error": f"Value must be numeric, got: {value}"}
            
        is_safe, message = self.server.safety_checker.validate_parameter(parameter, value)
        
        return {
            "status": "success",
            "parameter": parameter,
            "value": value,
            "is_safe": is_safe,
            "message": message,
            "timestamp": self._get_timestamp()
        }
        
    async def _emergency_stop(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Activate emergency stop"""
        
        reason = args.get("reason", "Manual emergency stop via MCP")
        
        self.logger.critical(f"Emergency stop activated: {reason}")
        
        result = self.server.safety_checker.emergency_stop(reason)
        
        # Update server state
        self.server.runlevel = RunLevel.DRY_RUN
        
        # Stop all active measurements
        stopped_measurements = []
        for measurement_id, state in self.server.measurement_state.items():
            if state["status"] == "running":
                state["status"] = "emergency_stopped"
                stopped_measurements.append(measurement_id)
                
        return {
            "status": "emergency_stop_activated",
            "reason": reason,
            "stopped_measurements": stopped_measurements,
            "runlevel": self.server.runlevel.value,
            "message": result["message"],
            "timestamp": self._get_timestamp()
        }
        
    async def _reset_emergency(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Reset emergency stop (requires confirmation)"""
        
        reason = args.get("reason", "Manual reset via MCP")
        confirm = args.get("confirm", False)
        
        if not confirm:
            return {
                "error": "Emergency stop reset requires explicit confirmation",
                "message": "Set 'confirm': true to proceed with reset",
                "current_status": "emergency_stop_active"
            }
            
        self.logger.info(f"Emergency stop reset: {reason}")
        
        result = self.server.safety_checker.reset_emergency_stop(reason)
        
        return {
            "status": "emergency_stop_reset",
            "reason": reason,
            "runlevel": self.server.runlevel.value,
            "message": result["message"],
            "timestamp": self._get_timestamp()
        }
        
    async def _get_limits(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get all safety limits and their descriptions"""
        
        parameter = args.get("parameter")
        
        if parameter:
            # Get specific parameter limit
            if parameter in self.server.safety_checker.safety_limits:
                limit = self.server.safety_checker.safety_limits[parameter]
                return {
                    "status": "success",
                    "parameter": parameter,
                    "limit": {
                        "min_value": limit.min_value,
                        "max_value": limit.max_value,
                        "unit": limit.unit,
                        "description": limit.description
                    }
                }
            else:
                return {"error": f"No safety limit defined for parameter: {parameter}"}
        else:
            # Get all limits
            limits = {}
            for name, limit in self.server.safety_checker.safety_limits.items():
                limits[name] = {
                    "min_value": limit.min_value,
                    "max_value": limit.max_value,
                    "unit": limit.unit,
                    "description": limit.description
                }
                
            return {
                "status": "success",
                "limits": limits,
                "total_parameters": len(limits)
            }
            
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()