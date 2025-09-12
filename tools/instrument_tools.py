"""
Instrument control tools for qudi MCP integration

Handles instrument loading, parameter management, and control operations
through qudi's instrument abstraction layer.
"""

import logging
from typing import Dict, List, Any, Optional
from ..safety import RunLevel


class InstrumentTools:
    """Instrument control and management tools"""
    
    def __init__(self, server):
        self.server = server
        self.logger = logging.getLogger("qudi-mcp.instruments")
        
    async def handle_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route instrument tool calls"""
        
        tool_map = {
            "instrument.list": self._list_instruments,
            "instrument.load": self._load_instrument, 
            "instrument.get_parameters": self._get_parameters,
            "instrument.set_parameter": self._set_parameter,
            "instrument.get_status": self._get_status
        }
        
        if name not in tool_map:
            return {"error": f"Unknown instrument tool: {name}"}
            
        return await tool_map[name](arguments)
        
    async def _list_instruments(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List available instruments"""
        
        if self.server.runlevel == RunLevel.DRY_RUN:
            # Return simulated instrument list
            return {
                "instruments": [
                    {
                        "name": "laser_controller",
                        "type": "laser", 
                        "status": "available",
                        "description": "Main laser controller for quantum dot excitation"
                    },
                    {
                        "name": "gate_dac",
                        "type": "dac",
                        "status": "available", 
                        "description": "Gate voltage DAC for quantum device control"
                    },
                    {
                        "name": "bias_dac", 
                        "type": "dac",
                        "status": "available",
                        "description": "Bias voltage DAC"
                    },
                    {
                        "name": "photon_counter",
                        "type": "counter",
                        "status": "available",
                        "description": "Single photon counting module"
                    },
                    {
                        "name": "spectrometer",
                        "type": "spectrometer", 
                        "status": "available",
                        "description": "High resolution spectrometer"
                    },
                    {
                        "name": "temperature_controller",
                        "type": "temperature",
                        "status": "available",
                        "description": "Cryostat temperature controller"
                    }
                ],
                "runlevel": self.server.runlevel.value,
                "message": "Simulated instrument list"
            }
            
        # TODO: Get real instrument list from qudi station
        return {
            "instruments": [],
            "runlevel": self.server.runlevel.value,
            "message": "qudi station not connected - use dry-run mode for testing"
        }
        
    async def _load_instrument(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Load and initialize an instrument"""
        
        instrument_name = args.get("instrument_name")
        if not instrument_name:
            return {"error": "instrument_name is required"}
            
        if self.server.runlevel == RunLevel.DRY_RUN:
            # Simulate instrument loading
            self.server.instruments[instrument_name] = {
                "name": instrument_name,
                "type": "simulated",
                "status": "loaded",
                "parameters": self._get_simulated_parameters(instrument_name)
            }
            
            return {
                "status": "success",
                "instrument": instrument_name,
                "message": f"Simulated loading of {instrument_name}",
                "runlevel": self.server.runlevel.value
            }
            
        # TODO: Implement real instrument loading via qudi
        return {
            "status": "error",
            "message": "Real instrument loading not yet implemented - use dry-run mode"
        }
        
    async def _get_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get instrument parameters and current values"""
        
        instrument_name = args.get("instrument_name")
        if not instrument_name:
            return {"error": "instrument_name is required"}
            
        if instrument_name not in self.server.instruments:
            return {"error": f"Instrument {instrument_name} not loaded"}
            
        if self.server.runlevel == RunLevel.DRY_RUN:
            parameters = self._get_simulated_parameters(instrument_name)
            return {
                "instrument": instrument_name,
                "parameters": parameters,
                "runlevel": self.server.runlevel.value,
                "message": "Simulated parameter values"
            }
            
        # TODO: Get real parameters from qudi instrument
        return {
            "error": "Real parameter reading not yet implemented"
        }
        
    async def _set_parameter(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set instrument parameter with safety validation"""
        
        instrument_name = args.get("instrument_name")
        parameter_name = args.get("parameter_name") 
        value = args.get("value")
        
        if not all([instrument_name, parameter_name, value is not None]):
            return {"error": "instrument_name, parameter_name, and value are required"}
            
        # Safety validation
        is_safe, message = self.server.safety_checker.validate_parameter(parameter_name, value)
        if not is_safe:
            return {"error": f"Safety violation: {message}"}
            
        if self.server.runlevel == RunLevel.DRY_RUN:
            return {
                "status": "success",
                "instrument": instrument_name,
                "parameter": parameter_name,
                "value": value,
                "message": f"Dry-run: Would set {parameter_name} = {value}",
                "safety_check": message
            }
            
        if self.server.runlevel == RunLevel.LIVE:
            # TODO: Implement real parameter setting with approval
            return {
                "error": "LIVE mode parameter setting requires approval (not implemented)"
            }
            
        # SIM mode
        return {
            "status": "success",
            "instrument": instrument_name, 
            "parameter": parameter_name,
            "value": value,
            "message": f"Simulated: Set {parameter_name} = {value}",
            "safety_check": message
        }
        
    async def _get_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get instrument status"""
        
        instrument_name = args.get("instrument_name")
        if not instrument_name:
            # Return status of all instruments
            return {
                "all_instruments": {
                    name: instr.get("status", "unknown")
                    for name, instr in self.server.instruments.items()
                },
                "loaded_count": len(self.server.instruments),
                "runlevel": self.server.runlevel.value
            }
            
        if instrument_name not in self.server.instruments:
            return {"error": f"Instrument {instrument_name} not loaded"}
            
        instrument = self.server.instruments[instrument_name]
        return {
            "instrument": instrument_name,
            "status": instrument.get("status", "unknown"),
            "type": instrument.get("type", "unknown"),
            "parameters": len(instrument.get("parameters", {})),
            "runlevel": self.server.runlevel.value
        }
        
    def _get_simulated_parameters(self, instrument_name: str) -> Dict[str, Any]:
        """Get simulated parameters for different instrument types"""
        
        parameter_sets = {
            "laser_controller": {
                "power": {"value": 1.0, "unit": "mW", "min": 0.0, "max": 10.0},
                "wavelength": {"value": 637.0, "unit": "nm", "min": 630.0, "max": 650.0},
                "current": {"value": 50.0, "unit": "mA", "min": 0.0, "max": 100.0},
                "temperature": {"value": 25.0, "unit": "C", "min": 20.0, "max": 30.0}
            },
            "gate_dac": {
                "gate1_voltage": {"value": 0.0, "unit": "V", "min": -2.0, "max": 2.0},
                "gate2_voltage": {"value": 0.0, "unit": "V", "min": -2.0, "max": 2.0}, 
                "gate3_voltage": {"value": 0.0, "unit": "V", "min": -2.0, "max": 2.0}
            },
            "bias_dac": {
                "bias_voltage": {"value": 0.0, "unit": "V", "min": -1.0, "max": 1.0}
            },
            "photon_counter": {
                "count_rate": {"value": 1000, "unit": "Hz", "min": 0, "max": 1000000},
                "integration_time": {"value": 0.1, "unit": "s", "min": 0.001, "max": 10.0}
            },
            "spectrometer": {
                "center_wavelength": {"value": 637.0, "unit": "nm", "min": 400.0, "max": 1000.0},
                "resolution": {"value": 0.1, "unit": "nm", "min": 0.01, "max": 1.0},
                "integration_time": {"value": 1.0, "unit": "s", "min": 0.001, "max": 60.0}
            },
            "temperature_controller": {
                "temperature": {"value": 4.2, "unit": "K", "min": 0.01, "max": 300.0},
                "heater_power": {"value": 0.0, "unit": "W", "min": 0.0, "max": 10.0}
            }
        }
        
        return parameter_sets.get(instrument_name, {
            "generic_param": {"value": 0.0, "unit": "au", "min": 0.0, "max": 100.0}
        })