"""
Safety and runlevel management for qudi MCP integration

Implements safety interlocks, runlevel management, and parameter validation
to ensure safe operation of quantum photonics experiments.
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


class RunLevel(Enum):
    """System run levels for safety"""
    DRY_RUN = "dry-run"  # No hardware interaction, simulation only
    SIM = "sim"          # Hardware simulation with realistic responses  
    LIVE = "live"        # Actual hardware control (requires approval)


@dataclass
class SafetyLimit:
    """Safety limit definition"""
    parameter: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = ""
    description: str = ""


@dataclass 
class InterLock:
    """Safety interlock definition"""
    name: str
    check_function: str  # Function name to call
    description: str
    critical: bool = False  # If True, violations halt all operations


class SafetyChecker:
    """Safety checking and validation"""
    
    def __init__(self):
        self.logger = logging.getLogger("qudi-mcp.safety")
        self.runlevel = RunLevel.DRY_RUN
        self.safety_limits = self._load_default_limits()
        self.interlocks = self._load_default_interlocks()
        self.emergency_stop_active = False
        
    def _load_default_limits(self) -> Dict[str, SafetyLimit]:
        """Load default safety limits for quantum photonics experiments"""
        
        return {
            # Laser safety
            "laser_power": SafetyLimit(
                parameter="laser_power",
                min_value=0.0,
                max_value=10.0,  # 10 mW default max
                unit="mW",
                description="Laser power limit"
            ),
            
            # Voltage limits
            "gate_voltage": SafetyLimit(
                parameter="gate_voltage", 
                min_value=-2.0,
                max_value=2.0,
                unit="V",
                description="Gate voltage limits"
            ),
            
            "bias_voltage": SafetyLimit(
                parameter="bias_voltage",
                min_value=-1.0, 
                max_value=1.0,
                unit="V", 
                description="Bias voltage limits"
            ),
            
            # Current limits  
            "source_current": SafetyLimit(
                parameter="source_current",
                min_value=-100e-6,  # 100 µA
                max_value=100e-6,
                unit="A",
                description="Source current limits"
            ),
            
            # Temperature limits
            "temperature": SafetyLimit(
                parameter="temperature",
                min_value=0.010,  # 10 mK minimum
                max_value=300.0,  # Room temperature max
                unit="K", 
                description="Cryostat temperature limits"
            ),
            
            # Magnetic field limits
            "magnetic_field": SafetyLimit(
                parameter="magnetic_field",
                min_value=-9.0,  # Tesla
                max_value=9.0,
                unit="T",
                description="Magnetic field limits"
            ),
            
            # Timing limits
            "measurement_time": SafetyLimit(
                parameter="measurement_time",
                min_value=0.001,  # 1 ms minimum
                max_value=3600.0,  # 1 hour max
                unit="s",
                description="Maximum measurement time"
            )
        }
        
    def _load_default_interlocks(self) -> List[InterLock]:
        """Load default safety interlocks"""
        
        return [
            InterLock(
                name="emergency_stop",
                check_function="check_emergency_stop", 
                description="Emergency stop button status",
                critical=True
            ),
            InterLock(
                name="cryostat_pressure",
                check_function="check_cryostat_pressure",
                description="Cryostat pressure within limits", 
                critical=True
            ),
            InterLock(
                name="laser_shutter",
                check_function="check_laser_shutter",
                description="Laser safety shutter operational",
                critical=False
            ),
            InterLock(
                name="temperature_stable",
                check_function="check_temperature_stable", 
                description="Temperature stability check",
                critical=False
            ),
            InterLock(
                name="instrument_connectivity",
                check_function="check_instrument_connectivity",
                description="All instruments responding",
                critical=False
            )
        ]
    
    def validate_parameter(self, parameter: str, value: float) -> Tuple[bool, str]:
        """Validate a parameter against safety limits"""
        
        if parameter not in self.safety_limits:
            return True, f"No safety limit defined for {parameter}"
            
        limit = self.safety_limits[parameter]
        
        if limit.min_value is not None and value < limit.min_value:
            return False, f"{parameter} {value} {limit.unit} below minimum {limit.min_value} {limit.unit}"
            
        if limit.max_value is not None and value > limit.max_value:
            return False, f"{parameter} {value} {limit.unit} above maximum {limit.max_value} {limit.unit}"
            
        return True, f"{parameter} {value} {limit.unit} within limits"
        
    def check_all_interlocks(self) -> Dict[str, Any]:
        """Check all safety interlocks"""
        
        results = {
            "overall_status": "safe",
            "emergency_stop": not self.emergency_stop_active,
            "interlocks": {},
            "critical_failures": [],
            "warnings": []
        }
        
        for interlock in self.interlocks:
            # Simulate interlock checks in dry-run mode
            if self.runlevel == RunLevel.DRY_RUN:
                status = self._simulate_interlock(interlock)
            else:
                status = self._check_real_interlock(interlock)
                
            results["interlocks"][interlock.name] = {
                "status": status["status"], 
                "description": interlock.description,
                "critical": interlock.critical,
                "message": status.get("message", "")
            }
            
            if not status["status"] and interlock.critical:
                results["critical_failures"].append(interlock.name)
                results["overall_status"] = "unsafe"
            elif not status["status"]:
                results["warnings"].append(interlock.name)
                
        return results
        
    def _simulate_interlock(self, interlock: InterLock) -> Dict[str, Any]:
        """Simulate interlock check for dry-run mode"""
        
        # Simulate all interlocks as OK for dry-run
        return {
            "status": True,
            "message": f"Simulated OK - {interlock.description}"
        }
        
    def _check_real_interlock(self, interlock: InterLock) -> Dict[str, Any]:
        """Check real interlock (placeholder for actual implementation)"""
        
        # TODO: Implement real interlock checking when qudi integration is complete
        if interlock.name == "emergency_stop":
            return {"status": not self.emergency_stop_active, "message": "Emergency stop clear"}
        
        # Default to simulated for now
        return {
            "status": True,
            "message": f"Placeholder check - {interlock.description}"
        }
        
    def request_runlevel_change(self, target_runlevel: RunLevel, reason: str = "") -> Dict[str, Any]:
        """Request runlevel change with safety checks"""
        
        self.logger.info(f"Runlevel change requested: {self.runlevel.value} -> {target_runlevel.value}")
        
        # Check if change is allowed
        if target_runlevel == RunLevel.LIVE:
            # LIVE mode requires explicit approval and safety checks
            interlock_results = self.check_all_interlocks()
            
            if interlock_results["overall_status"] != "safe":
                return {
                    "success": False,
                    "current_runlevel": self.runlevel.value,
                    "message": "Cannot enter LIVE mode - safety interlocks failed",
                    "interlocks": interlock_results
                }
                
            # TODO: Implement human approval mechanism
            return {
                "success": False, 
                "current_runlevel": self.runlevel.value,
                "message": "LIVE mode requires explicit human approval (not yet implemented)",
                "reason": reason
            }
            
        # Allow DRY_RUN and SIM mode changes
        self.runlevel = target_runlevel
        return {
            "success": True,
            "current_runlevel": self.runlevel.value,
            "previous_runlevel": self.runlevel.value,
            "message": f"Runlevel changed to {target_runlevel.value}",
            "reason": reason
        }
        
    def emergency_stop(self, reason: str = "Manual trigger") -> Dict[str, Any]:
        """Activate emergency stop"""
        
        self.emergency_stop_active = True
        self.runlevel = RunLevel.DRY_RUN  # Force back to safe mode
        
        self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        
        # TODO: Implement actual hardware stop procedures
        
        return {
            "status": "emergency_stop_active",
            "reason": reason,
            "runlevel": self.runlevel.value,
            "message": "All operations halted - manual reset required"
        }
        
    def reset_emergency_stop(self, reason: str = "") -> Dict[str, Any]:
        """Reset emergency stop (requires manual confirmation)"""
        
        # TODO: Implement safety checks before reset
        self.emergency_stop_active = False
        
        self.logger.info(f"Emergency stop reset: {reason}")
        
        return {
            "status": "emergency_stop_cleared", 
            "reason": reason,
            "runlevel": self.runlevel.value,
            "message": "Emergency stop cleared - system ready for operation"
        }
        
    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status"""
        
        return {
            "runlevel": self.runlevel.value,
            "emergency_stop_active": self.emergency_stop_active,
            "safety_limits": {
                name: {
                    "min": limit.min_value,
                    "max": limit.max_value, 
                    "unit": limit.unit,
                    "description": limit.description
                } for name, limit in self.safety_limits.items()
            },
            "interlocks": self.check_all_interlocks(),
            "last_check": "simulation_mode" if self.runlevel != RunLevel.LIVE else "hardware_check"
        }