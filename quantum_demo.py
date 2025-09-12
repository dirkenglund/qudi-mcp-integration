#!/usr/bin/env python3
"""
Live Demo: Quantum Photonics Experiment Automation via Claude

This demo showcases the full capabilities of the qudi MCP integration,
simulating realistic quantum dot experiments that can be controlled
via natural language through Claude Desktop.
"""

import asyncio
import time
import json
import sys
from pathlib import Path
import numpy as np

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcp_integration.safety import RunLevel, SafetyChecker
from mcp_integration.tools.instrument_tools import InstrumentTools
from mcp_integration.tools.measurement_tools import MeasurementTools
from mcp_integration.tools.safety_tools import SafetyTools


class QuantumPhotonicsDemo:
    """Live demo of quantum photonics experiment automation"""
    
    def __init__(self):
        # Mock server for demo
        self.runlevel = RunLevel.SIM
        self.safety_checker = SafetyChecker()
        self.safety_checker.runlevel = RunLevel.SIM
        self.instruments = {}
        self.measurement_state = {}
        
        # Initialize tool modules
        self.instrument_tools = InstrumentTools(self)
        self.measurement_tools = MeasurementTools(self)
        self.safety_tools = SafetyTools(self)
        
    def demo_print(self, message, style='info'):
        """Formatted printing for demo"""
        styles = {
            'header': '\033[95m',  # Purple
            'info': '\033[94m',    # Blue  
            'success': '\033[92m', # Green
            'warning': '\033[93m', # Yellow
            'fail': '\033[91m',    # Red
            'bold': '\033[1m',     # Bold
            'end': '\033[0m'       # End
        }
        
        icons = {
            'header': '🚀',
            'info': '📋',
            'success': '✅',
            'warning': '⚠️',
            'fail': '❌',
            'quantum': '🔬',
            'data': '📊',
            'safety': '🛡️'
        }
        
        icon = icons.get(style, '📋')
        color = styles.get(style, styles['info'])
        
        print(f"{color}{icon} {message}{styles['end']}")
        
    def demo_delay(self, seconds, message=""):
        """Realistic demo timing with progress"""
        if message:
            print(f"   ⏱️  {message} ", end="", flush=True)
        
        for i in range(int(seconds * 4)):
            print(".", end="", flush=True)
            time.sleep(0.25)
        print(" Done!")
        
    async def demo_system_startup(self):
        """Demo 1: System Startup and Safety Checks"""
        
        self.demo_print("QUANTUM PHOTONICS LAB AUTOMATION DEMO", 'header')
        print("=" * 60)
        self.demo_print("Initializing quantum experiment control system...", 'info')
        print()
        
        # System status
        self.demo_print("System Status Check:", 'quantum')
        self.demo_delay(1, "Checking safety interlocks")
        
        safety_result = await self.safety_tools.handle_tool("safety.check_interlocks", {})
        self.demo_print(f"Safety Status: {safety_result['interlocks']['overall_status'].upper()}", 'success')
        self.demo_print(f"Runlevel: {self.runlevel.value.upper()} (Simulation Mode)", 'info')
        
        # List available instruments
        self.demo_print("Available Quantum Instruments:", 'quantum')
        instruments_result = await self.instrument_tools.handle_tool("instrument.list", {})
        
        for instrument in instruments_result['instruments'][:4]:
            self.demo_print(f"  • {instrument['name']}: {instrument['description']}", 'info')
            
        print()
        
    async def demo_instrument_loading(self):
        """Demo 2: Loading and Configuring Quantum Instruments"""
        
        self.demo_print("INSTRUMENT INITIALIZATION", 'header')
        print("-" * 40)
        
        instruments_to_load = [
            "laser_controller",
            "gate_dac", 
            "photon_counter",
            "spectrometer"
        ]
        
        for instrument in instruments_to_load:
            self.demo_delay(0.8, f"Loading {instrument}")
            result = await self.instrument_tools.handle_tool("instrument.load", 
                                                           {"instrument_name": instrument})
            self.demo_print(f"{instrument}: {result['status']}", 'success')
            
        # Show instrument parameters
        self.demo_print("Laser Controller Parameters:", 'quantum')
        params_result = await self.instrument_tools.handle_tool("instrument.get_parameters",
                                                              {"instrument_name": "laser_controller"})
        
        for param, details in list(params_result['parameters'].items())[:3]:
            self.demo_print(f"  • {param}: {details['value']} {details['unit']} (Range: {details['min']}-{details['max']})", 'info')
            
        print()
        
    async def demo_safety_validation(self):
        """Demo 3: Safety Parameter Validation"""
        
        self.demo_print("QUANTUM DEVICE SAFETY VALIDATION", 'header')  
        print("-" * 40)
        
        test_parameters = [
            ("laser_power", 2.5, "Safe operating power"),
            ("laser_power", 12.0, "Exceeds safety limit"),
            ("gate_voltage", 1.0, "Safe gate voltage"),
            ("gate_voltage", 3.5, "Dangerous voltage level"),
            ("temperature", 4.2, "Liquid helium temperature")
        ]
        
        for param, value, description in test_parameters:
            result = await self.safety_tools.handle_tool("safety.validate_parameter",
                                                       {"parameter": param, "value": value})
            
            status = "✅ SAFE" if result['is_safe'] else "❌ UNSAFE"
            self.demo_print(f"{param} = {value}: {status} - {description}", 
                           'success' if result['is_safe'] else 'warning')
            self.demo_print(f"   {result['message']}", 'info')
            
        print()
        
    async def demo_quantum_measurements(self):
        """Demo 4: Quantum Photonics Measurements"""
        
        self.demo_print("QUANTUM PHOTONICS EXPERIMENTS", 'header')
        print("-" * 40)
        
        # 1. Photoluminescence Spectroscopy
        self.demo_print("Starting Photoluminescence Spectroscopy...", 'quantum')
        pl_params = {
            "module_name": "photoluminescence_scan",
            "parameters": {
                "wavelength_start": 630,
                "wavelength_end": 650,
                "wavelength_step": 0.1,
                "integration_time": 1.0,
                "laser_power": 2.5
            }
        }
        
        pl_result = await self.measurement_tools.handle_tool("measurement.start", pl_params)
        measurement_id_1 = pl_result['measurement_id']
        self.demo_print(f"PL Scan Started - ID: {measurement_id_1}", 'success')
        self.demo_print(f"Estimated Duration: {pl_result['estimated_duration']:.1f} seconds", 'info')
        
        # 2. Gate Voltage Sweep
        self.demo_delay(1, "Preparing transport measurement")
        self.demo_print("Starting Quantum Transport Measurement...", 'quantum')
        
        gate_params = {
            "module_name": "gate_sweep", 
            "parameters": {
                "gate_start": -1.5,
                "gate_end": 1.5,
                "gate_step": 0.05,
                "bias_voltage": 0.1,
                "measurement_time": 0.1
            }
        }
        
        gate_result = await self.measurement_tools.handle_tool("measurement.start", gate_params)
        measurement_id_2 = gate_result['measurement_id']
        self.demo_print(f"Gate Sweep Started - ID: {measurement_id_2}", 'success')
        
        # Let measurements "run"
        self.demo_delay(2, "Acquiring quantum data")
        
        # Check status
        status_result = await self.measurement_tools.handle_tool("measurement.status", {})
        self.demo_print(f"Active Measurements: {status_result['active_measurements']}", 'info')
        
        print()
        
    async def demo_measurement_data(self):
        """Demo 5: Realistic Quantum Data"""
        
        self.demo_print("QUANTUM MEASUREMENT DATA", 'header')
        print("-" * 40)
        
        # Get measurement data
        data_result = await self.measurement_tools.handle_tool("measurement.get_data", 
                                                             {"measurement_id": "demo"})
        
        if 'data' in data_result:
            data = data_result['data']
            self.demo_print("Photoluminescence Spectrum Data:", 'data')
            
            # Show sample data points
            if 'wavelength' in data and 'intensity' in data:
                wavelengths = data['wavelength'][:5]  # First 5 points
                intensities = data['intensity'][:5]
                
                for w, i in zip(wavelengths, intensities):
                    self.demo_print(f"  λ = {w:.1f} nm → Intensity = {i:.3f}", 'info')
                    
                self.demo_print(f"  ... ({len(data['wavelength'])} total data points)", 'info')
                
                # Find peak
                peak_idx = np.argmax(data['intensity'])
                peak_wavelength = data['wavelength'][peak_idx]
                peak_intensity = data['intensity'][peak_idx]
                
                self.demo_print(f"Peak Emission: {peak_wavelength:.2f} nm (I = {peak_intensity:.3f})", 'success')
                
        print()
        
    async def demo_advanced_experiment(self):
        """Demo 6: Advanced Multi-Parameter Experiment"""
        
        self.demo_print("ADVANCED QUANTUM DOT CHARACTERIZATION", 'header')
        print("-" * 40)
        
        # 2D Stability Diagram
        self.demo_print("Generating 2D Charge Stability Diagram...", 'quantum')
        
        stability_params = {
            "module_name": "2d_gate_map",
            "parameters": {
                "gate1_start": -1.0, "gate1_end": 1.0, "gate1_steps": 50,
                "gate2_start": -0.5, "gate2_end": 0.5, "gate2_steps": 25,
                "bias_voltage": 0.05, "integration_time": 0.1
            }
        }
        
        stability_result = await self.measurement_tools.handle_tool("measurement.start", stability_params)
        self.demo_print(f"2D Map Started - {stability_params['parameters']['gate1_steps']}×{stability_params['parameters']['gate2_steps']} points", 'success')
        
        self.demo_delay(2, "Mapping charge states")
        
        # Simulate finding charge transitions
        self.demo_print("Quantum Dot Charge Analysis:", 'data')
        self.demo_print("  • (0,0) → (1,0) transition at Gate1 = -0.3V", 'info')
        self.demo_print("  • (1,0) → (0,1) transition at Gate2 = 0.2V", 'info')
        self.demo_print("  • Charging energy ≈ 2.1 meV", 'info')
        self.demo_print("  • Tunnel coupling ≈ 0.08 meV", 'info')
        
        print()
        
    async def demo_claude_integration(self):
        """Demo 7: Natural Language Integration"""
        
        self.demo_print("CLAUDE DESKTOP INTEGRATION", 'header')
        print("-" * 40)
        
        # Simulate Claude commands and responses
        claude_examples = [
            {
                "command": "Start a photoluminescence scan from 630-650 nm",
                "response": "✅ PL scan initiated with 200 wavelength points, estimated 30 seconds",
                "details": "Laser power: 2.5mW, Integration: 1.0s per point"
            },
            {
                "command": "Can I safely increase laser power to 8 milliwatts?",
                "response": "✅ Safe - 8.0mW is within the 10mW safety limit",
                "details": "Current: 2.5mW → Requested: 8.0mW → Status: APPROVED"
            },
            {
                "command": "What's the status of my gate sweep measurement?", 
                "response": "📊 Gate sweep 75% complete, 180/240 data points acquired",
                "details": "Current gate voltage: +0.8V, Signal-to-noise ratio: 23.4 dB"
            }
        ]
        
        for example in claude_examples:
            self.demo_print(f'User: "{example["command"]}"', 'info')
            self.demo_delay(0.5, "Processing natural language")
            self.demo_print(f'Claude: {example["response"]}', 'success')
            self.demo_print(f'Details: {example["details"]}', 'info')
            print()
            
    async def demo_emergency_procedures(self):
        """Demo 8: Emergency and Safety Procedures"""
        
        self.demo_print("EMERGENCY SAFETY DEMONSTRATION", 'header')
        print("-" * 40)
        
        # Emergency stop
        self.demo_print("Simulating Emergency Stop Procedure...", 'warning')
        self.demo_delay(1, "Activating emergency protocols")
        
        emergency_result = await self.safety_tools.handle_tool("system.emergency_stop",
                                                             {"reason": "Demo emergency procedure"})
        
        self.demo_print("🚨 EMERGENCY STOP ACTIVATED", 'fail')
        self.demo_print(f"Status: {emergency_result['status']}", 'warning')
        self.demo_print(f"Stopped Measurements: {len(emergency_result.get('stopped_measurements', []))}", 'info')
        
        self.demo_delay(1, "Systems halted, awaiting reset")
        
        # Reset
        self.demo_print("Resetting Emergency Stop...", 'info')
        reset_result = await self.safety_tools.handle_tool("system.reset_emergency",
                                                          {"reason": "Demo complete", "confirm": True})
        
        self.demo_print("✅ Emergency Reset Complete", 'success')
        self.demo_print("System ready for normal operation", 'info')
        
        print()
        
    async def demo_summary(self):
        """Demo Summary and Capabilities"""
        
        self.demo_print("DEMONSTRATION SUMMARY", 'header')
        print("=" * 60)
        
        capabilities = [
            "🔬 Quantum Device Control (Lasers, DACs, Spectrometers, Counters)",
            "📊 Advanced Measurements (PL, Transport, Time-resolved, 2D Maps)", 
            "🛡️ Comprehensive Safety (Parameter limits, Interlocks, Emergency stop)",
            "🤖 Natural Language Control (Claude Desktop integration)",
            "📈 Real-time Monitoring (Status, Progress, Data acquisition)",
            "⚡ Runlevel Management (Dry-run → Sim → Live progression)",
            "🧪 Simulation Modes (Safe testing and protocol development)",
            "🔧 Instrument Management (Loading, Configuration, Health checks)"
        ]
        
        self.demo_print("Demonstrated Capabilities:", 'success')
        for capability in capabilities:
            self.demo_print(f"  {capability}", 'info')
            
        print()
        self.demo_print("🎉 QUANTUM PHOTONICS AUTOMATION: FULLY OPERATIONAL!", 'header')
        self.demo_print("Ready for real-world quantum experiments via Claude Desktop", 'success')
        
        print()
        print("💡 Try these commands in Claude Desktop:")
        print("   • 'Get qudi station information'")
        print("   • 'Start a photoluminescence scan from 635-645 nm'") 
        print("   • 'Check all safety interlocks before measurement'")
        print("   • 'Generate a 2D stability diagram for quantum dot analysis'")
        
    async def run_full_demo(self):
        """Execute the complete demonstration"""
        start_time = time.time()
        
        demos = [
            self.demo_system_startup,
            self.demo_instrument_loading,
            self.demo_safety_validation,
            self.demo_quantum_measurements, 
            self.demo_measurement_data,
            self.demo_advanced_experiment,
            self.demo_claude_integration,
            self.demo_emergency_procedures,
            self.demo_summary
        ]
        
        for i, demo in enumerate(demos, 1):
            await demo()
            if i < len(demos):
                print("\n" + "⬇️ " * 20 + "\n")
                time.sleep(1)
                
        duration = time.time() - start_time
        print(f"\n🏁 Demo completed in {duration:.1f} seconds")


async def main():
    """Main demo runner"""
    demo = QuantumPhotonicsDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())