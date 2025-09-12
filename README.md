# 🔬 qudi MCP Integration

**Model Context Protocol (MCP) integration for qudi quantum photonics experiments**

Enable natural language control of quantum experiments through Claude Desktop!

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Claude Desktop](https://img.shields.io/badge/Claude-Desktop-orange.svg)](https://claude.ai/desktop)

## 🚀 Quick Start

### 1. Installation Options

**📋 Standalone Mode (Recommended for Testing)**
```bash
# Clone this repository
git clone https://github.com/dirkenglund/qudi-mcp-integration.git
cd qudi-mcp-integration

# Install minimal dependencies (simulation only)
pip install -r requirements-standalone.txt
```

**🖼️ With Plot Extraction Capabilities**
```bash  
# Install standalone + plot extraction
pip install -r requirements-standalone.txt
pip install -r requirements-plot-extraction.txt
```

**🔬 Full qudi Integration (For Hardware Control)**
```bash
# For real quantum hardware control
pip install -r requirements-full.txt

# Additional setup required:
# 1. Install qudi-core separately
# 2. Configure hardware drivers  
# 3. Set up measurement modules
```

### 2. Configure Claude Desktop
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "qudi-mcp": {
      "command": "/opt/homebrew/bin/python3",
      "args": ["/path/to/qudi-mcp-integration/simple_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/qudi-mcp-integration",
        "QUDI_MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### 3. Test with Claude Desktop
Restart Claude Desktop and try:
```
"Get qudi station information"
```

**Example Commands:**
- "Start a photoluminescence scan from 630-650 nm" 
- "Check all safety interlocks before measurement"
- "Can I safely set laser power to 5 milliwatts?"

## Features

### 🛡️ Safety First
- **Runlevels**: `dry-run` → `sim` → `live` progression
- **Parameter Validation**: All values checked against safety limits
- **Emergency Stop**: Immediate halt capability
- **Interlocks**: Critical system monitoring

### 🔧 Instrument Control  
- List and load qudi instruments
- Get/set parameters with safety validation
- Real-time status monitoring
- Simulated operation for safe testing

### 📊 Measurement Execution
- Available modules: PL scan, gate sweep, resonance scan, time trace, 2D maps
- Progress monitoring and data acquisition
- Configurable measurement parameters
- Data export capabilities

### 🤖 LLM Integration
- Natural language command processing
- Contextual tool suggestions
- Comprehensive error reporting
- Audit logging for all operations

### 🖼️ Plot Extraction (Optional)
- **RKHS Spline Projection** - Mathematical smoothing using reproducing kernel Hilbert spaces
- **Computer Vision** - Extract data points from scientific plots and graphs
- **Spectrum Analysis** - Advanced processing for spectroscopy data
- **Multiple Formats** - Support for PNG, JPG, TIFF, and other image formats
- **Quantum Data** - Optimized for photoluminescence and transport measurements

**Plot Extraction Commands:**
- "Extract data from this plot image: /path/to/spectrum.png"
- "Analyze spectrum with RKHS smoothing using epsilon 0.05"
- "What plot extraction capabilities are available?"

## Architecture

```
Claude ←→ MCP ←→ qudi_mcp_server ←→ qudi ←→ Instruments
           ↓
     Safety System
     Runlevel Control
     Parameter Validation
```

### Core Components

- **`qudi_mcp_server.py`**: Main MCP server and tool router
- **`safety.py`**: Safety system and runlevel management  
- **`tools/`**: Tool implementations (instruments, measurements, safety)
- **`claude_config/`**: Claude Desktop configuration templates

## Safety System

### Runlevels
- **`dry-run`** (default): Simulation only, no hardware interaction
- **`sim`**: Realistic simulation with hardware-like responses  
- **`live`**: Real hardware control (requires approval + safety checks)

### Built-in Limits
- Laser power: 0-10 mW
- Gate voltages: ±2.0 V
- Bias voltages: ±1.0 V  
- Temperature: 0.01-300 K
- Magnetic field: ±9.0 T
- Measurement time: 0.001-3600 s

### Emergency Procedures
All emergency stop triggers:
- Halt all running measurements
- Force runlevel to `dry-run`
- Log incident with timestamp
- Require manual reset with confirmation

## Usage Examples

### System Status
```bash
# Check system status
→ Get qudi station information

# Response: runlevel, loaded instruments, active measurements, safety status
```

### Instrument Control
```bash
# List instruments
→ List available qudi instruments

# Load an instrument  
→ Load the laser_controller instrument

# Set parameter safely
→ Set laser power to 2.5 mW
```

### Measurements
```bash
# Start a measurement
→ Start a photoluminescence scan from 630 to 650 nm with 0.5 second integration

# Check progress
→ What's the status of running measurements?

# Get results
→ Get the measurement data for the PL scan
```

### Safety Operations
```bash
# Check safety systems
→ Check all safety interlocks

# Change runlevel (when ready for hardware)
→ Set runlevel to sim mode for realistic testing

# Emergency procedures
→ Emergency stop all operations
```

## Development Status

### ✅ Completed (Phase 1)
- MCP server framework
- Safety system with runlevels and limits
- Tool architecture for instruments, measurements, safety
- Claude Desktop integration
- Comprehensive simulation mode
- Documentation and setup guides

### 🚧 In Progress (Phase 2)  
- qudi core integration
- Real instrument driver connections
- Hardware abstraction layer
- Live mode approval workflows

### 📋 Planned (Phase 3)
- Advanced measurement protocols
- Data analysis tool integration
- Multi-user access control
- Web-based monitoring interface

## File Structure

```
mcp_integration/
├── __init__.py                 # Package initialization
├── qudi_mcp_server.py         # Main MCP server
├── safety.py                  # Safety and runlevel system
├── tools/                     # MCP tool implementations
│   ├── __init__.py
│   ├── instrument_tools.py    # Instrument control tools
│   ├── measurement_tools.py   # Measurement execution tools  
│   └── safety_tools.py        # Safety and emergency tools
├── claude_config/             # Claude configuration templates
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## Development Guidelines

### Adding New Tools
1. Implement in appropriate `tools/` module
2. Register in `qudi_mcp_server.py` tool list  
3. Add safety validation for parameters
4. Test thoroughly in dry-run mode
5. Document in tool docstrings

### Safety Requirements
- All write operations must validate parameters
- Critical operations need explicit approval in live mode
- Comprehensive error handling and logging required
- Emergency stop must work from any state

### Testing Protocol
1. **Dry-run**: Logic validation without hardware
2. **Simulation**: Realistic behavior testing
3. **Hardware**: Real instrument validation (when available)
4. **Safety**: Verify all safety mechanisms
5. **Integration**: End-to-end workflow testing

## Troubleshooting

### Common Issues

**"MCP package not found"**
```bash
pip install mcp
```

**"Tool not found" errors**  
- Check tool registration in `qudi_mcp_server.py`
- Verify tool implementation in `tools/` modules

**Safety validation failures**
- Check parameter values against limits in `safety.py`
- Use `safety.get_limits` to see current constraints

**Claude Desktop not seeing tools**
- Verify absolute paths in configuration file
- Restart Claude Desktop completely
- Check Python path and MCP server execution

### Getting Help

1. **Check logs**: MCP server logs to stderr
2. **Test tools directly**: Run `python qudi_mcp_server.py` 
3. **Validate config**: Check Claude Desktop config file syntax
4. **Start simple**: Begin with `station.info` and `safety.check_interlocks`

## Contributing

This integration is part of the MIT QPG development branch. To contribute:

1. Fork the repository
2. Create feature branches from `dev/llm-mcp-automation`
3. Follow safety-first development practices
4. Include comprehensive tests
5. Update documentation for new features

---

**Repository**: https://github.com/dirkenglund/qudi-iqo-modules-QPG  
**Branch**: `dev/llm-mcp-automation`  
**Documentation**: See `docs/LLM_MCP_INTEGRATION.md` for full details