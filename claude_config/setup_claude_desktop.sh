#!/bin/bash
# Setup Claude Desktop for qudi MCP Integration

set -e

echo "🔧 Setting up Claude Desktop for qudi MCP Integration"
echo "================================================="

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_PATH="$(which python3)"

echo "Project root: $PROJECT_ROOT"
echo "Python path: $PYTHON_PATH"

# Claude Desktop config directory
CLAUDE_DIR="$HOME/Library/Application Support/Claude"
CONFIG_FILE="$CLAUDE_DIR/claude_desktop_config.json"

# Create Claude directory if it doesn't exist
mkdir -p "$CLAUDE_DIR"

# Create the configuration
cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "qudi-mcp": {
      "command": "$PYTHON_PATH",
      "args": ["$PROJECT_ROOT/mcp_integration/qudi_mcp_server.py"],
      "env": {
        "PYTHONPATH": "$PROJECT_ROOT",
        "QUDI_MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
EOF

echo ""
echo "✅ Claude Desktop configuration created at:"
echo "   $CONFIG_FILE"
echo ""
echo "📋 Configuration contents:"
cat "$CONFIG_FILE"
echo ""
echo "🚀 Next steps:"
echo "   1. Restart Claude Desktop completely"
echo "   2. Test with: 'List available qudi measurement modules'"
echo "   3. Try: 'Check all safety interlocks'"
echo "   4. Start with: 'Get qudi station information'"
echo ""
echo "⚠️  Note: The integration starts in dry-run mode for safety"
echo "   Use 'Set runlevel to sim' for realistic testing"
echo ""