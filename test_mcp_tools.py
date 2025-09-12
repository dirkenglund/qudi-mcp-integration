#!/usr/bin/env python3
"""
Comprehensive test script for qudi MCP integration tools
Tests all available tools systematically and documents any issues found.
"""

import json
import sys
import subprocess
import asyncio
from typing import Dict, List, Any
from pathlib import Path

class MCPTester:
    """Test all MCP tools systematically"""
    
    def __init__(self):
        self.server_path = Path(__file__).parent / "simple_mcp_server.py"
        self.results = {}
        self.issues = []
        
    async def test_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test a single tool with given arguments"""
        
        if arguments is None:
            arguments = {}
            
        # Create MCP message
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            # Send message to server
            process = await asyncio.create_subprocess_exec(
                "python3", str(self.server_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send message and get response
            message_str = json.dumps(message) + "\n"
            stdout, stderr = await process.communicate(message_str.encode())
            
            # Parse response
            if stdout:
                response = json.loads(stdout.decode().strip())
                if "result" in response and "content" in response["result"]:
                    result_text = response["result"]["content"][0]["text"]
                    return {
                        "success": True,
                        "result": json.loads(result_text),
                        "raw_response": response
                    }
                elif "error" in response:
                    return {
                        "success": False,
                        "error": response["error"],
                        "raw_response": response
                    }
            
            return {
                "success": False,
                "error": "No valid response received",
                "stderr": stderr.decode() if stderr else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception occurred: {str(e)}"
            }
    
    async def test_all_tools(self):
        """Test all available tools systematically"""
        
        print("🧪 Testing qudi MCP Integration Tools")
        print("=" * 50)
        
        # Test cases for each tool
        test_cases = [
            # Station tools
            {
                "name": "station_info",
                "description": "Get station information",
                "args": {}
            },
            
            # Instrument tools
            {
                "name": "instrument_list",
                "description": "List available instruments",
                "args": {}
            },
            {
                "name": "instrument_load",
                "description": "Load an instrument",
                "args": {"instrument_name": "test_laser"}
            },
            
            # Measurement tools
            {
                "name": "measurement_list_modules",
                "description": "List measurement modules",
                "args": {}
            },
            {
                "name": "measurement_start",
                "description": "Start a measurement",
                "args": {
                    "module_name": "photoluminescence_scan",
                    "parameters": {
                        "wavelength_start": 630,
                        "wavelength_end": 650,
                        "integration_time": 0.5
                    }
                }
            },
            
            # Safety tools
            {
                "name": "safety_check_interlocks",
                "description": "Check safety interlocks",
                "args": {}
            },
            {
                "name": "safety_validate_parameter",
                "description": "Validate a parameter",
                "args": {"parameter": "laser_power", "value": 2.5}
            },
            {
                "name": "safety_validate_parameter",
                "description": "Test parameter validation (should fail)",
                "args": {"parameter": "laser_power", "value": 15.0}  # Over limit
            },
            {
                "name": "safety_set_runlevel",
                "description": "Set runlevel to simulation",
                "args": {"runlevel": "sim", "reason": "Testing MCP integration"}
            },
            
            # System tools
            {
                "name": "system_emergency_stop",
                "description": "Test emergency stop",
                "args": {"reason": "Testing emergency procedures"}
            },
            
            # Feedback tool
            {
                "name": "feedback_submit",
                "description": "Submit test feedback",
                "args": {
                    "feedback_type": "usage_improvement",
                    "message": "Testing the feedback system during MCP integration validation",
                    "user_context": "Running comprehensive tool tests"
                }
            }
        ]
        
        # Run tests
        for i, test_case in enumerate(test_cases, 1):
            tool_name = test_case["name"]
            description = test_case["description"]
            args = test_case["args"]
            
            print(f"\n{i:2d}. Testing {tool_name}")
            print(f"    Description: {description}")
            print(f"    Arguments: {args}")
            
            # Run test
            result = await self.test_tool(tool_name, args)
            
            # Store result
            self.results[f"{tool_name}_{i}"] = {
                "tool_name": tool_name,
                "description": description,
                "arguments": args,
                "result": result
            }
            
            # Display result
            if result["success"]:
                print(f"    ✅ SUCCESS")
                if "result" in result:
                    # Pretty print key parts of result
                    res = result["result"]
                    if isinstance(res, dict):
                        if "status" in res:
                            print(f"    📊 Status: {res['status']}")
                        if "runlevel" in res:
                            print(f"    🔒 Runlevel: {res['runlevel']}")
                        if "message" in res:
                            print(f"    💬 Message: {res['message']}")
                        if "error" in res:
                            print(f"    ⚠️  Tool Error: {res['error']}")
                            self.issues.append({
                                "tool": tool_name,
                                "issue": res["error"],
                                "type": "tool_error"
                            })
                    else:
                        print(f"    📄 Result: {res}")
            else:
                print(f"    ❌ FAILED: {result.get('error', 'Unknown error')}")
                self.issues.append({
                    "tool": tool_name,
                    "issue": result.get("error", "Unknown error"),
                    "type": "test_failure"
                })
            
            # Small delay between tests
            await asyncio.sleep(0.1)
    
    def generate_report(self):
        """Generate comprehensive test report"""
        
        print("\n" + "=" * 60)
        print("📋 QUDI MCP INTEGRATION TEST REPORT")
        print("=" * 60)
        
        # Summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r["result"]["success"])
        failed_tests = total_tests - successful_tests
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"   • Total tests run: {total_tests}")
        print(f"   • Successful tests: {successful_tests}")
        print(f"   • Failed tests: {failed_tests}")
        print(f"   • Success rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Tool categories analysis
        categories = {
            "station": ["station_info"],
            "instrument": ["instrument_list", "instrument_load"],
            "measurement": ["measurement_list_modules", "measurement_start"],
            "safety": ["safety_check_interlocks", "safety_validate_parameter", "safety_set_runlevel"],
            "system": ["system_emergency_stop"],
            "feedback": ["feedback_submit"]
        }
        
        print(f"\n🔧 TOOL CATEGORIES:")
        for category, tools in categories.items():
            category_tests = [r for r in self.results.values() if any(r["tool_name"] == tool for tool in tools)]
            category_success = sum(1 for r in category_tests if r["result"]["success"])
            print(f"   • {category.capitalize()}: {category_success}/{len(category_tests)} working")
        
        # Issues found
        if self.issues:
            print(f"\n⚠️  ISSUES IDENTIFIED ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. [{issue['tool']}] {issue['issue']} ({issue['type']})")
        else:
            print(f"\n✅ NO ISSUES IDENTIFIED - All tools working correctly!")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if failed_tests == 0:
            print("   • All tools are functioning correctly")
            print("   • MCP integration is ready for use")
            print("   • Consider testing with real instruments in 'sim' mode")
        else:
            print("   • Review failed tests for missing implementations")
            print("   • Check safety parameter validation logic")
            print("   • Test error handling for edge cases")
        
        # Detailed results section
        print(f"\n📄 DETAILED TEST RESULTS:")
        print("-" * 40)
        
        for test_id, test_data in self.results.items():
            tool_name = test_data["tool_name"]
            success = test_data["result"]["success"]
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"\n{tool_name} - {status}")
            if success and "result" in test_data["result"]:
                result = test_data["result"]["result"]
                if isinstance(result, dict):
                    for key, value in list(result.items())[:3]:  # Show first 3 keys
                        print(f"  {key}: {value}")
                    if len(result) > 3:
                        print(f"  ... and {len(result)-3} more fields")
            elif not success:
                print(f"  Error: {test_data['result'].get('error', 'Unknown')}")


async def main():
    """Run comprehensive MCP tool testing"""
    
    tester = MCPTester()
    
    try:
        await tester.test_all_tools()
        tester.generate_report()
        
        # Save results to file
        results_file = Path("mcp_test_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": "2024-09-11",
                "results": tester.results,
                "issues": tester.issues,
                "summary": {
                    "total_tests": len(tester.results),
                    "successful_tests": sum(1 for r in tester.results.values() if r["result"]["success"]),
                    "failed_tests": sum(1 for r in tester.results.values() if not r["result"]["success"])
                }
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file.absolute()}")
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing failed with exception: {e}")


if __name__ == "__main__":
    asyncio.run(main())