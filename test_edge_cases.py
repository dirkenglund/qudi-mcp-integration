#!/usr/bin/env python3
"""
Test edge cases and specific validation scenarios for qudi MCP integration
"""

import json
import asyncio
from pathlib import Path
from test_mcp_tools import MCPTester

class EdgeCaseTester(MCPTester):
    """Extended tester for edge cases and validation scenarios"""
    
    async def test_parameter_validation_edge_cases(self):
        """Test parameter validation with various edge cases"""
        
        print("\n🔬 TESTING PARAMETER VALIDATION EDGE CASES")
        print("-" * 50)
        
        edge_cases = [
            # Valid parameters
            {"parameter": "laser_power", "value": 0.1, "should_be_safe": True},
            {"parameter": "laser_power", "value": 10.0, "should_be_safe": True},  # Max allowed
            
            # Invalid parameters - over limits
            {"parameter": "laser_power", "value": 15.0, "should_be_safe": False},
            {"parameter": "gate_voltage", "value": 3.0, "should_be_safe": False},  # Over ±2.0V limit
            
            # Valid gate voltages
            {"parameter": "gate_voltage", "value": 1.5, "should_be_safe": True},
            {"parameter": "gate_voltage", "value": -1.5, "should_be_safe": True},
            
            # Edge cases - zero values
            {"parameter": "laser_power", "value": 0.0, "should_be_safe": True},
            {"parameter": "gate_voltage", "value": 0.0, "should_be_safe": True},
            
            # Temperature validation
            {"parameter": "temperature", "value": 4.0, "should_be_safe": True},  # Typical operating
            {"parameter": "temperature", "value": 400.0, "should_be_safe": False},  # Over 300K limit
            
            # Unknown parameter
            {"parameter": "unknown_param", "value": 1.0, "should_be_safe": True},  # Should pass with warning
        ]
        
        results = []
        for i, case in enumerate(edge_cases, 1):
            param = case["parameter"]
            value = case["value"]
            expected_safe = case["should_be_safe"]
            
            print(f"\n{i:2d}. Testing {param} = {value} (expect safe: {expected_safe})")
            
            result = await self.test_tool("safety_validate_parameter", {
                "parameter": param,
                "value": value
            })
            
            if result["success"] and "result" in result:
                actual_safe = result["result"].get("is_safe", True)
                message = result["result"].get("message", "")
                
                if actual_safe == expected_safe:
                    status = "✅ CORRECT"
                else:
                    status = "❌ MISMATCH"
                    
                print(f"    {status}: is_safe={actual_safe}, message='{message}'")
                
                results.append({
                    "parameter": param,
                    "value": value,
                    "expected_safe": expected_safe,
                    "actual_safe": actual_safe,
                    "correct": actual_safe == expected_safe,
                    "message": message
                })
            else:
                print(f"    ❌ FAILED: {result.get('error', 'Unknown error')}")
                results.append({
                    "parameter": param,
                    "value": value,
                    "expected_safe": expected_safe,
                    "actual_safe": None,
                    "correct": False,
                    "error": result.get("error", "Unknown error")
                })
                
        return results
    
    async def test_runlevel_transitions(self):
        """Test runlevel transition logic"""
        
        print("\n🔄 TESTING RUNLEVEL TRANSITIONS")
        print("-" * 50)
        
        transitions = [
            ("dry-run", "Expected initial state"),
            ("sim", "Move to simulation mode"),
            ("dry-run", "Back to dry-run mode"),
            ("live", "Attempt live mode (should warn)"),
            ("invalid", "Invalid runlevel (should fail)")
        ]
        
        results = []
        for i, (runlevel, description) in enumerate(transitions, 1):
            print(f"\n{i}. Setting runlevel to '{runlevel}': {description}")
            
            result = await self.test_tool("safety_set_runlevel", {
                "runlevel": runlevel,
                "reason": f"Testing transition to {runlevel}"
            })
            
            if result["success"] and "result" in result:
                res_data = result["result"]
                current = res_data.get("current_runlevel", "unknown")
                message = res_data.get("message", "")
                
                print(f"    ✅ Result: runlevel={current}, message='{message}'")
                
                results.append({
                    "target_runlevel": runlevel,
                    "actual_runlevel": current,
                    "success": True,
                    "message": message
                })
            else:
                error = result.get("error", "Unknown error")
                print(f"    ❌ FAILED: {error}")
                results.append({
                    "target_runlevel": runlevel,
                    "actual_runlevel": None,
                    "success": False,
                    "error": error
                })
                
        return results
    
    async def test_measurement_parameter_validation(self):
        """Test measurement with various parameter combinations"""
        
        print("\n📊 TESTING MEASUREMENT PARAMETER VALIDATION")
        print("-" * 50)
        
        test_measurements = [
            {
                "name": "Valid PL scan",
                "module": "photoluminescence_scan",
                "params": {
                    "wavelength_start": 630,
                    "wavelength_end": 650,
                    "integration_time": 0.5,
                    "laser_power": 2.0
                },
                "should_succeed": True
            },
            {
                "name": "PL scan with high laser power",
                "module": "photoluminescence_scan", 
                "params": {
                    "wavelength_start": 630,
                    "wavelength_end": 650,
                    "integration_time": 0.5,
                    "laser_power": 15.0  # Over limit
                },
                "should_succeed": True  # Should start but with validation warnings
            },
            {
                "name": "Gate sweep measurement",
                "module": "gate_sweep",
                "params": {
                    "gate_start": -1.0,
                    "gate_end": 1.0,
                    "gate_step": 0.1,
                    "bias_voltage": 0.5,
                    "measurement_time": 0.1
                },
                "should_succeed": True
            },
            {
                "name": "Unknown measurement module",
                "module": "nonexistent_measurement",
                "params": {"param1": 1.0},
                "should_succeed": False
            }
        ]
        
        results = []
        for i, test_case in enumerate(test_measurements, 1):
            name = test_case["name"]
            module = test_case["module"]
            params = test_case["params"]
            should_succeed = test_case["should_succeed"]
            
            print(f"\n{i}. {name}")
            print(f"    Module: {module}")
            print(f"    Parameters: {params}")
            
            result = await self.test_tool("measurement_start", {
                "module_name": module,
                "parameters": params
            })
            
            if result["success"] and "result" in result:
                res_data = result["result"]
                status = res_data.get("status", "unknown")
                message = res_data.get("message", "")
                
                # Check validation results if available
                validation_results = res_data.get("validation_results", [])
                unsafe_params = [v for v in validation_results if not v.get("safe", True)]
                
                if unsafe_params:
                    print(f"    ⚠️  Unsafe parameters detected: {len(unsafe_params)}")
                    for unsafe in unsafe_params:
                        print(f"        - {unsafe['parameter']}: {unsafe['message']}")
                else:
                    print(f"    ✅ All parameters validated safely")
                
                print(f"    Status: {status}, Message: '{message}'")
                
                results.append({
                    "test_name": name,
                    "module": module,
                    "success": True,
                    "status": status,
                    "unsafe_params": len(unsafe_params),
                    "message": message
                })
            else:
                error = result.get("error", "Unknown error")
                success_match = not should_succeed  # If we expected failure, this is correct
                status_icon = "✅" if success_match else "❌"
                print(f"    {status_icon} FAILED (expected): {error}")
                
                results.append({
                    "test_name": name,
                    "module": module,
                    "success": False,
                    "error": error,
                    "expected_failure": not should_succeed
                })
                
        return results
    
    async def test_feedback_system(self):
        """Test feedback system with different types"""
        
        print("\n💬 TESTING FEEDBACK SYSTEM")
        print("-" * 50)
        
        feedback_tests = [
            {
                "type": "bug_report",
                "message": "Parameter validation for temperature seems to allow negative values",
                "context": "Testing edge cases for temperature control"
            },
            {
                "type": "feature_request", 
                "message": "Add support for multi-channel measurements",
                "context": "Working with complex quantum device characterization"
            },
            {
                "type": "usage_improvement",
                "message": "Tool names could be more intuitive - maybe use dots instead of underscores",
                "context": "Initial MCP integration testing"
            },
            {
                "type": "general",
                "message": "Overall system works well, documentation is comprehensive"
            }
        ]
        
        results = []
        for i, feedback in enumerate(feedback_tests, 1):
            fb_type = feedback["type"]
            message = feedback["message"]
            context = feedback.get("context", "")
            
            print(f"\n{i}. Submitting {fb_type} feedback")
            print(f"    Message: {message}")
            if context:
                print(f"    Context: {context}")
            
            args = {
                "feedback_type": fb_type,
                "message": message
            }
            if context:
                args["user_context"] = context
                
            result = await self.test_tool("feedback_submit", args)
            
            if result["success"] and "result" in result:
                res_data = result["result"]
                status = res_data.get("status", "unknown")
                feedback_id = res_data.get("feedback_id", "")
                
                print(f"    ✅ Submitted: {status}, ID: {feedback_id}")
                
                results.append({
                    "feedback_type": fb_type,
                    "success": True,
                    "feedback_id": feedback_id,
                    "status": status
                })
            else:
                error = result.get("error", "Unknown error")
                print(f"    ❌ FAILED: {error}")
                results.append({
                    "feedback_type": fb_type,
                    "success": False,
                    "error": error
                })
                
        return results

async def main():
    """Run edge case testing"""
    
    print("🧪 QUDI MCP INTEGRATION - EDGE CASE TESTING")
    print("=" * 60)
    
    tester = EdgeCaseTester()
    
    try:
        # Run all edge case tests
        param_results = await tester.test_parameter_validation_edge_cases()
        runlevel_results = await tester.test_runlevel_transitions()
        measurement_results = await tester.test_measurement_parameter_validation()
        feedback_results = await tester.test_feedback_system()
        
        # Generate summary report
        print("\n" + "=" * 60)
        print("📋 EDGE CASE TEST SUMMARY")
        print("=" * 60)
        
        print(f"\n🔍 Parameter Validation Tests:")
        param_correct = sum(1 for r in param_results if r.get("correct", False))
        print(f"   • {param_correct}/{len(param_results)} validations behaved correctly")
        
        param_incorrect = [r for r in param_results if not r.get("correct", False)]
        if param_incorrect:
            print(f"   • Issues found:")
            for issue in param_incorrect:
                print(f"     - {issue['parameter']} = {issue['value']}: expected safe={issue['expected_safe']}, got safe={issue.get('actual_safe', 'N/A')}")
        
        print(f"\n🔄 Runlevel Transition Tests:")
        runlevel_success = sum(1 for r in runlevel_results if r.get("success", False))
        print(f"   • {runlevel_success}/{len(runlevel_results)} transitions completed successfully")
        
        print(f"\n📊 Measurement Parameter Tests:")
        measurement_success = sum(1 for r in measurement_results if r.get("success", True))
        print(f"   • {measurement_success}/{len(measurement_results)} measurement starts completed")
        
        print(f"\n💬 Feedback System Tests:")
        feedback_success = sum(1 for r in feedback_results if r.get("success", False))
        print(f"   • {feedback_success}/{len(feedback_results)} feedback submissions successful")
        
        # Overall assessment
        total_tests = len(param_results) + len(runlevel_results) + len(measurement_results) + len(feedback_results)
        total_success = param_correct + runlevel_success + measurement_success + feedback_success
        
        print(f"\n📈 OVERALL EDGE CASE RESULTS:")
        print(f"   • Total edge case tests: {total_tests}")
        print(f"   • Successful tests: {total_success}")  
        print(f"   • Success rate: {(total_success/total_tests)*100:.1f}%")
        
        if total_success == total_tests:
            print(f"\n✅ ALL EDGE CASES PASSED - System robust and reliable!")
        else:
            print(f"\n⚠️  Some edge cases failed - review findings for improvements")
        
        # Save detailed results
        results_file = Path("edge_case_test_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": "2024-09-11",
                "parameter_validation": param_results,
                "runlevel_transitions": runlevel_results,
                "measurement_validation": measurement_results,
                "feedback_tests": feedback_results,
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": total_success,
                    "success_rate": (total_success/total_tests)*100
                }
            }, f, indent=2)
        
        print(f"\n💾 Edge case results saved to: {results_file.absolute()}")
        
    except KeyboardInterrupt:
        print("\n🛑 Edge case testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Edge case testing failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())