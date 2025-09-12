"""
Plot Extraction Tools for qudi MCP Integration

Provides MCP tools for extracting data from scientific plots and graphs
using advanced computer vision and RKHS spline projection techniques.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
import base64

try:
    from ..plot_extraction.rkhs_spline_projection import RKHSSplineProjector
    from ..plot_extraction.simple_image_extractor import SimpleImageExtractor
    PLOT_EXTRACTION_AVAILABLE = True
except ImportError:
    PLOT_EXTRACTION_AVAILABLE = False


class PlotExtractionTools:
    """MCP tools for extracting data from scientific plots"""
    
    def __init__(self, server):
        self.server = server
        self.logger = server.logger
        
    async def handle_tool(self, name: str, arguments: dict):
        """Route plot extraction tool calls"""
        
        if not PLOT_EXTRACTION_AVAILABLE:
            return {
                "error": "Plot extraction dependencies not available. Install with: pip install opencv-python pillow scipy matplotlib"
            }
        
        # Convert dot notation to underscore for routing
        tool_name = name.replace(".", "_")
        
        if tool_name == "plot_extract_data":
            return await self._extract_plot_data(arguments)
        elif tool_name == "plot_extract_spectrum":
            return await self._extract_spectrum_data(arguments)
        elif tool_name == "plot_analyze_with_rkhs":
            return await self._analyze_with_rkhs(arguments)
        elif tool_name == "plot_list_capabilities":
            return await self._list_capabilities(arguments)
        else:
            return {"error": f"Unknown plot extraction tool: {name}"}
    
    async def _extract_plot_data(self, args):
        """Extract data points from a plot image"""
        
        try:
            image_path = args.get("image_path")
            if not image_path or not os.path.exists(image_path):
                return {"error": "Image path not provided or file doesn't exist"}
            
            extractor = SimpleImageExtractor()
            
            # Extract data points
            data = extractor.extract_from_image(image_path)
            
            if data is None:
                return {"error": "Failed to extract data from image"}
            
            return {
                "status": "success",
                "data_points": len(data.get('x', [])),
                "x_values": data.get('x', [])[:10],  # First 10 points for preview
                "y_values": data.get('y', [])[:10],
                "extraction_method": "computer_vision",
                "message": f"Extracted {len(data.get('x', []))} data points from plot"
            }
            
        except Exception as e:
            self.logger.error(f"Plot data extraction failed: {e}")
            return {"error": f"Extraction failed: {str(e)}"}
    
    async def _extract_spectrum_data(self, args):
        """Extract spectrum data with advanced processing"""
        
        try:
            image_path = args.get("image_path")
            wavelength_range = args.get("wavelength_range", [400, 800])  # Default visible range
            
            if not image_path or not os.path.exists(image_path):
                return {"error": "Image path not provided or file doesn't exist"}
            
            # Use RKHS projector for advanced extraction
            projector = RKHSSplineProjector(
                epsilon=args.get("epsilon", 0.05),
                lambda_reg=args.get("lambda_reg", 0.001)
            )
            
            # Extract and process spectrum
            extractor = SimpleImageExtractor()
            raw_data = extractor.extract_from_image(image_path)
            
            if raw_data is None:
                return {"error": "Failed to extract raw data from spectrum"}
            
            # Apply RKHS smoothing if we have enough points
            x_data = raw_data.get('x', [])
            y_data = raw_data.get('y', [])
            
            if len(x_data) > 10:
                # Scale to wavelength range
                x_min, x_max = min(x_data), max(x_data)
                wavelength_data = [
                    wavelength_range[0] + (x - x_min) * (wavelength_range[1] - wavelength_range[0]) / (x_max - x_min)
                    for x in x_data
                ]
                
                # Fit RKHS model
                projector.fit(x_data, y_data)
                
                # Generate smooth curve
                smooth_x = [x_min + i * (x_max - x_min) / 199 for i in range(200)]
                smooth_y = projector.predict(smooth_x)
                smooth_wavelengths = [
                    wavelength_range[0] + (x - x_min) * (wavelength_range[1] - wavelength_range[0]) / (x_max - x_min)
                    for x in smooth_x
                ]
                
                return {
                    "status": "success",
                    "raw_points": len(x_data),
                    "wavelengths": wavelength_data[:20],  # First 20 for preview
                    "intensities": y_data[:20],
                    "smooth_wavelengths": smooth_wavelengths[::10],  # Every 10th point
                    "smooth_intensities": smooth_y[::10],
                    "wavelength_range": wavelength_range,
                    "processing": "RKHS_spline_projection",
                    "message": f"Extracted and smoothed spectrum with {len(x_data)} data points"
                }
            else:
                return {
                    "status": "success",
                    "raw_points": len(x_data),
                    "data": raw_data,
                    "message": "Insufficient points for RKHS processing, returning raw data"
                }
            
        except Exception as e:
            self.logger.error(f"Spectrum extraction failed: {e}")
            return {"error": f"Spectrum extraction failed: {str(e)}"}
    
    async def _analyze_with_rkhs(self, args):
        """Analyze data using RKHS spline projection"""
        
        try:
            x_data = args.get("x_data", [])
            y_data = args.get("y_data", [])
            
            if not x_data or not y_data or len(x_data) != len(y_data):
                return {"error": "Valid x_data and y_data arrays required"}
            
            # Configure RKHS parameters
            epsilon = args.get("epsilon", 0.05)
            lambda_reg = args.get("lambda_reg", 0.001)
            kernel_type = args.get("kernel_type", "gaussian")
            
            projector = RKHSSplineProjector(
                epsilon=epsilon,
                lambda_reg=lambda_reg,
                kernel_type=kernel_type
            )
            
            # Fit model
            projector.fit(x_data, y_data)
            
            # Generate predictions
            x_min, x_max = min(x_data), max(x_data)
            prediction_points = args.get("prediction_points", 100)
            x_pred = [x_min + i * (x_max - x_min) / (prediction_points - 1) for i in range(prediction_points)]
            y_pred = projector.predict(x_pred)
            
            # Calculate fit quality
            y_fit = projector.predict(x_data)
            mse = sum((y_actual - y_predicted) ** 2 for y_actual, y_predicted in zip(y_data, y_fit)) / len(y_data)
            
            return {
                "status": "success",
                "original_points": len(x_data),
                "prediction_points": prediction_points,
                "x_predicted": x_pred[::max(1, prediction_points//20)],  # Sample for preview
                "y_predicted": y_pred[::max(1, prediction_points//20)],
                "fit_quality": {
                    "mse": mse,
                    "kernel_type": kernel_type,
                    "epsilon": epsilon,
                    "lambda_reg": lambda_reg
                },
                "message": f"RKHS analysis complete with MSE = {mse:.6f}"
            }
            
        except Exception as e:
            self.logger.error(f"RKHS analysis failed: {e}")
            return {"error": f"RKHS analysis failed: {str(e)}"}
    
    async def _list_capabilities(self, args):
        """List plot extraction capabilities"""
        
        capabilities = {
            "available": PLOT_EXTRACTION_AVAILABLE,
            "tools": [
                {
                    "name": "plot.extract_data",
                    "description": "Extract x,y data points from plot images using computer vision"
                },
                {
                    "name": "plot.extract_spectrum", 
                    "description": "Advanced spectrum extraction with RKHS smoothing"
                },
                {
                    "name": "plot.analyze_with_rkhs",
                    "description": "Apply RKHS spline projection to smooth and analyze data"
                }
            ],
            "supported_formats": ["PNG", "JPG", "JPEG", "TIFF", "BMP"],
            "techniques": [
                "Computer Vision Edge Detection",
                "RKHS Spline Projection", 
                "Gaussian Kernel Smoothing",
                "Peak Detection and Analysis"
            ]
        }
        
        if not PLOT_EXTRACTION_AVAILABLE:
            capabilities["install_instructions"] = "pip install opencv-python pillow scipy matplotlib"
        
        return capabilities