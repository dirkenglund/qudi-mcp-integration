"""
Plot Extraction Module for qudi MCP Integration

Advanced plot data extraction capabilities using RKHS spline projection
and computer vision techniques for quantum photonics data analysis.
"""

from .rkhs_spline_projection import RKHSSplineProjector
from .simple_image_extractor import SimpleImageExtractor

__all__ = ['RKHSSplineProjector', 'SimpleImageExtractor']