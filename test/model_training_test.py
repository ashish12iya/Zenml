"""
In this File we are basically defining the test-case and building the test for training model 
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__": 
    training_pipeline()