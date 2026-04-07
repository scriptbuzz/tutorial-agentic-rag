import sys
import os
import logging

"""
The main entry point for the Agentic RAG application.
This script initializes the environment, suppresses non-critical log spam,
and launches the Gradio web interface.
"""
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Suppress OpenTelemetry detachment context warnings during generator execution.
# This prevents log spam without affecting actual trace captures.
# Issue tracking: https://github.com/open-telemetry/opentelemetry-python/issues/2606
class _SuppressOtelDetachWarning(logging.Filter):
    """
    OpenTelemetry sometimes triggers 'Failed to detach context' warnings 
    when streaming responses from FastAPI/Gradio. This filter silences 
    those warnings to keep the console output clean while 
    preserving the actual tracing data.
    """
    def filter(self, record):
        return "Failed to detach context" not in record.getMessage()

logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachWarning())

from ui.css import custom_css
from ui.gradio_app import create_gradio_ui

if __name__ == "__main__":
    print("\n🔨 Initializing Assistant...")
    demo = create_gradio_ui()
    print("\n🚀 Starting Assistant...")
    demo.launch(css=custom_css)