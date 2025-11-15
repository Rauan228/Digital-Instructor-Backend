"""
Application package for the Digital Inspector backend.

This file marks ``backend/app`` as a Python package so that relative imports
inside ``main.py`` (for example ``from .inference import ...``) work both when
running Uvicorn locally and when packaging the service.
"""

__all__ = ["main", "inference", "pdf_utils", "preprocess"]

