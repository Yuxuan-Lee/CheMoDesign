"""Reduce Boltz logging during design."""

import sys
from contextlib import contextmanager
from io import StringIO

@contextmanager
def suppress_boltz2_warnings():
    """suppress boltz2 warnings."""
    
    old_stderr = sys.stderr
    
    try:
        
        sys.stderr = StringIO()
        yield
    finally:
        
        sys.stderr = old_stderr


@contextmanager
def suppress_boltz2_stdout():
    """suppress boltz2 stdout."""
    old_stdout = sys.stdout
    
    try:
        sys.stdout = StringIO()
        yield
    finally:
        sys.stdout = old_stdout


@contextmanager
def suppress_all_boltz2_output():
    """suppress all boltz2 output."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

