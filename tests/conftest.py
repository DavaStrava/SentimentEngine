"""Pytest configuration and fixtures"""

from hypothesis import settings, Verbosity

# Register Hypothesis profiles
settings.register_profile("ci", max_examples=100, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=20, verbosity=Verbosity.normal)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

# Use CI profile by default
settings.load_profile("ci")
