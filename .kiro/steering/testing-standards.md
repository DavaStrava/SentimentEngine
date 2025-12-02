# Testing Standards

## Property-Based Test Tagging

All property-based tests MUST include a comment tag that references the design document property:

```python
from hypothesis import given, strategies as st

# Feature: realtime-sentiment-analysis, Property 6: Fusion score normalization
@given(
    acoustic_score=st.floats(min_value=-1.0, max_value=1.0),
    visual_score=st.floats(min_value=-1.0, max_value=1.0),
    linguistic_score=st.floats(min_value=-1.0, max_value=1.0)
)
def test_fusion_score_normalization(acoustic_score, visual_score, linguistic_score):
    """Verify fusion output is always in [-1, 1] range"""
    # Test implementation
    pass
```

**Tag Format**: `# Feature: {feature-name}, Property {number}: {property-title}`

Where:
- `{feature-name}` is the spec feature name (e.g., "realtime-sentiment-analysis")
- `{number}` is the property number from design.md (e.g., "6")
- `{property-title}` is the short property description (e.g., "Fusion score normalization")

This enables traceability between design properties and test coverage.

## Property Test Configuration

- **Minimum iterations**: 100 per property test (configured via Hypothesis settings)
- **Test location**: All property tests go in `tests/property/` directory
- **Naming convention**: `test_property_{number}_{descriptive_name}.py`

Example:
```
tests/property/
├── test_property_01_acoustic_completeness.py
├── test_property_06_fusion_normalization.py
├── test_property_07_quality_weighted_fusion.py
└── test_property_09_temporal_smoothing.py
```

## Hypothesis Configuration

Set up Hypothesis profiles in `tests/conftest.py`:

```python
from hypothesis import settings, Verbosity

settings.register_profile("ci", max_examples=100, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=20, verbosity=Verbosity.normal)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

# Use CI profile by default
settings.load_profile("ci")
```

## Unit Test Organization

- **Location**: `tests/unit/` organized by module
- **Naming**: `test_{module_name}.py`
- **Focus**: Concrete examples demonstrating correct behavior

Example:
```
tests/unit/
├── test_acoustic_analyzer.py
├── test_visual_analyzer.py
├── test_linguistic_analyzer.py
├── test_fusion_engine.py
└── test_stream_manager.py
```

## Integration Test Requirements

- **Location**: `tests/integration/`
- **Test data**: Use small sample video files (< 10 seconds)
- **Assertions**: Verify end-to-end pipeline produces valid sentiment scores
- **Performance**: Include timing assertions for latency requirements

## Test Data Management

- Store test fixtures in `tests/fixtures/`
- Use small, synthetic data for unit tests
- Use real-world samples for integration tests
- Never commit large model files to the repository

## Running Tests

```bash
# All tests
pytest tests/

# Property tests only
pytest tests/property/

# Unit tests only
pytest tests/unit/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific property test
pytest tests/property/test_property_06_fusion_normalization.py -v
```
