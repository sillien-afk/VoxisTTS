"""Basic tests for VoxisTTS package."""

import pytest


def test_import():
    """Test that the package can be imported."""
    import voxistts
    assert hasattr(voxistts, '__version__')
    assert voxistts.__version__ == '0.1.0'


def test_vocab():
    """Test VOCAB and VALID_PHONEMES are exported."""
    from voxistts import VOCAB, VALID_PHONEMES
    assert isinstance(VOCAB, dict)
    assert isinstance(VALID_PHONEMES, set)
    assert len(VOCAB) > 0
    assert len(VALID_PHONEMES) > 0


def test_pipeline_class():
    """Test VoxisPipeline class exists."""
    from voxistts.pipeline import VoxisPipeline
    assert hasattr(VoxisPipeline, '__call__')
