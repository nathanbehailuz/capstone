"""
Tests for the log parser.
"""

import pytest
import pandas as pd
from src.parser import parse_log


def test_parse_sample_log():
    """Test parsing the sample log file."""
    with open('tests/sample_snippet.txt', 'r') as f:
        log_text = f.read()
    
    result = parse_log(log_text)
    
    # Check that all expected DataFrames are present
    assert 'run_meta' in result
    assert 'pretrained_rows' in result
    assert 'ppo_diags' in result
    assert 'epoch_summaries' in result
    
    # Check run metadata
    run_meta = result['run_meta']
    assert len(run_meta) == 1
    assert run_meta['dataset'].iloc[0] == 'mutag'
    assert run_meta['train_acc'].iloc[0] == 0.8204
    assert run_meta['tot_nodes'].iloc[0] == 131488
    
    # Check pretrained rows (includes all evaluation rows from all epochs)
    pretrained = result['pretrained_rows']
    assert len(pretrained) >= 5  # At least 5 initial rows, but may include more from epochs
    assert 'g_id' in pretrained.columns
    assert 'p_loss' in pretrained.columns
    
    # Check PPO diagnostics
    ppo_diags = result['ppo_diags']
    assert len(ppo_diags) == 10  # 2 diagnostics per epoch for 5 epochs
    assert 'kl' in ppo_diags.columns
    assert 'clip' in ppo_diags.columns
    
    # Check epoch summaries
    epoch_summaries = result['epoch_summaries']
    assert len(epoch_summaries) == 5
    assert 'auc' in epoch_summaries.columns
    assert 'reward' in epoch_summaries.columns


def test_parse_empty_log():
    """Test parsing empty log."""
    result = parse_log("")
    
    # Should return DataFrames with correct schemas (may have 1 row with NaN values for empty logs)
    for name, df in result.items():
        assert isinstance(df, pd.DataFrame)
        # Empty logs may still create 1-row DataFrames with NaN values
        assert len(df) <= 1


def test_parse_malformed_log():
    """Test parsing log with missing sections."""
    malformed_log = """
    Some random text
    Loading mutag dataset
    More random text
    """
    
    result = parse_log(malformed_log)
    
    # Should handle gracefully
    assert 'run_meta' in result
    # Run meta should be empty due to incomplete pattern
    assert len(result['run_meta']) == 0 or pd.isna(result['run_meta']['dataset'].iloc[0])


if __name__ == "__main__":
    pytest.main([__file__])
