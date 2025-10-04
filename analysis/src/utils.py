"""
Utility functions for the log analysis toolkit.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


def load_text(path: str) -> str:
    """Load text from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamped_dir(root: str, dataset: str) -> str:
    """Create timestamped directory path."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root, f"{dataset}_{timestamp}")


def dump_json(obj: Any, path: str) -> None:
    """Save object as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


def save_df(df: pd.DataFrame, path: str, save_csv: bool = False) -> None:
    """Save DataFrame as parquet and optionally CSV."""
    ensure_dir(os.path.dirname(path))
    
    # Save as parquet
    df.to_parquet(path, index=False)
    
    # Save as CSV if requested
    if save_csv:
        csv_path = path.replace('.parquet', '.csv')
        df.to_csv(csv_path, index=False)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Compute exponential moving average."""
    if len(series) == 0:
        return series
    
    # Handle NaN values
    series_clean = series.dropna()
    if len(series_clean) == 0:
        return pd.Series(index=series.index, dtype=float)
    
    # Compute EMA
    ema_values = series_clean.ewm(span=span, adjust=False).mean()
    
    # Reindex to match original series
    result = pd.Series(index=series.index, dtype=float)
    result.loc[series_clean.index] = ema_values
    
    return result


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def create_manifest(output_dir: str, config: Dict, file_paths: Dict[str, str]) -> Dict:
    """Create manifest with paths, config, and metadata."""
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'output_dir': output_dir,
        'config': config,
        'files': file_paths,
        'git_commit': get_git_commit_hash(),
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    }
    return manifest
