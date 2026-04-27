"""Unit tests for data generation and labeling."""

import json
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataGeneration:
    """Test dataset generation."""
    
    def test_dataset_files_exist(self):
        """Test that dataset files exist."""
        assert Path("data/dataset.json").exists()
        
        with open("data/dataset.json") as f:
            dataset = json.load(f)
        
        assert len(dataset) > 0
        assert all('id' in d for d in dataset)
        assert all('text' in d for d in dataset)
        assert all('label' in d for d in dataset)
    
    def test_labels_file_exists(self):
        """Test that labels file exists."""
        if Path("data/labels_oracle.json").exists():
            with open("data/labels_oracle.json") as f:
                labels = json.load(f)
            
            assert 'labels' in labels
            assert 'stats' in labels
    
    def test_results_file_exists(self):
        """Test that results file exists."""
        if Path("results/summary.json").exists():
            with open("results/summary.json") as f:
                results = json.load(f)
            
            assert 'hybrid' in results or 'random' in results
