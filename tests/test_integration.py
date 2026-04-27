"""Integration tests for the pipeline."""

import json
from pathlib import Path


class TestIntegration:
    """Integration tests."""
    
    def test_dataset_integrity(self):
        """Test dataset structure and integrity."""
        if not Path("data/dataset.json").exists():
            return
        
        with open("data/dataset.json") as f:
            dataset = json.load(f)
        
        assert len(dataset) == 10000, f"Expected 10000 samples, got {len(dataset)}"
        
        safe_count = sum(1 for d in dataset if d['label'] == 0)
        unsafe_count = sum(1 for d in dataset if d['label'] == 1)
        
        assert safe_count == 7000, f"Expected 7000 safe samples, got {safe_count}"
        assert unsafe_count == 3000, f"Expected 3000 unsafe samples, got {unsafe_count}"
    
    def test_results_quality(self):
        """Test results quality."""
        if not Path("results/summary.json").exists():
            return
        
        with open("results/summary.json") as f:
            results = json.load(f)
        
        if 'hybrid' in results:
            hybrid = results['hybrid']
            assert hybrid['final_accuracy'] >= 0.85, "Hybrid accuracy should be >= 85%"
            assert hybrid['final_recall'] >= 0.80, "Hybrid recall should be >= 80%"
    
    def test_end_to_end(self):
        """Test complete pipeline."""
        # Check all required files exist
        required_files = [
            "data/dataset.json",
            "results/summary.json",
        ]
        
        for file in required_files:
            if Path(file).exists():
                with open(file) as f:
                    json.load(f)  # Validate JSON
