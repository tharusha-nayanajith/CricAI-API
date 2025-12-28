"""
Model Registry
Manages model versions, tracking, and deployment
"""

import os
import json
import joblib
from datetime import datetime
from typing import Dict, List, Optional
import shutil


class ModelRegistry:
    """
    Registry for managing trained model versions
    Tracks performance, timestamps, and deployment status
    """
    
    def __init__(self, registry_dir: str = "features/SHOT_CLASSIFICATION_SYSTEM/trained_models"):
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(registry_dir, "model_registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing registry or create new one"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'models': {},
                'active_version': None,
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
            }
    
    def _save_registry(self):
        """Save registry to disk"""
        self.registry['metadata']['last_updated'] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, version_id: str, model_info: Dict):
        """
        Register a new model version
        
        Args:
            version_id: Unique version identifier (e.g., "20250120_143022")
            model_info: Dictionary with model metadata
        """
        self.registry['models'][version_id] = {
            'timestamp': datetime.now().isoformat(),
            'performance': model_info.get('performance', {}),
            'config': model_info.get('config', {}),
            'file_paths': model_info.get('file_paths', {}),
            'dataset_info': model_info.get('dataset_info', {}),
            'status': 'registered'
        }
        
        self._save_registry()
        print(f"✓ Registered model version: {version_id}")
    
    def set_active_version(self, version_id: str):
        """Set a model version as active (deployed)"""
        if version_id not in self.registry['models']:
            raise ValueError(f"Model version {version_id} not found in registry")
        
        # Update previous active version
        if self.registry['active_version']:
            old_version = self.registry['active_version']
            if old_version in self.registry['models']:
                self.registry['models'][old_version]['status'] = 'inactive'
        
        # Set new active version
        self.registry['active_version'] = version_id
        self.registry['models'][version_id]['status'] = 'active'
        self.registry['models'][version_id]['activated_at'] = datetime.now().isoformat()
        
        self._save_registry()
        print(f"✓ Activated model version: {version_id}")
    
    def get_active_version(self) -> Optional[str]:
        """Get currently active model version"""
        return self.registry['active_version']
    
    def get_model_info(self, version_id: str) -> Dict:
        """Get information about a specific model version"""
        if version_id not in self.registry['models']:
            raise ValueError(f"Model version {version_id} not found")
        return self.registry['models'][version_id]
    
    def list_versions(self, status: Optional[str] = None) -> List[Dict]:
        """
        List all model versions
        
        Args:
            status: Filter by status ('active', 'inactive', 'registered')
        """
        versions = []
        for version_id, info in self.registry['models'].items():
            if status is None or info['status'] == status:
                versions.append({
                    'version_id': version_id,
                    **info
                })
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda x: x['timestamp'], reverse=True)
        return versions
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare performance between two model versions"""
        info1 = self.get_model_info(version1)
        info2 = self.get_model_info(version2)
        
        comparison = {
            'version1': {
                'id': version1,
                'timestamp': info1['timestamp'],
                'performance': info1['performance']
            },
            'version2': {
                'id': version2,
                'timestamp': info2['timestamp'],
                'performance': info2['performance']
            }
        }
        
        # Calculate improvements
        if 'performance' in info1 and 'performance' in info2:
            perf1 = info1['performance']
            perf2 = info2['performance']
            
            comparison['improvements'] = {}
            
            for metric in ['train_acc', 'test_acc', 'f1_weighted']:
                if metric in perf1 and metric in perf2:
                    diff = perf2[metric] - perf1[metric]
                    improvement = (diff / perf1[metric]) * 100 if perf1[metric] > 0 else 0
                    comparison['improvements'][metric] = {
                        'absolute': round(diff, 4),
                        'percentage': round(improvement, 2)
                    }
        
        return comparison
    
    def rollback_to_version(self, version_id: str):
        """
        Rollback to a previous model version
        Copies old version files to 'latest' paths
        """
        if version_id not in self.registry['models']:
            raise ValueError(f"Model version {version_id} not found")
        
        model_info = self.registry['models'][version_id]
        
        # Copy model files
        for model_type in ['random_forest', 'xgboost', 'gradient_boosting']:
            old_path = f"{self.registry_dir}/{model_type}/model_{version_id}.pkl"
            new_path = f"{self.registry_dir}/{model_type}/model_latest.pkl"
            
            if os.path.exists(old_path):
                shutil.copy2(old_path, new_path)
                print(f"✓ Rolled back {model_type}")
        
        # Set as active
        self.set_active_version(version_id)
        
        print(f"\n✓ Successfully rolled back to version: {version_id}")
    
    def delete_version(self, version_id: str, confirm: bool = False):
        """
        Delete a model version
        
        Args:
            version_id: Version to delete
            confirm: Safety flag - must be True to actually delete
        """
        if not confirm:
            print("⚠️  Delete operation requires confirm=True parameter")
            return
        
        if version_id not in self.registry['models']:
            raise ValueError(f"Model version {version_id} not found")
        
        if version_id == self.registry['active_version']:
            raise ValueError("Cannot delete active version. Switch to another version first.")
        
        # Delete model files
        for model_type in ['random_forest', 'xgboost', 'gradient_boosting']:
            file_path = f"{self.registry_dir}/{model_type}/model_{version_id}.pkl"
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove from registry
        del self.registry['models'][version_id]
        self._save_registry()
        
        print(f"✓ Deleted model version: {version_id}")
    
    def print_summary(self):
        """Print registry summary"""
        print("\n" + "="*70)
        print(" "*20 + "MODEL REGISTRY SUMMARY")
        print("="*70)
        
        print(f"\nTotal Versions: {len(self.registry['models'])}")
        print(f"Active Version: {self.registry['active_version'] or 'None'}")
        
        if self.registry['active_version']:
            active_info = self.get_model_info(self.registry['active_version'])
            print(f"\nActive Model Performance:")
            if 'performance' in active_info:
                for metric, value in active_info['performance'].items():
                    print(f"  {metric}: {value}")
        
        print("\nAll Versions:")
        print("-" * 70)
        print(f"{'Version ID':<20} {'Status':<12} {'Test Accuracy':<15} {'Timestamp':<20}")
        print("-" * 70)
        
        for version in self.list_versions():
            test_acc = version.get('performance', {}).get('test_acc', 'N/A')
            if isinstance(test_acc, float):
                test_acc = f"{test_acc*100:.2f}%"
            
            timestamp = version['timestamp'][:19].replace('T', ' ')
            
            print(f"{version['version_id']:<20} {version['status']:<12} {str(test_acc):<15} {timestamp:<20}")
        
        print("="*70 + "\n")


def main():
    """Example usage"""
    registry = ModelRegistry()
    registry.print_summary()


if __name__ == "__main__":
    main()