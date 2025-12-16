"""
Model Evaluation Script
Comprehensive evaluation of trained Random Forest model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.model_selection import cross_val_score
import joblib
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from features.SHOT_CLASSIFICATION_SYSTEM.utils.config import SHOT_TYPES, MODEL_PATH, SCALER_PATH


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize evaluator with trained model
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully")
    
    def evaluate_comprehensive(self, X_train, y_train, X_test, y_test):
        """
        Perform comprehensive model evaluation
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Get prediction probabilities
        y_train_proba = self.model.predict_proba(X_train_scaled)
        y_test_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        results = {
            'training': self._calculate_metrics(y_train, y_train_pred, y_train_proba),
            'testing': self._calculate_metrics(y_test, y_test_pred, y_test_proba),
            'confusion_matrix': {
                'training': confusion_matrix(y_train, y_train_pred).tolist(),
                'testing': confusion_matrix(y_test, y_test_pred).tolist()
            }
        }
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        results['cross_validation'] = {
            'mean_accuracy': float(cv_scores.mean()),
            'std_accuracy': float(cv_scores.std()),
            'all_scores': cv_scores.tolist()
        }
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        results['feature_importance'] = {
            f'feature_{i}': float(imp) 
            for i, imp in enumerate(feature_importance)
        }
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate all classification metrics"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=SHOT_TYPES
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        # Per-class details
        per_class_metrics = {}
        for i, shot_type in enumerate(SHOT_TYPES):
            per_class_metrics[shot_type] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        return {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_score_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_score_macro': float(f1_macro),
            'per_class_metrics': per_class_metrics
        }
    
    def print_evaluation_report(self, results):
        """Print formatted evaluation report"""
        print("\n" + "="*70)
        print(" "*20 + "MODEL EVALUATION REPORT")
        print("="*70)
        
        # Training metrics
        print("\n📊 TRAINING SET PERFORMANCE")
        print("-" * 70)
        train = results['training']
        print(f"Accuracy:           {train['accuracy']*100:.2f}%")
        print(f"Precision (Weighted): {train['precision_weighted']*100:.2f}%")
        print(f"Recall (Weighted):    {train['recall_weighted']*100:.2f}%")
        print(f"F1-Score (Weighted):  {train['f1_score_weighted']*100:.2f}%")
        print(f"F1-Score (Macro):     {train['f1_score_macro']*100:.2f}%")
        
        # Testing metrics
        print("\n📊 TESTING SET PERFORMANCE")
        print("-" * 70)
        test = results['testing']
        print(f"Accuracy:           {test['accuracy']*100:.2f}%")
        print(f"Precision (Weighted): {test['precision_weighted']*100:.2f}%")
        print(f"Recall (Weighted):    {test['recall_weighted']*100:.2f}%")
        print(f"F1-Score (Weighted):  {test['f1_score_weighted']*100:.2f}%")
        print(f"F1-Score (Macro):     {test['f1_score_macro']*100:.2f}%")
        
        # Cross-validation
        print("\n📊 CROSS-VALIDATION (5-Fold)")
        print("-" * 70)
        cv = results['cross_validation']
        print(f"Mean Accuracy:      {cv['mean_accuracy']*100:.2f}%")
        print(f"Std Deviation:      {cv['std_accuracy']*100:.2f}%")
        
        # Per-class performance (Testing)
        print("\n📊 PER-CLASS PERFORMANCE (Testing Set)")
        print("-" * 70)
        print(f"{'Shot Type':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        for shot_type, metrics in test['per_class_metrics'].items():
            print(f"{shot_type:<15} "
                  f"{metrics['precision']*100:>10.2f}%  "
                  f"{metrics['recall']*100:>10.2f}%  "
                  f"{metrics['f1_score']*100:>10.2f}%  "
                  f"{metrics['support']:>8}")
        
        print("="*70)
        
        # Model assessment
        print("\n🎯 MODEL ASSESSMENT")
        print("-" * 70)
        test_acc = test['accuracy']
        test_f1 = test['f1_score_weighted']
        
        if test_acc >= 0.87 and test_f1 >= 0.85:
            status = "✅ EXCELLENT"
            message = "Model meets research objectives (>87% accuracy)"
        elif test_acc >= 0.80 and test_f1 >= 0.78:
            status = "✅ GOOD"
            message = "Model performs well, close to target"
        elif test_acc >= 0.70:
            status = "⚠️  MODERATE"
            message = "Model needs improvement - consider more training data"
        else:
            status = "❌ POOR"
            message = "Model requires significant improvements"
        
        print(f"Status: {status}")
        print(f"Assessment: {message}")
        print("="*70 + "\n")
    
    def visualize_confusion_matrix(self, results, output_dir='evaluation_results'):
        """Create and save confusion matrix visualization"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Training confusion matrix
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        cm_train = np.array(results['confusion_matrix']['training'])
        cm_test = np.array(results['confusion_matrix']['testing'])
        
        # Plot training
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                   xticklabels=SHOT_TYPES, yticklabels=SHOT_TYPES,
                   ax=axes[0])
        axes[0].set_title('Training Set Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Plot testing
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
                   xticklabels=SHOT_TYPES, yticklabels=SHOT_TYPES,
                   ax=axes[1])
        axes[1].set_title('Testing Set Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrices saved to {output_dir}/confusion_matrices.png")
        plt.close()
    
    def visualize_per_class_metrics(self, results, output_dir='evaluation_results'):
        """Visualize per-class performance metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        test_metrics = results['testing']['per_class_metrics']
        
        # Prepare data
        shot_types = list(test_metrics.keys())
        precision = [test_metrics[s]['precision'] for s in shot_types]
        recall = [test_metrics[s]['recall'] for s in shot_types]
        f1 = [test_metrics[s]['f1_score'] for s in shot_types]
        
        # Create bar plot
        x = np.arange(len(shot_types))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        ax.bar(x, recall, width, label='Recall', color='#2ecc71')
        ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Shot Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics (Testing Set)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(shot_types)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Per-class metrics chart saved to {output_dir}/per_class_metrics.png")
        plt.close()
    
    def save_evaluation_report(self, results, output_dir='evaluation_results'):
        """Save evaluation results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'evaluation_report.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Evaluation report saved to {output_file}")


def load_processed_data(data_file='processed_dataset.npz'):
    """
    Load preprocessed dataset (you should save this during training)
    
    Args:
        data_file: Path to saved numpy dataset
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Processed data not found: {data_file}\n"
            "Please run train_model.py first to generate the dataset"
        )
    
    data = np.load(data_file, allow_pickle=True)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']


def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print(" "*15 + "CRICKET SHOT CLASSIFICATION")
    print(" "*20 + "MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Trained model not found!")
        print(f"Expected location: {MODEL_PATH}")
        print("\nPlease run train_model.py first to train the model.")
        return
    
    # Load model
    evaluator = ModelEvaluator(MODEL_PATH, SCALER_PATH)
    
    # Load processed data
    try:
        print("Loading processed dataset...")
        X_train, X_test, y_train, y_test = load_processed_data()
        print(f"✅ Data loaded successfully")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
    except FileNotFoundError as e:
        print(f"\n❌ {str(e)}")
        print("\nTo generate the dataset, modify train_model.py to save data:")
        print("Add after train_test_split:")
        print("  np.savez('processed_dataset.npz',")
        print("           X_train=X_train, X_test=X_test,")
        print("           y_train=y_train, y_test=y_test)")
        return
    
    # Perform comprehensive evaluation
    print("\nPerforming comprehensive evaluation...")
    results = evaluator.evaluate_comprehensive(X_train, y_train, X_test, y_test)
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    evaluator.visualize_confusion_matrix(results)
    evaluator.visualize_per_class_metrics(results)
    
    # Save results
    evaluator.save_evaluation_report(results)
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to 'evaluation_results/' directory")


if __name__ == "__main__":
    main()