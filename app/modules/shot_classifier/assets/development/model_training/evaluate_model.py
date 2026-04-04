"""
Model Evaluation Script - Ensemble Support
Comprehensive evaluation of trained ensemble models
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
)
import joblib
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from features.SHOT_CLASSIFICATION_SYSTEM.model_training.model_registry import ModelRegistry


class EnsembleEvaluator:
    """Comprehensive evaluation for ensemble models"""
    
    def __init__(self, model_dir: str = "features/SHOT_CLASSIFICATION_SYSTEM/trained_models"):
        self.model_dir = model_dir
        
        # Load models
        self.models = self._load_models()
        self.scaler = joblib.load(f"{model_dir}/ensemble/scaler.pkl")
        self.label_encoder = joblib.load(f"{model_dir}/ensemble/label_encoder.pkl")
        self.shot_types = self.label_encoder.classes_
        
        print("✓ Models loaded successfully")
    
    def _load_models(self) -> dict:
        """Load all trained models"""
        models = {}
        for model_name in ['random_forest', 'xgboost', 'gradient_boosting']:
            model_path = f"{self.model_dir}/{model_name}/model_latest.pkl"
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                print(f"✓ Loaded {model_name}")
        return models
    
    def ensemble_predict(self, X):
        """Get ensemble predictions"""
        # Get predictions from each model
        predictions = {}
        all_probas = []
        
        for name, model in self.models.items():
            proba = model.predict_proba(X)
            predictions[name] = model.predict(X)
            all_probas.append(proba)
        
        # Average probabilities (soft voting)
        ensemble_proba = np.mean(all_probas, axis=0)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred, predictions, ensemble_proba
    
    def evaluate_comprehensive(self, X_train, y_train, X_test, y_test):
        """Perform comprehensive evaluation"""
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {
            'individual_models': {},
            'ensemble': {}
        }
        
        # Evaluate individual models
        print("\n" + "="*70)
        print("EVALUATING INDIVIDUAL MODELS")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\n{name.upper().replace('_', ' ')}:")
            
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            test_metrics = self._calculate_metrics(y_test, y_test_pred)
            
            results['individual_models'][name] = {
                'training': train_metrics,
                'testing': test_metrics,
                'confusion_matrix': {
                    'training': confusion_matrix(y_train, y_train_pred).tolist(),
                    'testing': confusion_matrix(y_test, y_test_pred).tolist()
                }
            }
            
            print(f"  Train Accuracy: {train_metrics['accuracy']*100:.2f}%")
            print(f"  Test Accuracy:  {test_metrics['accuracy']*100:.2f}%")
            print(f"  Test F1 (Weighted): {test_metrics['f1_score_weighted']*100:.2f}%")
        
        # Evaluate ensemble
        print("\n" + "="*70)
        print("EVALUATING ENSEMBLE (SOFT VOTING)")
        print("="*70)
        
        y_train_pred_ens, _, _ = self.ensemble_predict(X_train_scaled)
        y_test_pred_ens, _, _ = self.ensemble_predict(X_test_scaled)
        
        train_metrics_ens = self._calculate_metrics(y_train, y_train_pred_ens)
        test_metrics_ens = self._calculate_metrics(y_test, y_test_pred_ens)
        
        results['ensemble'] = {
            'training': train_metrics_ens,
            'testing': test_metrics_ens,
            'confusion_matrix': {
                'training': confusion_matrix(y_train, y_train_pred_ens).tolist(),
                'testing': confusion_matrix(y_test, y_test_pred_ens).tolist()
            }
        }
        
        print(f"\nTrain Accuracy: {train_metrics_ens['accuracy']*100:.2f}%")
        print(f"Test Accuracy:  {test_metrics_ens['accuracy']*100:.2f}%")
        print(f"Test F1 (Weighted): {test_metrics_ens['f1_score_weighted']*100:.2f}%")
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate all classification metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.shot_types)), zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        per_class_metrics = {}
        for i, shot_type in enumerate(self.shot_types):
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
        print(" "*15 + "ENSEMBLE EVALUATION REPORT")
        print("="*70)
        
        # Best individual model
        best_model = max(
            results['individual_models'].items(),
            key=lambda x: x[1]['testing']['accuracy']
        )
        
        print(f"\n🏆 Best Individual Model: {best_model[0].upper().replace('_', ' ')}")
        print(f"   Test Accuracy: {best_model[1]['testing']['accuracy']*100:.2f}%")
        
        # Ensemble performance
        ens = results['ensemble']['testing']
        print(f"\n📊 ENSEMBLE PERFORMANCE (Testing Set)")
        print("-" * 70)
        print(f"Accuracy:           {ens['accuracy']*100:.2f}%")
        print(f"F1-Score (Weighted): {ens['f1_score_weighted']*100:.2f}%")
        print(f"F1-Score (Macro):    {ens['f1_score_macro']*100:.2f}%")
        print(f"Precision (Weighted): {ens['precision_weighted']*100:.2f}%")
        print(f"Recall (Weighted):    {ens['recall_weighted']*100:.2f}%")
        
        # Per-class performance
        print("\n📊 PER-CLASS PERFORMANCE (Testing Set)")
        print("-" * 70)
        print(f"{'Shot Type':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        for shot_type, metrics in ens['per_class_metrics'].items():
            print(f"{shot_type:<15} "
                  f"{metrics['precision']*100:>10.2f}%  "
                  f"{metrics['recall']*100:>10.2f}%  "
                  f"{metrics['f1_score']*100:>10.2f}%  "
                  f"{metrics['support']:>8}")
        
        print("="*70)
        
        # Model assessment
        print("\n🎯 MODEL ASSESSMENT")
        print("-" * 70)
        test_acc = ens['accuracy']
        test_f1 = ens['f1_score_weighted']
        
        if test_acc >= 0.80 and test_f1 >= 0.78:
            status = "✅ EXCELLENT - RESEARCH READY"
            message = "Model meets research standards with strong generalization"
        elif test_acc >= 0.70 and test_f1 >= 0.68:
            status = "✅ GOOD"
            message = "Model performs well, suitable for deployment"
        elif test_acc >= 0.60:
            status = "⚠️  MODERATE"
            message = "Model needs improvement - consider more data or tuning"
        else:
            status = "❌ POOR"
            message = "Model requires significant improvements"
        
        print(f"Status: {status}")
        print(f"Assessment: {message}")
        
        # Improvement over old model
        print("\n📈 IMPROVEMENT ANALYSIS")
        print("-" * 70)
        print(f"Previous Model (Old): ~17% test accuracy (severe overfitting)")
        print(f"Current Ensemble: {test_acc*100:.2f}% test accuracy")
        improvement = ((test_acc - 0.17) / 0.17) * 100
        print(f"Absolute Improvement: {(test_acc - 0.17)*100:.1f} percentage points")
        print(f"Relative Improvement: {improvement:.1f}%")
        
        print("="*70 + "\n")
    
    def visualize_results(self, results, output_dir='evaluation_results'):
        """Create visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Confusion Matrix (Ensemble)
        self._plot_confusion_matrix(results, output_dir)
        
        # 2. Model Comparison
        self._plot_model_comparison(results, output_dir)
        
        # 3. Per-class F1 scores
        self._plot_per_class_metrics(results, output_dir)
    
    def _plot_confusion_matrix(self, results, output_dir):
        """Plot confusion matrix for ensemble"""
        cm_test = np.array(results['ensemble']['confusion_matrix']['testing'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.shot_types, yticklabels=self.shot_types)
        plt.title('Ensemble Model - Testing Set Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_ensemble.png", dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved")
        plt.close()
    
    def _plot_model_comparison(self, results, output_dir):
        """Compare all models"""
        models = list(results['individual_models'].keys()) + ['ensemble']
        accuracies = [results['individual_models'][m]['testing']['accuracy'] for m in results['individual_models'].keys()]
        accuracies.append(results['ensemble']['testing']['accuracy'])
        
        f1_scores = [results['individual_models'][m]['testing']['f1_score_weighted'] for m in results['individual_models'].keys()]
        f1_scores.append(results['ensemble']['testing']['f1_score_weighted'])
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, [a*100 for a in accuracies], width, label='Accuracy', color='#3498db')
        ax.bar(x + width/2, [f*100 for f in f1_scores], width, label='F1-Score', color='#2ecc71')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison (Testing Set)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison chart saved")
        plt.close()
    
    def _plot_per_class_metrics(self, results, output_dir):
        """Plot per-class F1 scores"""
        metrics = results['ensemble']['testing']['per_class_metrics']
        
        shots = list(metrics.keys())
        f1_scores = [metrics[s]['f1_score'] * 100 for s in shots]
        
        colors = ['#2ecc71' if f > 75 else '#f39c12' if f > 60 else '#e74c3c' for f in f1_scores]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(shots, f1_scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Shot Type', fontsize=12, fontweight='bold')
        plt.ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
        plt.title('Per-Class F1-Scores (Ensemble Model)', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/per_class_f1_scores.png", dpi=300, bbox_inches='tight')
        print(f"✓ Per-class metrics chart saved")
        plt.close()
    
    def save_results(self, results, output_dir='evaluation_results'):
        """Save results to JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f"{output_dir}/evaluation_report.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Evaluation report saved to {output_file}")


def load_processed_data(data_file='features/SHOT_CLASSIFICATION_SYSTEM/trained_models/ensemble/processed_dataset.npz'):
    """Load preprocessed dataset"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Processed data not found: {data_file}\n"
            "Please run ensemble_trainer.py first"
        )
    
    data = np.load(data_file, allow_pickle=True)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']


def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print(" "*10 + "CRICKET SHOT CLASSIFICATION - ENSEMBLE EVALUATION")
    print("="*70 + "\n")
    
    # Load evaluator
    evaluator = EnsembleEvaluator()
    
    # Load data
    try:
        print("Loading processed dataset...")
        X_train, X_test, y_train, y_test = load_processed_data()
        print(f"✓ Data loaded: {len(X_train)} train, {len(X_test)} test samples\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Evaluate
    print("Evaluating models...")
    results = evaluator.evaluate_comprehensive(X_train, y_train, X_test, y_test)
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    evaluator.visualize_results(results)
    
    # Save results
    evaluator.save_results(results)
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to 'evaluation_results/' directory\n")


if __name__ == "__main__":
    main()