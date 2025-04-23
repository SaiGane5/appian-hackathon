# model_evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision_macro': precision_score(self.y_test, self.y_pred, average='macro'),
            'recall_macro': recall_score(self.y_test, self.y_pred, average='macro'),
            'f1_macro': f1_score(self.y_test, self.y_test, average='macro'),
            'precision_weighted': precision_score(self.y_test, self.y_pred, average='weighted'),
            'recall_weighted': recall_score(self.y_test, self.y_pred, average='weighted'),
            'f1_weighted': f1_score(self.y_test, self.y_pred, average='weighted')
        }
        
        if self.y_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(self.y_test, self.y_proba),
                'pr_auc': self._calculate_pr_auc()
            })
            
        return metrics
    
    def _calculate_pr_auc(self):
        """Calculate Precision-Recall AUC"""
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
        return auc(recall, precision)
    
    def plot_confusion_matrix(self):
        """Plot annotated confusion matrix"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Purchase', 'Purchase'],
                    yticklabels=['No Purchase', 'Purchase'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_roc_curve(self):
        """Plot ROC curve with AUC score"""
        if self.y_proba is None:
            return
            
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.calculate_metrics()["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()
    
    def generate_report(self, save_to_csv=True):
        """Generate comprehensive evaluation report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': type(self.model).__name__,
            **self.calculate_metrics(),
            'classification_report': classification_report(self.y_test, self.y_pred, output_dict=True)
        }
        
        if save_to_csv:
            df = pd.DataFrame([report]).drop('classification_report', axis=1)
            df.to_csv('model_evaluation_report.csv', mode='a', header=not pd.io.common.file_exists('model_evaluation_report.csv'), index=False)
        
        return report

# Example usage:
if __name__ == "__main__":
    # Assuming you have trained model and test data
    evaluator = ModelEvaluator(trained_model, X_test, y_test)
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curve()
    print(evaluator.generate_report())
    pass
