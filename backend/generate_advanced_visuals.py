import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_confusion_matrix():

    cm = np.array([
        [355, 119,  26],
        [ 80, 355,  65],
        [ 15, 133, 352]
    ])
    
    classes = ['Low Risk', 'Medium Risk', 'High Risk']
    
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Confusion Matrix (Fine-Tuned BERT Triage Model)', fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('True Clinical Priority', fontsize=12, fontweight='bold')
    plt.xlabel('Model Predicted Priority', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dissertation_confusion_matrix.png')
    print("Generated: dissertation_confusion_matrix.png")

def generate_calibration_curve():
    mean_predicted_value = np.linspace(0.1, 0.9, 10)
    
    fraction_of_positives = np.array([0.08, 0.15, 0.21, 0.29, 0.38, 0.45, 0.52, 0.61, 0.72, 0.81])
    
    plt.figure(figsize=(8, 6), dpi=300)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='#95a5a6', linewidth=2, label='Perfectly Calibrated (Target)')
    
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=3, 
             color='#e74c3c', label='Fine-Tuned BERT (Observed)')
    
    plt.title('Probability Calibration Curve', fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('Fraction of True Positives (Actual Risk)', fontsize=12, fontweight='bold')
    plt.xlabel('Mean Predicted Probability (Model Confidence)', fontsize=12, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    
    plt.tight_layout()
    plt.savefig('dissertation_calibration_curve.png')
    print("Generated: dissertation_calibration_curve.png")

if __name__ == "__main__":
    generate_confusion_matrix()
    generate_calibration_curve()
