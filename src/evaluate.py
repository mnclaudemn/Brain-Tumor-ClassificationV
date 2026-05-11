import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from models.resnet import ResNet50Classifier


# ===================== METRICS EVALUATION =====================
def evaluate(model, dataloader, device, class_names, save_dir="outputs"):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ===================== CORE METRICS =====================
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\n================ EVALUATION RESULTS ================\n")
    print(f"🔥 Accuracy: {acc * 100:.2f}%")
    print(f"📊 F1 Score (Weighted): {f1:.4f}\n")

    print("📄 Classification Report:")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )
    print(report)

    # ===================== SAVE REPORT =====================
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    with open(f"{save_dir}/logs/classification_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write(report)

    # ===================== CONFUSION MATRIX =====================
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # annotate
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center", color="black")

    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plots/confusion_matrix.png")
    plt.close()

    print("\n✅ Confusion matrix saved to outputs/plots/")
    print("✅ Classification report saved to outputs/logs/")


# ===================== MODEL LOADING =====================
def load_model(model_path, device, num_classes=4):
    model = ResNet50Classifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ===================== RUN FUNCTION =====================
def run_evaluation(model_path, dataloader, class_names):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device, len(class_names))

    evaluate(model, dataloader, device, class_names)
