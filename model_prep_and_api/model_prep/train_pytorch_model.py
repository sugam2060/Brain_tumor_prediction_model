import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model_def import BrainTumorVGG16

def main():
    prep_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(prep_dir)
    
    train_dir = os.path.join(prep_dir, "Training")
    test_dir = os.path.join(prep_dir, "Testing")
    
    # Save report images inside model_prep/report/
    report_dir = os.path.join(prep_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Save model.pth into the api folder
    api_dir = os.path.join(project_root, "api")
    os.makedirs(api_dir, exist_ok=True)
    save_model_path = os.path.join(api_dir, "model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU] Using GPU Acceleration: {gpu_name}")
    else:
        print(f"[CPU] Using CPU (CUDA not available or PyTorch CPU version installed)")
    print(f"==================================================")

    # Image transformations (128x128 resolution, scaled to [0, 1])
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    print("Loading datasets from model_prep...")
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)

    class_names = train_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Training samples: {len(train_dataset)} | Testing samples: {len(test_dataset)}")

    # Optimize dataloaders for CUDA if available
    pin_mem = (device.type == "cuda")
    batch_size = 64 if device.type == "cuda" else 32
    num_workers = 2 if device.type == "cuda" else 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # Initialize model
    model = BrainTumorVGG16(num_classes=len(class_names), freeze_features=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    epochs = 5
    history = {"accuracy": [], "loss": []}

    print("\n=================== STARTING TRAINING ===================")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch", leave=True)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

            current_loss = running_loss / total
            current_acc = (correct / total) * 100
            train_pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"})

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        print(f"Summary Epoch [{epoch+1}/{epochs}] -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc * 100:.2f}%\n")

    # Save trained state dictionary into api folder
    torch.save(model.state_dict(), save_model_path)
    print(f"[OK] Model weights saved successfully to: {save_model_path}")

    # Plot & Save Training History in report/
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), history["accuracy"], "g-o", label="Accuracy", linewidth=2.5)
    plt.plot(range(1, epochs + 1), history["loss"], "r-o", label="Loss", linewidth=2.5)
    plt.title("Model Training History", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.xticks(range(1, epochs + 1))
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(True)
    history_plot_path = os.path.join(report_dir, "model_training_history(accuracy vs loss).png")
    plt.savefig(history_plot_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved training history plot to: {history_plot_path}")

    # Evaluate on Test Set
    print("\n=================== EVALUATING ON TEST SET ===================")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    test_pbar = tqdm(test_loader, desc="Evaluating [Test]", unit="batch", leave=True)
    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    clf_report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(clf_report)

    # Save Classification Report text file in report/
    report_text_path = os.path.join(report_dir, "classification_report.txt")
    with open(report_text_path, "w") as f:
        f.write(clf_report)
    print(f"[OK] Saved classification report to: {report_text_path}")

    # Confusion Matrix Plot in report/
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    cm_plot_path = os.path.join(report_dir, "confusion_matrics.png")
    plt.savefig(cm_plot_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved confusion matrix plot to: {cm_plot_path}")

    # ROC Curves Plot in report/
    y_test_bin = label_binarize(all_labels, classes=list(range(len(class_names))))
    n_classes = len(class_names)

    plt.figure(figsize=(9, 7))
    colors = ["cyan", "darkorange", "cornflowerblue", "green"]
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"ROC curve for {class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True)
    roc_plot_path = os.path.join(report_dir, "ROC_curve.png")
    plt.savefig(roc_plot_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved ROC curve plot to: {roc_plot_path}")
    print("\n=================== TRAINING & EVALUATION COMPLETE ===================")

if __name__ == "__main__":
    main()
