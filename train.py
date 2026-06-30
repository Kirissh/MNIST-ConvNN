import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from model import SimpleCNN
from preprocess import MNIST_TRANSFORM


def plot_training_curves(history, output_dir):
  epochs = range(1, len(history["train_loss"]) + 1)

  fig, axes = plt.subplots(1, 2, figsize=(12, 4))

  axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
  axes[0].set_xlabel("Epoch")
  axes[0].set_ylabel("Loss")
  axes[0].set_title("Training Loss")
  axes[0].grid(True, alpha=0.3)

  axes[1].plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
  axes[1].plot(epochs, history["test_acc"], marker="s", label="Test accuracy")
  axes[1].set_xlabel("Epoch")
  axes[1].set_ylabel("Accuracy (%)")
  axes[1].set_title("Train vs Test Accuracy")
  axes[1].legend()
  axes[1].grid(True, alpha=0.3)

  fig.tight_layout()
  path = output_dir / "training_curves.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def plot_confusion_matrix(y_true, y_pred, output_dir):
  cm = confusion_matrix(y_true, y_pred)
  fig, ax = plt.subplots(figsize=(8, 7))
  im = ax.imshow(cm, cmap="Blues")
  ax.set_xlabel("Predicted label")
  ax.set_ylabel("True label")
  ax.set_title("Confusion Matrix (Test Set)")
  ax.set_xticks(range(10))
  ax.set_yticks(range(10))
  for i in range(10):
    for j in range(10):
      color = "white" if cm[i, j] > cm.max() / 2 else "black"
      ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
  fig.colorbar(im, ax=ax, fraction=0.046)
  fig.tight_layout()
  path = output_dir / "confusion_matrix.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def plot_sample_predictions(model, test_loader, device, output_dir, num_samples=16):
  model.eval()
  images, labels = next(iter(test_loader))
  images, labels = images[:num_samples].to(device), labels[:num_samples]

  with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(dim=1)

  fig, axes = plt.subplots(4, 4, figsize=(8, 8))
  for ax, image, label, pred in zip(axes.flat, images.cpu(), labels, preds.cpu()):
    ax.imshow(image.squeeze(), cmap="gray")
    color = "green" if label == pred else "red"
    ax.set_title(f"True: {label}  Pred: {pred}", color=color, fontsize=10)
    ax.axis("off")

  fig.suptitle("Sample Test Predictions (green=correct, red=wrong)", fontsize=12)
  fig.tight_layout()
  path = output_dir / "sample_predictions.png"
  fig.savefig(path, dpi=150)
  plt.close(fig)
  return path


def evaluate(model, data_loader, device):
  model.eval()
  all_preds = []
  all_labels = []
  correct = 0
  total = 0
  running_loss = 0.0

  with torch.no_grad():
    for images, labels in data_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = F.cross_entropy(outputs, labels)
      preds = outputs.argmax(dim=1)
      all_preds.extend(preds.cpu().tolist())
      all_labels.extend(labels.tolist())
      running_loss += loss.item() * labels.size(0)
      correct += preds.eq(labels).sum().item()
      total += labels.size(0)

  accuracy = 100.0 * correct / total
  avg_loss = running_loss / total
  return accuracy, avg_loss, all_labels, all_preds


def train(
  epochs=15,
  batch_size=64,
  lr=0.001,
  data_dir="data",
  output="cnn_weights.pth",
  viz_dir="outputs",
  use_wandb=False,
  wandb_project="MNIST ConvNN",
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  output_dir = Path(viz_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"Using device: {device}")
  print(f"Training for {epochs} epochs\n")

  transform = MNIST_TRANSFORM

  train_loader = DataLoader(
    datasets.MNIST(data_dir, train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
  )
  test_loader = DataLoader(
    datasets.MNIST(data_dir, train=False, download=True, transform=transform),
    batch_size=1000,
    shuffle=False,
  )

  model = SimpleCNN().to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

  if use_wandb:
    import wandb

    wandb.init(
      project=wandb_project,
      name=f"CNN_bs{batch_size}_lr{lr}_ep{epochs}",
      config={
        "model": "SimpleCNN",
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": "Adam",
        "dataset": "MNIST",
        "conv1_channels": 32,
        "conv2_channels": 64,
        "hidden_dim": 128,
        "num_classes": 10,
      },
    )

  try:
    for epoch in range(1, epochs + 1):
      model.train()
      running_loss = 0.0
      correct = 0
      total = 0

      for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += outputs.argmax(dim=1).eq(labels).sum().item()
        total += labels.size(0)

      train_loss = running_loss / total
      train_acc = 100.0 * correct / total
      test_acc, test_loss, _, _ = evaluate(model, test_loader, device)

      history["train_loss"].append(train_loss)
      history["train_acc"].append(train_acc)
      history["test_loss"].append(test_loss)
      history["test_acc"].append(test_acc)

      print(
        f"Epoch {epoch:2d}/{epochs} | "
        f"loss: {train_loss:.4f} | "
        f"train acc: {train_acc:.2f}% | "
        f"test acc: {test_acc:.2f}%"
      )

      if use_wandb:
        wandb.log({
          "train_loss": train_loss,
          "train_acc": train_acc,
          "test_loss": test_loss,
          "test_acc": test_acc,
          "epoch": epoch,
        })

    output_path = Path(output)
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved model to {output_path}")

    final_acc, final_loss, y_true, y_pred = evaluate(model, test_loader, device)
    report = classification_report(y_true, y_pred, digits=4)
    best_test_acc = max(history["test_acc"])
    best_epoch = history["test_acc"].index(best_test_acc) + 1

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Final test accuracy: {final_acc:.2f}%")
    print(f"Best test accuracy:  {best_test_acc:.2f}% (epoch {best_epoch})")
    print(f"Final train accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final train loss:     {history['train_loss'][-1]:.4f}")
    print("\nClassification Report:\n")
    print(report)

    curves_path = plot_training_curves(history, output_dir)
    cm_path = plot_confusion_matrix(y_true, y_pred, output_dir)
    samples_path = plot_sample_predictions(model, test_loader, device, output_dir)

    summary = {
      "epochs": epochs,
      "device": str(device),
      "final_test_accuracy": round(final_acc, 4),
      "best_test_accuracy": round(best_test_acc, 4),
      "best_epoch": best_epoch,
      "final_train_accuracy": round(history["train_acc"][-1], 4),
      "final_train_loss": round(history["train_loss"][-1], 6),
      "history": history,
      "visualizations": {
        "training_curves": str(curves_path),
        "confusion_matrix": str(cm_path),
        "sample_predictions": str(samples_path),
      },
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
      json.dump(summary, f, indent=2)

    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
      f.write(f"Final test accuracy: {final_acc:.2f}%\n")
      f.write(f"Best test accuracy: {best_test_acc:.2f}%\n\n")
      f.write(report)

    print("\nVisualizations saved:")
    print(f"  - {curves_path}")
    print(f"  - {cm_path}")
    print(f"  - {samples_path}")
    print(f"  - {summary_path}")
    print(f"  - {report_path}")

    if use_wandb:
      wandb.log({
        "final_test_accuracy": final_acc,
        "final_test_loss": final_loss,
        "best_test_accuracy": best_test_acc,
        "best_epoch": best_epoch,
        "training_curves": wandb.Image(str(curves_path)),
        "confusion_matrix": wandb.Image(str(cm_path)),
        "sample_predictions": wandb.Image(str(samples_path)),
      })
      if wandb.run is not None:
        wandb.save(str(output_path))

    return model, history, final_acc
  finally:
    if use_wandb:
      wandb.finish()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train MNIST CNN")
  parser.add_argument("--epochs", type=int, default=15)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--data-dir", default="data")
  parser.add_argument("--output", default="cnn_weights.pth")
  parser.add_argument("--viz-dir", default="outputs")
  parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
  parser.add_argument("--wandb-project", default="MNIST ConvNN", help="W&B project name")
  args = parser.parse_args()
  train(
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    data_dir=args.data_dir,
    output=args.output,
    viz_dir=args.viz_dir,
    use_wandb=args.wandb,
    wandb_project=args.wandb_project,
  )
