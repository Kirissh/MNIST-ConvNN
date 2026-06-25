import torch
import torchvision
from torch.utils.data import DataLoader

from model import SimpleCNN
from preprocess import MNIST_TRANSFORM


def evaluate_accuracy():
  test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=MNIST_TRANSFORM,
  )
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

  model = SimpleCNN()
  try:
    model.load_state_dict(torch.load("cnn_weights.pth", weights_only=True))
  except FileNotFoundError:
    print("Error: Train the model first to generate 'cnn_weights.pth'!")
    return

  model.eval()

  correct_predictions = 0
  total_images = 0

  print("Evaluating model performance against 10,000 unseen testing images...")
  with torch.no_grad():
    for images, labels in test_loader:
      outputs = model(images)
      _, predicted_digits = torch.max(outputs.data, 1)
      total_images += labels.size(0)
      correct_predictions += (predicted_digits == labels).sum().item()

  final_accuracy = (correct_predictions / total_images) * 100
  print("\n" + "=" * 40)
  print(f"FINAL VALIDATION ACCURACY: {final_accuracy:.2f}%")
  print(f"Correctly Classified: {correct_predictions} / {total_images} images")
  print("=" * 40)


if __name__ == "__main__":
  evaluate_accuracy()
