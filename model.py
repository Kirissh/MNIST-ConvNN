import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.dropout1(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.dropout2(x)
    return self.fc2(x)


def load_model(weights_path, device=None):
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = MNISTCNN().to(device)
  state = torch.load(weights_path, map_location=device, weights_only=True)
  model.load_state_dict(state)
  model.eval()
  return model, device
