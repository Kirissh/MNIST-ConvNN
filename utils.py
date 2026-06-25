import numpy as np
import torch
from PIL import Image


def preprocess_canvas(surface, size=28):
  """Convert a pygame surface to a normalized MNIST-style tensor."""
  array = pygame_surface_to_array(surface)
  image = Image.fromarray(array).convert("L")
  bbox = image.getbbox()

  if bbox is None:
    return torch.zeros(1, 1, size, size)

  cropped = image.crop(bbox)
  width, height = cropped.size
  max_dim = max(width, height)
  padded = Image.new("L", (max_dim, max_dim), 0)
  offset = ((max_dim - width) // 2, (max_dim - height) // 2)
  padded.paste(cropped, offset)
  resized = padded.resize((size, size), Image.Resampling.LANCZOS)

  array = np.asarray(resized, dtype=np.float32) / 255.0
  tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
  return (tensor - 0.1307) / 0.3081


def pygame_surface_to_array(surface):
  width, height = surface.get_size()
  pixels = np.frombuffer(surface.get_view("0"), dtype=np.uint8)
  pixels = pixels.reshape((height, width, 4))
  rgb = pixels[:, :, :3]
  grayscale = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
  return 255 - grayscale
