from torchvision import transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

MNIST_TRANSFORM = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(MNIST_MEAN, MNIST_STD),
])
