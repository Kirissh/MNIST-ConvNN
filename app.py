import pygame
import numpy as np
import torch
from PIL import Image
from model import SimpleCNN
from preprocess import MNIST_TRANSFORM

pygame.init()
pygame.font.init()

CANVAS_SIZE = 300
UI_WIDTH = 380
HEIGHT = 350
screen = pygame.display.set_mode((CANVAS_SIZE + UI_WIDTH, HEIGHT))
pygame.display.set_caption("Neural Vision: Live CNN Engine")

BG_COLOR = (24, 24, 27)
CANVAS_BG = (15, 15, 18)
TEXT_DIM = (161, 161, 170)
TEXT_BRIGHT = (244, 244, 245)
BAR_BG = (63, 63, 70)
ACCENT_COLOR = (16, 185, 129)
BRUSH_COLOR = (255, 255, 255)

sys_font = "segoeui, helvetica, arial"
title_font = pygame.font.SysFont(sys_font, 22, bold=True)
label_font = pygame.font.SysFont(sys_font, 16, bold=True)
big_font = pygame.font.SysFont(sys_font, 48, bold=True)

model = SimpleCNN()
try:
  model.load_state_dict(torch.load("cnn_weights.pth", weights_only=True))
except FileNotFoundError:
  print("Error: Run train.py first to generate cnn_weights.pth!")
  pygame.quit()
  raise SystemExit(1)
model.eval()

canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
canvas.fill(CANVAS_BG)
drawing = False
last_pos = None
probs = np.zeros(10)


def predict_digit(surface):
  small = pygame.transform.smoothscale(surface, (28, 28))
  array = pygame.surfarray.array3d(small)
  gray = np.transpose(array[:, :, 0])
  tensor = MNIST_TRANSFORM(Image.fromarray(gray)).unsqueeze(0)

  with torch.no_grad():
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
  return probabilities


def draw_line_round_corners(surface, color, start, end, radius):
  dx, dy = end[0] - start[0], end[1] - start[1]
  distance = max(abs(dx), abs(dy))
  if distance == 0:
    pygame.draw.circle(surface, color, start, radius)
    return
  for i in range(distance):
    x = int(start[0] + float(i) / distance * dx)
    y = int(start[1] + float(i) / distance * dy)
    pygame.draw.circle(surface, color, (x, y), radius)


running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
    elif event.type == pygame.MOUSEBUTTONDOWN:
      drawing = True
      last_pos = pygame.mouse.get_pos()
    elif event.type == pygame.MOUSEBUTTONUP:
      drawing = False
      last_pos = None
      probs = predict_digit(canvas)
    elif event.type == pygame.KEYDOWN:
      if event.key == pygame.K_SPACE:
        canvas.fill(CANVAS_BG)
        probs = np.zeros(10)

  if drawing:
    current_pos = pygame.mouse.get_pos()
    if current_pos[0] < CANVAS_SIZE:
      if last_pos and last_pos[0] < CANVAS_SIZE:
        draw_line_round_corners(canvas, BRUSH_COLOR, last_pos, current_pos, 14)
      else:
        pygame.draw.circle(canvas, BRUSH_COLOR, current_pos, 14)
      last_pos = current_pos

  screen.fill(BG_COLOR)
  screen.blit(canvas, (0, 0))
  pygame.draw.line(screen, BAR_BG, (CANVAS_SIZE, 0), (CANVAS_SIZE, HEIGHT), 2)

  ui_x = CANVAS_SIZE + 25
  screen.blit(title_font.render("Live Neural Activation", True, TEXT_BRIGHT), (ui_x, 15))

  pred = np.argmax(probs)
  has_drawing = np.max(probs) > 0

  bar_x = ui_x + 65
  bar_max_width = 220
  bar_height = 14

  for i, p in enumerate(probs):
    y_pos = 60 + (i * 22)
    is_winner = i == pred and has_drawing
    accent = ACCENT_COLOR if is_winner else TEXT_BRIGHT
    text_color = TEXT_BRIGHT if is_winner else TEXT_DIM

    screen.blit(label_font.render(f"{i}", True, text_color), (ui_x, y_pos - 3))
    pygame.draw.rect(screen, BAR_BG, (bar_x, y_pos, bar_max_width, bar_height), border_radius=4)

    fill_width = int(p * bar_max_width)
    if fill_width > 0:
      pygame.draw.rect(screen, accent, (bar_x, y_pos, fill_width, bar_height), border_radius=4)

    pct_text = label_font.render(f"{p * 100:04.1f}%", True, text_color)
    screen.blit(pct_text, (bar_x + bar_max_width + 10, y_pos - 3))

  guess_str = str(pred) if has_drawing else "?"
  guess_color = ACCENT_COLOR if has_drawing else TEXT_DIM

  guess_label = title_font.render("NETWORK GUESS: ", True, TEXT_DIM)
  screen.blit(guess_label, (ui_x, HEIGHT - 55))

  guess_value = big_font.render(guess_str, True, guess_color)
  screen.blit(guess_value, (ui_x + 200, HEIGHT - 75))

  screen.blit(label_font.render("[ Spacebar to Clear Canvas ]", True, BAR_BG), (20, HEIGHT - 30))

  pygame.display.flip()

pygame.quit()
