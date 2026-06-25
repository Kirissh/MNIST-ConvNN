import sys
from pathlib import Path

import pygame
import torch
import torch.nn.functional as F

from model import load_model
from utils import preprocess_canvas

CANVAS_SIZE = 280
PREVIEW_SIZE = 84
BAR_WIDTH = 220
BAR_HEIGHT = 18
PANEL_WIDTH = 300
WINDOW_WIDTH = CANVAS_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = CANVAS_SIZE + 80
DRAW_COLOR = (0, 0, 0)
BG_COLOR = (255, 255, 255)
PANEL_BG = (245, 245, 250)
TEXT_COLOR = (30, 30, 30)
ACCENT_COLOR = (52, 120, 246)
MUTED_COLOR = (120, 120, 130)


def draw_probability_bars(screen, font, probabilities, prediction, origin):
  x, y = origin
  title = font.render("Class Probabilities", True, TEXT_COLOR)
  screen.blit(title, (x, y))
  y += 30

  for digit, prob in enumerate(probabilities):
    label = font.render(str(digit), True, TEXT_COLOR)
    screen.blit(label, (x, y + 2))

    bar_x = x + 24
    pygame.draw.rect(screen, (220, 220, 228), (bar_x, y, BAR_WIDTH, BAR_HEIGHT), border_radius=4)
    fill_width = int(BAR_WIDTH * prob)
    color = ACCENT_COLOR if digit == prediction else (160, 170, 190)
    if fill_width > 0:
      pygame.draw.rect(screen, color, (bar_x, y, fill_width, BAR_HEIGHT), border_radius=4)

    pct = font.render(f"{prob * 100:5.1f}%", True, MUTED_COLOR)
    screen.blit(pct, (bar_x + BAR_WIDTH + 10, y + 1))
    y += 26


def draw_preview(screen, tensor, origin):
  x, y = origin
  array = tensor.squeeze().numpy()
  array = ((array * 0.3081) + 0.1307).clip(0, 1)
  pixels = (array * 255).astype("uint8")
  surface = pygame.surfarray.make_surface(pixels.T)
  scaled = pygame.transform.scale(surface, (PREVIEW_SIZE, PREVIEW_SIZE))
  pygame.draw.rect(screen, (200, 200, 210), (x - 1, y - 1, PREVIEW_SIZE + 2, PREVIEW_SIZE + 2), 1)
  screen.blit(scaled, (x, y))


def main():
  weights_path = Path("mnist_cnn.pt")
  if not weights_path.exists():
    print("Model weights not found. Run: python train.py")
    sys.exit(1)

  pygame.init()
  pygame.display.set_caption("MNIST Digit Recognition")
  screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
  clock = pygame.time.Clock()

  title_font = pygame.font.SysFont("menlo", 28, bold=True)
  label_font = pygame.font.SysFont("menlo", 18)
  hint_font = pygame.font.SysFont("menlo", 14)

  canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
  canvas.fill(BG_COLOR)

  model, device = load_model(weights_path)
  prediction = None
  probabilities = [0.0] * 10
  drawing = False
  last_pos = None
  predict_timer = 0
  predict_interval = 100

  running = True
  while running:
    dt = clock.tick(60)
    predict_timer += dt

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_c:
          canvas.fill(BG_COLOR)
          prediction = None
          probabilities = [0.0] * 10
          predict_timer = predict_interval
        elif event.key == pygame.K_ESCAPE:
          running = False
      elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        mx, my = event.pos
        if mx < CANVAS_SIZE and my < CANVAS_SIZE:
          drawing = True
          last_pos = (mx, my)
          pygame.draw.circle(canvas, DRAW_COLOR, last_pos, 12)
      elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        drawing = False
        last_pos = None
        predict_timer = predict_interval
      elif event.type == pygame.MOUSEMOTION and drawing:
        mx, my = event.pos
        if mx < CANVAS_SIZE and my < CANVAS_SIZE:
          if last_pos:
            pygame.draw.line(canvas, DRAW_COLOR, last_pos, (mx, my), 24)
          pygame.draw.circle(canvas, DRAW_COLOR, (mx, my), 12)
          last_pos = (mx, my)

    if predict_timer >= predict_interval:
      predict_timer = 0
      tensor = preprocess_canvas(canvas).to(device)
      with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
      prediction = int(max(range(10), key=lambda i: probs[i]))
      probabilities = probs

    screen.fill(PANEL_BG)
    screen.blit(canvas, (0, 0))
    pygame.draw.rect(screen, (200, 200, 210), (0, 0, CANVAS_SIZE, CANVAS_SIZE), 2)

    panel_x = CANVAS_SIZE + 20
    heading = title_font.render("Prediction", True, TEXT_COLOR)
    screen.blit(heading, (panel_x, 20))

    if prediction is not None and max(probabilities) > 0.01:
      digit_text = title_font.render(str(prediction), True, ACCENT_COLOR)
      conf = label_font.render(f"{max(probabilities) * 100:.1f}% confidence", True, MUTED_COLOR)
      screen.blit(digit_text, (panel_x, 58))
      screen.blit(conf, (panel_x + 40, 66))
    else:
      placeholder = label_font.render("Draw a digit", True, MUTED_COLOR)
      screen.blit(placeholder, (panel_x, 62))

    preview_label = label_font.render("28x28 input", True, MUTED_COLOR)
    screen.blit(preview_label, (panel_x, 110))
    tensor = preprocess_canvas(canvas)
    draw_preview(screen, tensor, (panel_x, 134))

    draw_probability_bars(screen, label_font, probabilities, prediction, (panel_x, 240))

    hints = [
      "Left-click drag to draw",
      "C to clear canvas",
      "Esc to quit",
    ]
    for i, hint in enumerate(hints):
      text = hint_font.render(hint, True, MUTED_COLOR)
      screen.blit(text, (10, CANVAS_SIZE + 14 + i * 20))

    pygame.display.flip()

  pygame.quit()


if __name__ == "__main__":
  main()
