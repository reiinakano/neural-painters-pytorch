# Most of this code is based on Taehoon Kim's SPIRAL implementation https://github.com/carpedm20/SPIRAL-tensorflow

try:
  from lib import surface, tiledsurface, brush  # , floodfill for latest mypaint
except ImportError:
  print("Cannot find import name `lib` (MyPaint). Make sure to add it to your path e.g. with sys.append()."
        "See example Colab notebooks for details on installation and importing.")
  raise

import numpy as np
from PIL import Image

import math


## Curve Math
def point_on_curve_1(t, cx, cy, sx, sy, x1, y1, x2, y2):
  ratio = t / 100.0
  x3, y3 = multiply_add(sx, sy, x1, y1, ratio)
  x4, y4 = multiply_add(cx, cy, x2, y2, ratio)
  x5, y5 = difference(x3, y3, x4, y4)
  x, y = multiply_add(x3, y3, x5, y5, ratio)
  return x, y


def point_on_curve_2(t, cx, cy, sx, sy, kx, ky, x1, y1, x2, y2, x3, y3):
  ratio = t / 100.0
  x4, y4 = multiply_add(sx, sy, x1, y1, ratio)
  x5, y5 = multiply_add(cx, cy, x2, y2, ratio)
  x6, y6 = multiply_add(kx, ky, x3, y3, ratio)
  x1, y1 = difference(x4, y4, x5, y5)
  x2, y2 = difference(x5, y5, x6, y6)
  x4, y4 = multiply_add(x4, y4, x1, y1, ratio)
  x5, y5 = multiply_add(x5, y5, x2, y2, ratio)
  x1, y1 = difference(x4, y4, x5, y5)
  x, y = multiply_add(x4, y4, x1, y1, ratio)
  return x, y


## Ellipse Math
def starting_point_for_ellipse(x, y, rotate):
  # Rotate starting point
  r = math.radians(rotate)
  sin = math.sin(r)
  cos = math.cos(r)
  x, y = rotate_ellipse(x, y, cos, sin)
  return x, y, sin, cos


def point_in_ellipse(x, y, r_sin, r_cos, degree):
  # Find point in ellipse
  r2 = math.radians(degree)
  cos = math.cos(r2)
  sin = math.sin(r2)
  x = x * cos
  y = y * sin
  # Rotate Ellipse
  x, y = rotate_ellipse(y, x, r_sin, r_cos)
  return x, y


def rotate_ellipse(x, y, sin, cos):
  x1, y1 = multiply(x, y, sin)
  x2, y2 = multiply(x, y, cos)
  x = x2 - y1
  y = y2 + x1
  return x, y


## Vector Math
def get_angle(x1, y1, x2, y2):
  dot = dot_product(x1, y1, x2, y2)
  if abs(dot) < 1.0:
    angle = math.acos(dot) * 180 / math.pi
  else:
    angle = 0.0
  return angle


def constrain_to_angle(x, y, sx, sy):
  length, nx, ny = length_and_normal(sx, sy, x, y)
  # dot = nx*1 + ny*0 therefore nx
  angle = math.acos(nx) * 180 / math.pi
  angle = constraint_angle(angle)
  ax, ay = angle_normal(ny, angle)
  x = sx + ax * length
  y = sy + ay * length
  return x, y


def constraint_angle(angle):
  n = angle // 15
  n1 = n * 15
  rem = angle - n1
  if rem < 7.5:
    angle = n * 15.0
  else:
    angle = (n + 1) * 15.0
  return angle


def angle_normal(ny, angle):
  if ny < 0.0:
    angle = 360.0 - angle
  radians = math.radians(angle)
  x = math.cos(radians)
  y = math.sin(radians)
  return x, y


def length_and_normal(x1, y1, x2, y2):
  x, y = difference(x1, y1, x2, y2)
  length = vector_length(x, y)
  if length == 0.0:
    x, y = 0.0, 0.0
  else:
    x, y = x / length, y / length
  return length, x, y


def normal(x1, y1, x2, y2):
  junk, x, y = length_and_normal(x1, y1, x2, y2)
  return x, y


def vector_length(x, y):
  length = math.sqrt(x * x + y * y)
  return length


def distance(x1, y1, x2, y2):
  x, y = difference(x1, y1, x2, y2)
  length = vector_length(x, y)
  return length


def dot_product(x1, y1, x2, y2):
  return x1 * x2 + y1 * y2


def multiply_add(x1, y1, x2, y2, d):
  x3, y3 = multiply(x2, y2, d)
  x, y = add(x1, y1, x3, y3)
  return x, y


def multiply(x, y, d):
  # Multiply vector
  x = x * d
  y = y * d
  return x, y


def add(x1, y1, x2, y2):
  # Add vectors
  x = x1 + x2
  y = y1 + y2
  return x, y


def difference(x1, y1, x2, y2):
  # Difference in x and y between two points
  x = x2 - x1
  y = y2 - y1
  return x, y


def midpoint(x1, y1, x2, y2):
  # Midpoint between to points
  x = (x1 + x2) / 2.0
  y = (y1 + y2) / 2.0
  return x, y


def perpendicular(x1, y1):
  # Swap x and y, then flip one sign to give vector at 90 degree
  x = -y1
  y = x1
  return x, y


class PaintMode:
  STROKES_ONLY = 0
  JUMP_STROKES = 1
  CONNECTED_STROKES = 2


class ColorEnv():
  head = 0.25
  tail = 0.75

  # all 0 to 1
  actions_to_idx = {
    'pressure': 0,
    'size': 1,
    'control_x': 2,
    'control_y': 3,
    'end_x': 4,
    'end_y': 5,
    'color_r': 6,
    'color_g': 7,
    'color_b': 8,
    'start_x': 9,
    'start_y': 10,
    'entry_pressure': 11,
  }

  def __init__(self, paint_mode=PaintMode.JUMP_STROKES,
               brush_path='mypaint-brushes/brushes/classic/dry_brush.myb',
               screen_size=64, location_size=32, color_channel=3):
    self.paint_mode = paint_mode
    self.brush_path = brush_path

    # screen
    self.screen_size = screen_size
    self.height, self.width = self.screen_size, self.screen_size
    self.observation_shape = [
      self.height, self.width, color_channel]

    # location
    self.location_size = location_size
    self.location_shape = [self.location_size, self.location_size]

    self.prev_x, self.prev_y, self.prev_pressure = None, None, None

  @staticmethod
  def pretty_print_action(ac):
    for k, v in ColorEnv.actions_to_idx.items():
      print(k, ac[v])

  def random_action(self):
    return np.random.uniform(size=[len(self.actions_to_idx)])

  def reset(self):
    self.intermediate_images = []
    self.prev_x, self.prev_y, self.prev_pressure = None, None, None

    self.s = tiledsurface.Surface()
    self.s.flood_fill(0, 0, (255, 255, 255), (0, 0, 64, 64), 0, self.s)
    self.s.begin_atomic()

    with open(self.brush_path) as fp:
      self.bi = brush.BrushInfo(fp.read())
    self.b = brush.Brush(self.bi)

    # Two extra brushstrokes for MyPaint 2
    self.b.stroke_to(self.s.backend, 20, 20, 0.0, 0.0, 0.0, 0.1, 0, 0, 0)
    self.b.stroke_to(self.s.backend, 20, 20, 0.0, 0.0, 0.0, 0.1, 0, 0, 0)
    self.s.end_atomic()
    self.s.begin_atomic()

  def draw(self, ac, s=None, dtime=1):
    # Just added this
    if self.paint_mode == PaintMode.STROKES_ONLY:
      self.s.clear()
      self.s.flood_fill(0, 0, (255, 255, 255), (0, 0, 64, 64), 0, self.s)

      # Two extra brushstrokes for MyPaint 2
      self.b.stroke_to(self.s.backend, 20, 20, 0.0, 0.0, 0.0, 0.1, 0, 0, 0)
      self.b.stroke_to(self.s.backend, 20, 20, 0.0, 0.0, 0.0, 0.1, 0, 0, 0)
      self.s.end_atomic()
      self.s.begin_atomic()

    if s is None:
      s = self.s

    s_x, s_y = ac[self.actions_to_idx['start_x']] * 64, ac[self.actions_to_idx['start_y']] * 64
    e_x, e_y = ac[self.actions_to_idx['end_x']] * 64, ac[self.actions_to_idx['end_y']] * 64
    c_x, c_y = ac[self.actions_to_idx['control_x']] * 64, ac[self.actions_to_idx['control_y']] * 64
    color = (
      ac[self.actions_to_idx['color_r']],
      ac[self.actions_to_idx['color_g']],
      ac[self.actions_to_idx['color_b']],
    )
    pressure = ac[self.actions_to_idx['pressure']] * 0.8
    entry_pressure = ac[self.actions_to_idx['entry_pressure']] * 0.8
    size = ac[self.actions_to_idx['size']] * 2.

    if self.paint_mode == PaintMode.CONNECTED_STROKES:
      if self.prev_x is not None:
        s_x, s_y, entry_pressure = self.prev_x, self.prev_y, self.prev_pressure
      self.prev_x, self.prev_y, self.prev_pressure = e_x, e_y, pressure

    self.b.brushinfo.set_color_rgb(color)

    self.b.brushinfo.set_base_value('radius_logarithmic', size)

    # Move brush to starting point without leaving it on the canvas.
    self._stroke_to(s_x, s_y, 0)

    self._draw(s_x, s_y, e_x, e_y, c_x, c_y, entry_pressure, pressure, size, color, dtime)

  def _draw(self, s_x, s_y, e_x, e_y, c_x, c_y,
            entry_pressure, pressure, size, color, dtime):

    # if straight line or jump
    if pressure == 0:
      self.b.stroke_to(
        self.s.backend, e_x, e_y, pressure, 0, 0, dtime, 0, 0, 0)
    else:
      self.curve(c_x, c_y, s_x, s_y, e_x, e_y, entry_pressure, pressure)

    # Relieve brush pressure for next jump
    self._stroke_to(e_x, e_y, 0)

    self.s.end_atomic()
    self.s.begin_atomic()

  # sx, sy = starting point
  # ex, ey = end point
  # kx, ky = curve point from last line
  # lx, ly = last point from InteractionMode update
  def curve(self, cx, cy, sx, sy, ex, ey, entry_pressure, pressure):
    # entry_p, midpoint_p, junk, prange2, head, tail
    entry_p, midpoint_p, prange1, prange2, h, t = \
      self._line_settings(entry_pressure, pressure)

    points_in_curve = 100
    mx, my = midpoint(sx, sy, ex, ey)
    length, nx, ny = length_and_normal(mx, my, cx, cy)
    cx, cy = multiply_add(mx, my, nx, ny, length * 2)
    x1, y1 = difference(sx, sy, cx, cy)
    x2, y2 = difference(cx, cy, ex, ey)
    head = points_in_curve * h
    head_range = int(head) + 1
    tail = points_in_curve * t
    tail_range = int(tail) + 1
    tail_length = points_in_curve - tail

    # Beginning
    px, py = point_on_curve_1(1, cx, cy, sx, sy, x1, y1, x2, y2)
    length, nx, ny = length_and_normal(sx, sy, px, py)
    bx, by = multiply_add(sx, sy, nx, ny, 0.25)
    self._stroke_to(bx, by, entry_p)
    pressure = abs(1 / head * prange1 + entry_p)
    self._stroke_to(px, py, pressure)

    for i in range(2, head_range):
      px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
      pressure = abs(i / head * prange1 + entry_p)
      self._stroke_to(px, py, pressure)

    # Middle
    for i in range(head_range, tail_range):
      px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
      self._stroke_to(px, py, midpoint_p)

    # End
    for i in range(tail_range, points_in_curve + 1):
      px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
      pressure = abs((i - tail) / tail_length * prange2 + midpoint_p)
      self._stroke_to(px, py, pressure)

    return pressure

  def _stroke_to(self, x, y, pressure, duration=0.1):
    self.b.stroke_to(
      self.s.backend,
      x, y,
      pressure,
      0.0, 0.0,
      duration, 0, 0, 0)
    self.s.end_atomic()
    self.s.begin_atomic()
    self.intermediate_images.append(self.image)

  def save_image(self, path="test.png"):
    Image.fromarray(self.image.astype(np.uint8).squeeze()).save(path)
    # self.s.save_as_png(path, alpha=False)

  @property
  def image(self):
    rect = [0, 0, self.height, self.width]
    scanline_strips = \
      surface.scanline_strips_iter(self.s, rect)
    return next(scanline_strips)

  def _line_settings(self, entry_pressure, pressure):
    p1 = entry_pressure
    p2 = (entry_pressure + pressure) / 2
    p3 = pressure
    if self.head == 0.0001:
      p1 = p2
    prange1 = p2 - p1
    prange2 = p3 - p2
    return p1, p2, prange1, prange2, self.head, self.tail
