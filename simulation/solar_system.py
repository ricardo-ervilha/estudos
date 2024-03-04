import pygame
import math
pygame.init()

#Window size
WIDTH, HEIGHT = 800, 800
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

#Caption
pygame.display.set_caption("Planet Simulation")

#Colors
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
RED = (188, 39, 50)
DARK_GREY = (80, 78, 81)
ORANGE = (227, 158, 28)

#Add font
FONT = pygame.font.SysFont("comicsans", 16)

#Astronomical Object like Star or Planet
class AstronomicalObject:
  #astronomical unit
  AU = 149.6e6 * 1000
  
  #gravitation constant
  G = 6.67428e-11

  SCALE = 250 / AU # 1 AU = 100 pixels

  TIMESTEP = 3600 * 24 # 1 day

  def __init__(self, x, y, radius, color, mass, typeObject):
    self.x = x
    self.y = y
    self.radius = radius
    self.color = color
    self.mass = mass
    self.typeObject = typeObject # 0: sun, 1: planet

    self.orbit = []
    self.distance_to_sun = 0

    self.x_vel = 0
    self.y_vel = 0

  def draw(self, win):
    x = self.x * self.SCALE + WIDTH/2
    y = self.y * self.SCALE + HEIGHT/2

    if len(self.orbit) > 2:
      updated_points = []

      for point in self.orbit:
        x, y = point
        x = x * self.SCALE + WIDTH/2
        y = y * self.SCALE + HEIGHT/2
        updated_points.append((x,y))

      pygame.draw.lines(win, self.color, False, updated_points, 2)

    pygame.draw.circle(win, self.color, (x,y), self.radius)

    if self.typeObject != 0:
      distance_text = FONT.render(f"{round(self.distance_to_sun/1000, 1)}km", 1, WHITE)
      win.blit(distance_text, (x - distance_text.get_width() / 2 , y - distance_text.get_height() / 2))
  
  def attraction(self, other):
    other_x, other_y = other.x, other.y
    distance_x = other_x - self.x
    distance_y = other_y - self.y
    distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

    if other.typeObject == 0:
      self.distance_to_sun = distance

    force = ( self.G * self.mass * other.mass ) / ( distance ** 2 )
    theta = math.atan2(distance_y, distance_x)
    force_x = force * math.cos(theta)
    force_y = force * math.sin(theta)

    return force_x, force_y
  
  def update_position(self, objects):
    total_fx = total_fy = 0
    for object in objects:
      if object != self:
        fx, fy = self.attraction(object)
        total_fx += fx
        total_fy += fy
    
    self.x_vel += total_fx / self.mass * self.TIMESTEP
    self.y_vel += total_fy / self.mass * self.TIMESTEP

    self.x += self.x_vel * self.TIMESTEP
    self.y += self.y_vel * self.TIMESTEP

    self.orbit.append((self.x, self.y))


#Objects 
sun = AstronomicalObject(0, 0, 30, YELLOW, 1.98892 * 10**30, 0)
mercury = AstronomicalObject(0.387 * AstronomicalObject.AU, 0, 8, DARK_GREY, 3.3 * 10**23, 1)
venus = AstronomicalObject(0.723 * AstronomicalObject.AU, 0, 14, ORANGE, 4.8685 * 10**24, 1)
earth = AstronomicalObject(-1 * AstronomicalObject.AU, 0, 16, BLUE, 5.9742 * 10**24, 1)
mars = AstronomicalObject(-1.524 * AstronomicalObject.AU, 0, 12, RED, 6.39 * 10**23, 1)

mercury.y_vel = -47.4 * 1000
venus.y_vel = -35.02 * 1000
earth.y_vel  = 29.783 * 1000
mars.y_vel = 24.077 * 1000

objects = [sun, earth, mars, mercury, venus]

#Event loop
def main():
  run = True
  clock = pygame.time.Clock()

  while run:
    clock.tick(60)
    WINDOW.fill((0, 0, 0))

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        run = False

    for object in objects:
      object.update_position(objects)
      object.draw(WINDOW)

    pygame.display.update()

  pygame.quit()


main()