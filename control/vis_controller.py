import pygame
import sys

pygame.init()

# 初始化手柄
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("未检测到手柄，请检查是否连接")
    sys.exit(1)

js = pygame.joystick.Joystick(0)
js.init()
print("检测到手柄:", js.get_name())

# 创建可视化窗口
WIDTH, HEIGHT = 600, 300
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PS5 摇杆可视化 Demo")

clock = pygame.time.Clock()

def draw_stick(surface, x, y, pos, color):
    """ 绘制摇杆位置 """
    cx, cy = x, y
    dx = pos[0] * 80
    dy = pos[1] * 80

    pygame.draw.circle(surface, (200,200,200), (cx, cy), 90, 2)
    pygame.draw.circle(surface, color, (cx + int(dx), cy + int(dy)), 10)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.event.pump()

    # 获取摇杆值（范围 -1.0 到 1.0）
    lx = js.get_axis(0)
    ly = js.get_axis(1)

    rx = js.get_axis(2)
    ry = js.get_axis(3)

    window.fill((30, 30, 30))

    # 左摇杆圆圈与点
    draw_stick(window, 150, 150, (lx, ly), (0, 200, 255))

    # 右摇杆圆圈与点
    draw_stick(window, 450, 150, (rx, ry), (255, 120, 0))

    # 文本
    font = pygame.font.SysFont(None, 26)
    text1 = font.render(f"Left Stick : ({lx:.2f}, {ly:.2f})", True, (255,255,255))
    text2 = font.render(f"Right Stick: ({rx:.2f}, {ry:.2f})", True, (255,255,255))
    window.blit(text1, (100, 20))
    window.blit(text2, (350, 20))

    pygame.display.flip()
    clock.tick(60)
