import pygame
import sys
import time

class DualSenseJoystick:
    def __init__(self, filter=False):
        pygame.init()
        pygame.joystick.init()
        self.filter = filter
        self.filter_para = 0.8
        self.last_lx = 0
        self.last_ly = 0
        self.last_rx = 0
        self.last_ry = 0

        # 检查是否连接手柄
        if pygame.joystick.get_count() == 0:
            print("未检测到手柄，请检查是否连接")
            sys.exit(1)

        # 默认使用第 1 个手柄
        self.js = pygame.joystick.Joystick(0)
        self.js.init()

        print("检测到手柄:", self.js.get_name())

    def update_controller(self):
        """刷新内部事件，每帧调用"""
        pygame.event.pump()

    def get_left_stick(self):
        """返回 (lx, ly) 范围为 -1.0 ~ 1.0"""
        lx = self.js.get_axis(0)
        ly = self.js.get_axis(1)
        if self.filter == True and self.last_lx != 0:
            lx = self.filter_para * lx + (1-self.filter_para) * self.last_lx
        if self.filter == True and self.last_ly != 0:
            ly = self.filter_para * ly + (1-self.filter_para) * self.last_ly
        return lx, -ly

    def get_right_stick(self):
        """返回 (rx, ry) 范围为 -1.0 ~ 1.0"""
        rx = self.js.get_axis(2)
        ry = self.js.get_axis(3)
        if self.filter == True and self.last_rx != 0:
            rx = self.filter_para * rx + (1-self.filter_para) * self.last_rx
        if self.filter == True and self.last_ry != 0:
            ry = self.filter_para * ry + (1-self.filter_para) * self.last_ry
        return rx, -ry

    def close(self):
        """关闭手柄（可选）"""
        pygame.joystick.quit()
        pygame.quit()


# ------------- 使用示例 -------------
if __name__ == "__main__":
    joy = DualSenseJoystick(filter=True)

    print("开始读取摇杆位置，按 Ctrl+C 退出\n")

    try:
        while True:
            joy.update_controller()
            lx, ly = joy.get_left_stick()
            rx, ry = joy.get_right_stick()
            print(f"Left=({lx:.2f}, {ly:.2f})   Right=({rx:.2f}, {ry:.2f})")
            time.sleep(0.02)  # 50Hz
    except KeyboardInterrupt:
        joy.close()
        print("退出程序")
