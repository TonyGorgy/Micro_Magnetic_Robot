import pygame
import sys
import time

class DualSenseJoystick:
    def __init__(self, filter=False, filter_para=0.8):
        pygame.init()
        pygame.joystick.init()
        self.filter = filter
        self.filter_para = filter_para
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
        print("按键数量:", self.js.get_numbuttons())
        print("轴数量:", self.js.get_numaxes())

    def update_controller(self):
        """刷新内部事件，每帧调用"""
        pygame.event.pump()

    def get_left_stick(self):
        lx = self.js.get_axis(0)
        ly = self.js.get_axis(1)
        if self.filter and self.last_lx != 0:
            lx = self.filter_para * lx + (1-self.filter_para) * self.last_lx
        if self.filter and self.last_ly != 0:
            ly = self.filter_para * ly + (1-self.filter_para) * self.last_ly
        return lx, -ly

    def get_right_stick(self):
        rx = self.js.get_axis(2)
        ry = self.js.get_axis(3)
        if self.filter and self.last_rx != 0:
            rx = self.filter_para * rx + (1-self.filter_para) * self.last_rx
        if self.filter and self.last_ry != 0:
            ry = self.filter_para * ry + (1-self.filter_para) * self.last_ry
        return rx, -ry

    def get_buttons(self):
        """返回所有按钮状态，列表形式 [0/1, 0/1, ...]"""
        btn_count = self.js.get_numbuttons()
        return [self.js.get_button(i) for i in range(btn_count)]

    def get_button(self, button_id):
        """读取特定按钮（例如 0=X 1=○ 2=□ 3=△）"""
        return self.js.get_button(button_id)

    def get_triggers(self):
        """(L2, R2)"""
        try:
            L2 = self.js.get_axis(4)  
            R2 = self.js.get_axis(5)
        except:
            L2 = 0
            R2 = 0
        return L2, R2

    def close(self):
        pygame.joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    joy = DualSenseJoystick(filter=True)
    print("开始读取输入，Ctrl+C 退出\n")
    try:
        while True:
            joy.update_controller()
            lx, ly = joy.get_left_stick()
            rx, ry = joy.get_right_stick()
            L2, R2 = joy.get_triggers()
            buttons = joy.get_buttons()

            print(f"Left=({lx:.2f}, {ly:.2f})  Right=({rx:.2f}, {ry:.2f})  L2={L2:.2f}  R2={R2:.2f}")
            print("Buttons:", buttons)

            time.sleep(0.05)

    except KeyboardInterrupt:
        joy.close()
        print("退出程序")
