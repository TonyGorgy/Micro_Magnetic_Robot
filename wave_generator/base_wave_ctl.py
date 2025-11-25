import serial
import time
import math
import sys

class FY8300:
    """
    FY8300 Function Generator Controller (ASCII command line mode)
    The send, set_waveform, set_amplitude, set_amplitude, set_offset, set_phase, output_on, output_off has been tested
    """

    def __init__(self, port, baud=115200, timeout=0.1):
        try:
            self.ser = serial.Serial(port, baud, timeout=timeout)
        except serial.SerialException as e:
            print(f"[ERROR] 串口无法打开: \n {e}")
            # 你可以选择退出或重试
            sys.exit()

    def close(self):
        self.ser.close()

    # ---------------------------------
    # Basic I/O
    # ---------------------------------
    def send(self, cmd: str):
        """Send one command (LF terminated) and read one line back"""
        self.ser.write((cmd + "\n").encode())
        time.sleep(0.2)
        resp = self.ser.readline().decode(errors="ignore").strip()
        return resp

    # ---------------------------------
    # Waveform / Freq / Amp / Offset / Duty / Phase
    # ---------------------------------

    def set_waveform(self, ch: int, code: int):
        """
        Set waveform of channel ch.
        ch=1 -> WMW
        ch=2 -> WFW
        ch=3 -> TFW
        """
        wf_cmd = {1:"WMW", 2:"WFW", 3:"TFW"}[ch]
        return self.send(f"{wf_cmd}{code}")

    def set_freq(self, ch: int, hz: float):
        """
        Set frequency in Hz.
        FY8300 expects frequency in uHz in decimal.
        """
        f_cmd = {1:"WMF", 2:"WFF", 3:"TFF"}[ch]
        val_uHz = int(round(hz * 1_000_000))  # Hz -> uHz
        return self.send(f"{f_cmd}{val_uHz}")

    def set_amplitude(self, ch: int, vpp: float):
        """
        CH1: WMAxx.xxx
        CH2: WFAxx.xxx
        CH3: TFAxx.xxx
        """
        a_cmd = {1:"WMA", 2:"WFA", 3:"TFA"}[ch]
        return self.send(f"{a_cmd}{vpp:.3f}")

    def set_offset(self, ch: int, voff: float):
        """
        CH1: WMO xx.xxx
        CH2: WFO xx.xxx
        CH3: TFO xx.xxx
        """
        o_cmd = {1:"WMO", 2:"WFO", 3:"TFO"}[ch]
        return self.send(f"{o_cmd} {voff:.3f}")

    def set_duty(self, ch: int, percent: float):
        """
        Duty cycle in percent.
        CH1: WMDxx.x
        CH2: WFDxx.x
        CH3: TFDxx.x
        """
        d_cmd = {1:"WMD", 2:"WFD", 3:"TFD"}[ch]
        return self.send(f"{d_cmd}{percent:.1f}")

    def set_phase(self, ch: int, deg: float):
        """
        Set phase in degrees.
        CH1: WMPxx.x
        CH2: WFPxx.x
        CH3: TFPxx.x
        """
        p_cmd = {1:"WMP", 2:"WFP", 3:"TFP"}[ch]
        deg = max(0.0, min(359.9, deg))  # clamp range
        return self.send(f"{p_cmd}{deg:.1f}")

    # ---------------------------------
    # Output Enable/Disable
    # ---------------------------------
    def output_on(self, ch: int):
        """
        CH1: WMN1
        CH2: WFN1
        CH3: TFN1
        """
        n_cmd = {1:"WMN", 2:"WFN", 3:"TFN"}[ch]
        return self.send(f"{n_cmd}1")

    def output_off(self, ch: int):
        """
        CH1: WMN0
        CH2: WFN0
        CH3: TFN0
        """
        n_cmd = {1:"WMN", 2:"WFN", 3:"TFN"}[ch]
        return self.send(f"{n_cmd}0")

    # ---------------------------------
    # Query
    # ---------------------------------
    def read_freq(self, ch: int):
        r_cmd = {1:"RMF", 2:"RFF", 3:"RTF"}[ch]
        return self.send(r_cmd)

    # ---------------------------------
    # Arbitrary Waveform Upload (legacy compatible)
    # ---------------------------------
    @staticmethod
    def i2hex(val: int) -> str:
        return f"{int(val) & 0xFFFFFFFF:08X}"

    def select_arb(self, ch: int, arb_id: int):
        return self.send(f"WMAF {ch:02d} {arb_id:02d}")

    def upload_arb_point(self, ch: int, index: int, value: int):
        return self.send(f"WMAD {ch:02d} {index:04X} {value:04X}")

    def upload_arb_array(self, ch, arr):
        self.select_arb(ch, 1)
        for i, v in enumerate(arr):
            self.upload_arb_point(ch, i, v)
        # Arbitrary waveform code 36 (slot 1)
        self.set_waveform(ch, 36)

    def upload_arb_sine(self, ch):
        arr = []
        for i in range(1024):
            val = int((math.sin(2 * math.pi * i / 1024) + 1) * 2047.5)
            arr.append(val)
        self.upload_arb_array(ch, arr)


# DEV LOG
# dev = FY8300("/dev/tty.usbserial-1130")

# # 读取频率波特率偏低
# # 需要手动延迟防止寄存器数据写入丢失（在该条命令之后）
# # 或者减少写入指令（不要连续多种参数一起修改）

# # >>> main function test
# CH = 1
# dev.set_waveform(CH, 0)     # 正弦
# time.sleep(0.1)
# dev.set_freq(CH, 0.00002)      # 1 kHz
# time.sleep(0.1)
# dev.set_amplitude(CH, 3.000) # 幅度
# time.sleep(0.1)
# dev.set_phase(CH, 0.0)     # 相位
# time.sleep(0.1)
# dev.output_on(CH)
# # >>> continuious test
# # for i in range(1,51):
# #     amp = 1 + 0.02 * i
# #     dev.set_amplitude(1, amp) # 幅度
#     # time.sleep(0.01)
# time.sleep(.5)


# CH = 2
# time.sleep(0.1)
# dev.set_waveform(CH, 1)     # 正弦
# time.sleep(0.1)
# dev.set_freq(CH, 0.000002)      # 1 kHz
# time.sleep(0.1)
# dev.set_amplitude(CH, 4) # 幅度
# time.sleep(0.1)
# dev.set_phase(CH, 35.0)     # 相位
# time.sleep(0.1)
# dev.output_on(CH)
# # >>> continuious test
# # for i in range(1,51):
# #     amp = 1 + 0.02 * i
# #     dev.set_amplitude(1, amp) # 幅度
#     # time.sleep(0.01)
# time.sleep(.5)

# CH = 3
# time.sleep(0.1)
# dev.set_waveform(CH, 0)     # 正弦
# time.sleep(0.1)
# dev.set_freq(CH, 2)      # 1 kHz
# time.sleep(0.1)
# dev.set_amplitude(CH, 4.000) # 幅度
# time.sleep(0.2)             # 0.2 才不会被忽略，0.1 都会
# dev.set_phase(CH, 40.0)     # 相位
# time.sleep(0.1)
# dev.output_on(CH)
# # >>> continuious test
# # for i in range(1,51):
# #     amp = 1 + 0.02 * i
# #     dev.set_amplitude(1, amp) # 幅度
#     # time.sleep(0.01)
    
# # 一个例子
# # 这里不暂停的话 2 就又不能关闭了
# # 而且如果sleep 0.1 也不能控制，应该是控制模式切换会更耗时
# # 谁夹在中间谁的指令会更容易被忽略

# time.sleep(1)

# dev.output_off(1)
# dev.output_off(3)
# time.sleep(0.1)

# dev.output_off(2)
# time.sleep(0.1)

# dev.output_off(3)
# time.sleep(0.1)
# # <<< end test