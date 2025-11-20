import serial
import time
import math

class FY8300:
    """
    FY8300 Function Generator Controller (ASCII command line mode)
    The send, set_waveform, set_amplitude, set_amplitude, set_offset, set_phase, output_on, output_off has been tested
    """

    def __init__(self, port, baud=115200, timeout=0.1):
        self.ser = serial.Serial(port, baud, timeout=timeout)

    def close(self):
        self.ser.close()

    # ---------------------------------
    # Basic I/O
    # ---------------------------------
    def send(self, cmd: str):
        """Send one command (LF terminated) and read one line back"""
        self.ser.write((cmd + "\n").encode())
        time.sleep(0.03)
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

dev = FY8300("/dev/tty.usbserial-1130")

# >>> main function test
dev.set_waveform(1, 0)     # 正弦
dev.set_freq(1, 1000)      # 1 kHz
dev.set_amplitude(1, 1.000) # 幅度
dev.set_phase(1, 45.0)     # 相位
dev.output_on(1)
time.sleep(1)
dev.output_off(1)
# <<< end test