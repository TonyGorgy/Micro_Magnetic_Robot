import minimalmodbus
import serial
import time

class PowerSupply:
    """
    博瑞程控电源（Modbus-RTU）控制类
    """

    def __init__(self, port, slave_id=1, baudrate=9600, timeout=1):
        self.instrument = minimalmodbus.Instrument(port, slave_id)

        # 串口参数设置（按设备协议）
        self.instrument.serial.baudrate = baudrate
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity = serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout = timeout

        self.instrument.mode = minimalmodbus.MODE_RTU
        time.sleep(0.2)

        # 切换到远程控制模式，防止写入无效
        self.set_remote_mode(True)

    # ————————————
    # 寄存器地址映射
    # ————————————
    REG_REMOTE = 0x0000  # 远程模式
    REG_VSET   = 0x0001  # 设定电压 FLOAT
    REG_ASET   = 0x0003  # 设定电流 FLOAT
    REG_OUTPUT = 0x001B  # 输出开/关
    REG_VOUT   = 0x001D  # 实际输出电压 FLOAT
    REG_AOUT   = 0x001F  # 实际输出电流 FLOAT

    # ————————————
    # 设备控制方法
    # ————————————
    def set_remote_mode(self, enable=True):
        self.instrument.write_register(self.REG_REMOTE, 1 if enable else 0, functioncode=6)

    def set_voltage(self, voltage):
        self.instrument.write_float(self.REG_VSET, float(voltage), functioncode=16)

    def set_current(self, current):
        self.instrument.write_float(self.REG_ASET, float(current), functioncode=16)

    def output_on(self):
        self.instrument.write_register(self.REG_OUTPUT, 1, functioncode=6)

    def output_off(self):
        self.instrument.write_register(self.REG_OUTPUT, 0, functioncode=6)

    def read_voltage(self):
        return self.instrument.read_float(self.REG_VOUT, functioncode=3, number_of_registers=2)

    def read_current(self):
        return self.instrument.read_float(self.REG_AOUT, functioncode=3, number_of_registers=2)

    def read_output_state(self):
        return self.instrument.read_register(self.REG_OUTPUT, functioncode=3)

    def close(self):
        """关闭串口"""
        try:
            self.instrument.serial.close()
        except:
            print("Failed to close the serial port.")