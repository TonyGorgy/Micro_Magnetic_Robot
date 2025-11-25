import serial
import time

ser = serial.Serial("/dev/tty.usbserial-1130", 115200, timeout=0.1)

def send(cmd):
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    time.sleep(0.05)
    resp = ser.readline().decode(errors="ignore").strip()
    print(f"→ {cmd:<20} ← {resp}")
    return resp

print("\n====== 基础连通性检查（CH2 开关）======")
send("WFN0")     # CH2 OFF
send("WFN1")     # CH2 ON

print("\n====== CH2 波形（全部尝试）======")
send("WFW0")         # 文档规定格式
# send("WFW1")
# send("WMW0 2")        # 多通道格式
# send("WMB0")          # 老固件隐藏格式

# print("\n====== CH2 频率（全部尝试）======")
# send("WFF1000000000")    # 文档规定格式 (1kHz)
# send("WMF1000000000 2")  # 多通道格式
# send("WBF1000000000")    # 老前缀

# print("\n====== CH2 幅度（全部尝试）======")
send("WFA2.000")         # 文档格式
# send("WFA2.000")
# send("WMA1.000 2")       # 多通道格式
# send("WMB1.000")         # 老前缀

# print("\n====== CH2 相位（全部尝试）======")
send("WFP45.0")          # 文档格式
# send("WMP90.0 2")        # 多通道格式
# send("WBP90.0")          # 老前缀

print("\n====== 结束 ======")
