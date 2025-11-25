import time
import sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
print(ROOT)
sys.path.append(str(ROOT))
from wave_generator.base_wave_ctl import FY8300
from controller.Dualsense import DualSenseJoystick
dev = FY8300("/dev/tty.usbserial-1130")

def wave_init(w_generator):
    pass

def wait(sec:int):
    '''
    thread sleep for second(int)
    '''
    time.sleep(sec)
    
def wait_ms(sec:int):
    '''
    thread sleep for mili-second(int)
    '''
    ms = sec / 1000
    time.sleep(ms)
    

if __name__ == "__main__":
    l_phase_x = 0
    l_phase_y = 0
    l_phase_z = 0
    phase_x = 0
    phase_y = 0
    phase_z = 0
    
    def set_wave(x, y, z):
        global phase_x, phase_y, phase_z
        global l_phase_x, l_phase_y, l_phase_z
        '''
        func: 传入方向
        '''
        if y > 0:
            phase_y = 180
        else:
            phase_y = 0
            
        if x > 0:
            phase_x = 0
        else:
            phase_x = 180
        
        x = abs(x)
        y = abs(y)
        x = 5 * x/np.sqrt(x**2+y**2)
        y = 5 * y/np.sqrt(x**2+y**2)
        z = 5
        
        
        if phase_x != l_phase_x: 
            dev.set_phase(1, phase_x)
            wait_ms(200)
        dev.set_amplitude(1, x) # 幅度


        
        if phase_y != l_phase_y:  
            dev.set_phase(2, phase_y)
            wait_ms(200)
        dev.set_amplitude(2, y) # 幅度

        
        l_phase_x = phase_x
        l_phase_y = phase_y
        print("PHASE:",phase_x, phase_y)
        print("LAST PHASE",l_phase_x,l_phase_y)
    
    joy = DualSenseJoystick(filter=True, filter_para=0.6)
    
    # >>> Wave Generator Initialization
    # X-axis
    CH = 1
    dev.set_waveform(CH, 0)     # 正弦
    dev.set_freq(CH, 0.000001)      # 1 kHz
    dev.set_amplitude(CH, 1.000) # 幅度
    dev.set_phase(CH, 0.0)     # 相位
    dev.output_on(CH)
    
    # Y-axis
    CH = 2
    dev.set_waveform(CH, 0)     # 正弦
    dev.set_freq(CH, 0.000001)      # 1 kHz
    dev.set_amplitude(CH, 1.000) # 幅度
    dev.set_phase(CH, 0.0)     # 相位
    dev.output_on(CH)
   
    # Z-axis
    CH = 3
    dev.set_waveform(CH, 0)     # 正弦
    dev.set_freq(CH, 1)         # 1 kHz
    dev.set_amplitude(CH, 9.000) # 幅度
    dev.set_phase(CH, 90.0)     # 相位
    dev.output_on(CH)
     # <<< END Wave Generator Initialization
    
    try:
        while True:
            joy.update_controller()
            lx, ly = joy.get_left_stick()
            rx, ry = joy.get_right_stick()
            L2, R2 = joy.get_triggers()
            buttons = joy.get_buttons()

            print(f"Left=({lx:.2f}, {ly:.2f})  Right=({rx:.2f}, {ry:.2f})  L2={L2:.2f}  R2={R2:.2f}")
            print("Buttons:", buttons)

            set_wave(lx,ly,1)
            time.sleep(1)

    except KeyboardInterrupt:
        joy.close()
        print("退出程序")