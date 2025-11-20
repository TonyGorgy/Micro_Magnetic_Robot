import cv2
import numpy as np

# ====== 配置参数 ======
board_size_mm = 40          # 整块板子 40mm x 40mm
dpi = 600                   # 打印分辨率 600 DPI，保证小尺寸清晰
squares_x = 4               # 横向 4 格
squares_y = 4               # 纵向 4 格

# 每个方格 / marker 的实际尺寸（单位：米）
squareLength_m = 0.01       # 10mm
markerLength_m = 0.007      # 7mm

# ====== 计算输出图像像素大小 ======
board_size_inch = board_size_mm / 25.4
pixels = int(board_size_inch * dpi)

print(f"输出图像大小: {pixels} x {pixels} 像素 (约 {board_size_mm}mm @ {dpi}DPI)")

# ====== 创建 ChArUco 棋盘 ======
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

board = cv2.aruco.CharucoBoard_create(
    squares_x,
    squares_y,
    squareLength_m,
    markerLength_m,
    aruco_dict
)

# 生成纯黑白的板子图像
img = board.generateImage((pixels, pixels), marginSize=0, borderBits=1)

output_name = "charuco_40mm_4x4_600dpi.png"
cv2.imwrite(output_name, img)
print("已保存:", output_name)
