# train_yolo11_robot.py
# -*- coding: utf-8 -*-
"""
Train YOLO11 model for Micro Robot Detection
Author: MX
"""

import torch
from ultralytics import YOLO
import os
from pathlib import Path

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    print("====================================")
    print("   YOLO11 Micro-Robot Training")
    print("====================================")
    cur_path = Path(__file__).resolve().parent
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # ==========================
    # 加载 YOLO11 预训练模型
    # ==========================
    print("[INFO] Loading YOLO11m pretrained model ...")
    model_path = cur_path/ "models"/ "yolo11m.pt"
    model = YOLO(model_path)
    print("[INFO] Model loaded successfully")
    config_path = cur_path/ "micro.yaml"
    # ==========================
    # 开始训练
    # ==========================
    model.train(
        data= config_path,    # 数据配置
        epochs=100,              # 训练轮数
        imgsz=960,               # 输入图尺寸（提高小目标检测效果）
        batch=8,                 # 批大小
        device=device,           # 使用 GPU/MPS/CPU
        workers=4,               # 数据加载线程
        optimizer="Adam",         # 优化器，可改 Adam
        patience=20,             # 早停 patience
        save=True,               # 保存模型
        name="robot_yolo11_train" # 输出目录 runs/detect/robot_yolo11_train
    )

    print("\n====================================")
    print("训练结束！结果保存在：")
    print("runs/detect/robot_yolo11_train/")
    print("最重要的文件：weights/best.pt (最佳检测模型)")
    print("====================================\n")


if __name__ == "__main__":
    main()
