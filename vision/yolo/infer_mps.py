# realtime_cam_infer.py
# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from vision.calibration.base import Calib


def main():
    vision_dir = Path(__file__).resolve().parent.parent
    calib_path = vision_dir / "calibration"
    model_path = vision_dir / "yolo" / "models" / "best.mlpackage"

    print("使用模型:", model_path)

    # 加载相机标定
    calibration = Calib(
        kd_path=calib_path / "configs" / "charuco_camera.yaml",
        ext_path=calib_path / "configs" / "plane_extrinsics.yaml"
    )

    device = "mps"  # Mac M 系列硬件加速

    # 加载 YOLO
    model = YOLO(model_path)
    # model.to(device)
    try:
        model.fuse()   # Conv+BN 融合，有些版本不支持
        print("[info] Model fused for faster inference.")
    except:
        print("[info] Fuse not supported, skipping.")

    # 打开摄像头（保持你的原分辨率，不做修改）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 读取一帧确定分辨率 → 生成畸变映射表
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败")
        return

    h, w = frame.shape[:2]
    print(f"[info] Camera resolution: {w} x {h}")

    # ===== 预计算畸变映射（避免每帧 undistort）=====
    map1, map2 = cv2.initUndistortRectifyMap(
        calibration.K, calibration.D,
        None, calibration.K,
        (w, h),
        cv2.CV_16SC2
    )

    print("\n" + ">" * 20 + " START INFERENCE " + "<" * 20)
    print("[info] Device:", device)

    # 平滑记录
    l_Xw, l_Yw = 0.0, 0.0
    alpha = 0.8

    # 帧计数（每 2 帧推理一次）
    frame_id = 0
    last_results = None

    # ===========================================
    # 主循环
    # ===========================================
    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            break

        # remap → 快速矫正畸变
        undist_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        frame_id += 1

        # =======================================
        # 每 2 帧推理一次（提升约 30–50% FPS）
        # =======================================
        if frame_id % 2 == 0 or last_results is None:
            results = model.predict(
                undist_frame,
                imgsz=960,
                conf=0.5,
                device=device,
                verbose=False
            )
            last_results = results[0]

        results = last_results
        annotated_frame = undist_frame.copy()

        # =======================================
        # 提取检测框 + 世界坐标
        # =======================================
        if results is not None and results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # 手动绘制矩形框（比 plot() 快很多）
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 中心点
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2

                # 像素 → 世界坐标
                world_pos = calibration.pixel_to_world(u, v)
                if world_pos is not None:
                    Xw, Yw = world_pos

                    # 低通滤波平滑
                    Xw = alpha * Xw + (1 - alpha) * l_Xw
                    Yw = alpha * Yw + (1 - alpha) * l_Yw
                    l_Xw, l_Yw = Xw, Yw

                    cv2.circle(annotated_frame, (u, v), 5, (0, 255, 0), -1)
                    cv2.putText(
                        annotated_frame,
                        f"{Xw:.1f}, {Yw:.1f} mm",
                        (u, v + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

        # =======================================
        # FPS 显示
        # =======================================
        fps = 1.0 / (time.time() - start)
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # 显示画面
        cv2.imshow("YOLO Micro Robot Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n>>> END <<<\n")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
