import cv2
import numpy as np

def build_charuco_board():
    """
    构造与你的 PDF/PNG 一样的 Charuco 板：
    6x6 格子，每格 16mm，marker 12mm
    """
    squares_x = 6
    squares_y = 6
    square_length = 15.5   # mm
    marker_length = 11.04   # mm

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

    board = cv2.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=aruco_dict
    )
    return board, aruco_dict


def charuco_calibration(
        camera_id=0,
        min_corners_per_frame=15,   # 每帧至少 15 个 charuco 点
        min_frames=12               # 至少 12 帧才标定
):
    board, aruco_dict = build_charuco_board()

    # 新 API 的 Aruco Detector
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    all_charuco_corners = []
    all_charuco_ids = []
    img_size = None

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("❌ 摄像头无法打开")
        return

    print("---------------------------------------------")
    print("  Charuco 自动标定启动")
    print("---------------------------------------------")
    print("[s] 保存当前帧（角点足够才保存）")
    print("[c] 执行标定")
    print("[q] 退出")
    print("---------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取图像")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        # 1) ArUco detection
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

            # 2) Charuco corner interpolation
            num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=marker_corners,
                markerIds=marker_ids,
                image=gray,
                board=board
            )

            if num > 0:
                cv2.aruco.drawDetectedCornersCharuco(
                    frame, charuco_corners, charuco_ids
                )
                cv2.putText(frame, f"Charuco corners: {num}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
        else:
            num = 0

        # 已经保存的帧数
        cv2.putText(frame, f"Saved frames: {len(all_charuco_corners)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)

        cv2.imshow("Charuco Calibration (New API)", frame)
        key = cv2.waitKey(1) & 0xFF

        # -------- Controls --------
        if key == ord('q'):
            print("退出程序")
            break

        elif key == ord('s'):
            if num >= min_corners_per_frame:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                print(f"✓ 已保存帧 {len(all_charuco_corners)}")
            else:
                print(f"⚠ 角点不足（{num} < {min_corners_per_frame}），未保存")

        elif key == ord('c'):
            if len(all_charuco_corners) < min_frames:
                print(f"⚠ 帧数不够（{len(all_charuco_corners)} < {min_frames}）")
                continue

            print("\n---------------------------------------------")
            print("           🔧 开始相机标定（Charuco）          ")
            print("---------------------------------------------")
            print(f"图像尺寸：{img_size}")

            rms, camera_matrix, dist_coeffs, rvecs, tvecs = \
                cv2.aruco.calibrateCameraCharuco(
                    charucoCorners=all_charuco_corners,
                    charucoIds=all_charuco_ids,
                    board=board,
                    imageSize=img_size,
                    cameraMatrix=None,
                    distCoeffs=None
                )

            print(f"\n▶ RMS 重投影误差：{rms}")
            print("\n▶ 相机内参矩阵 K：\n", camera_matrix)
            print("\n▶ 畸变系数 D：\n", dist_coeffs)

            # 保存 YAML
            fs = cv2.FileStorage("charuco_camera.yaml", cv2.FILE_STORAGE_WRITE)
            fs.write("camera_matrix", camera_matrix)
            fs.write("dist_coeffs", dist_coeffs)
            fs.write("image_width", img_size[0])
            fs.write("image_height", img_size[1])
            fs.release()

            print("\n📁 标定结果已写入：charuco_camera.yaml")
            print("---------------------------------------------\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    charuco_calibration()
