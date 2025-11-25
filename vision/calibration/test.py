import cv2
import numpy as np


# --------------------------------------------
# 加载相机参数
# --------------------------------------------
def load_camera_parameters(path="charuco_camera.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeffs").mat()
    fs.release()
    return K, D


# --------------------------------------------
# 构造 Charuco 板 —— 必须与打印版参数一致
# --------------------------------------------
def build_charuco_board():
    squares_x = 6
    squares_y = 6
    square_length = 16.0     # mm
    marker_length = 12.0     # mm

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_5X5_100
    )
    board = cv2.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=aruco_dict
    )
    return board, aruco_dict


# --------------------------------------------
# 主程序：只验证姿态是否稳定
# --------------------------------------------
def verify():
    K, D = load_camera_parameters()
    board, aruco_dict = build_charuco_board()

    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        cv2.aruco.DetectorParameters()
    )

    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        undist = cv2.undistort(frame, K, D)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        # 1) Detect markers
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        if marker_ids is None:
            cv2.imshow("verify", undist)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # 2) Charuco refine
        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )

        if num is None or num < 10:
            cv2.imshow("verify", undist)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # 3) 估计 Charuco 姿态（rvec, tvec）
        rvec_guess = np.zeros((3,1), dtype=np.float32)
        tvec_guess = np.zeros((3,1), dtype=np.float32)

        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            board,
            K,
            D,
            rvec_guess,
            tvec_guess,
            False
        )

        if ok:
            cv2.aruco.drawDetectedCornersCharuco(undist, charuco_corners, charuco_ids)
            cv2.drawFrameAxes(undist, K, D, rvec, tvec, 30)  # 30mm 坐标轴

        cv2.imshow("verify", undist)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    verify()
