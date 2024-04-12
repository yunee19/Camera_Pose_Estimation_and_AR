import numpy as np
import cv2 as cv

def main():
    # Camera Calibration
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane.

    cap = cv.VideoCapture(0)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Define criteria here

    camera_matrix = None  # Initialize camera_matrix and dist_coeff
    dist_coeff = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (7,6), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria=criteria)  # Pass criteria here
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(frame, (7,6), corners2, ret)

            if len(objpoints) == 10: # Calibrate camera after capturing 10 frames
                ret, camera_matrix, dist_coeff, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                break

        cv.imshow('Camera Calibration', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    # Camera Pose Estimation
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (7,6), None)

        if ret == True:
            ret, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners, camera_matrix, dist_coeff)

            # AR Object Visualization
            axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coeff)

            frame = draw(frame, corners, imgpts)

        cv.imshow('AR Object Visualization', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def draw(img, corners, imgpts):
    # Remove extra dimensions from the array
    imgpts = np.squeeze(imgpts)

    # Convert 2D coordinates from float to int
    imgpts = np.round(imgpts).astype(int)

    # Draw lines from the corner of the chessboard to the drawing points
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img
    
if __name__ == "__main__":
    main()
