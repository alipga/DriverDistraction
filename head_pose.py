import dlib
import cv2 as cv
import numpy as np
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
from stabilizer import Stabilizer

def draw_annotation_box(image, rotation_vector, translation_vector,camera_matrix,dist_coeefs,color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    rear_size = 200
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    front_size = 600
    front_depth = 500
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
        # Map to 2d image points
    (point_2d, _) = cv.projectPoints(point_3d,
                                  rotation_vector,
                                  translation_vector,
                                  camera_matrix,
                                  dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    # Draw all the lines
    cv.polylines(image, [point_2d], True, color, line_width, cv.LINE_AA)
    cv.line(image, tuple(point_2d[1]), tuple(
    point_2d[6]), color, line_width, cv.LINE_AA)
    cv.line(image, tuple(point_2d[2]), tuple(
    point_2d[7]), color, line_width, cv.LINE_AA)
    cv.line(image, tuple(point_2d[3]), tuple(
    point_2d[8]), color, line_width, cv.LINE_AA)
    return image

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0],eye[3])
    EAR = (A+B)/(2.0*C)
    return EAR


pose_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1) for _ in range(6)]

face_detector = dlib.get_frontal_face_detector()
head_landmard_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


#3D model for human facial landamrk
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner

                        ])
dist_coeffs = np.zeros((4,1))

#read the video stream from webcam
cap = cv.VideoCapture(0)
ret , frame = cap.read()
frame = imutils.resize(frame,width=500)

size = frame.shape
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 1
eye_counter = 0
blink_counter = 0


while True:
    ret , frame = cap.read()
    frame = imutils.resize(frame,width=500)
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    faces = face_detector(gray_frame,1)
    for (i,face) in enumerate(faces):
        shape = head_landmard_detector(gray_frame,face)
        shape = face_utils.shape_to_np(shape)

        image_points=np.array([shape[33]],dtype='double')
        image_points=np.append(image_points,[shape[8]],axis=0)
        image_points=np.append(image_points,[shape[36]],axis=0)
        image_points=np.append(image_points,[shape[45]],axis=0)
        image_points=np.append(image_points,[shape[48]],axis=0)
        image_points=np.append(image_points,[shape[54]],axis=0)
        (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        #print("Rotation Vector:\n {0}".format(rotation_vector))
        #print ("Translation Vector:\n {0}".format(translation_vector))
        pose = (rotation_vector,translation_vector)
        stabile_pose = []
        pose_np = np.array(pose).flatten()
        for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                stabile_pose.append(ps_stb.state[0])
        stabile_pose = np.reshape(stabile_pose, (-1, 3))
        #uncomment the following line to get stablized pose estimation
        #draw_annotation_box(frame, stabile_pose[0], stabile_pose[1],camera_matrix,dist_coeffs)
        frame = draw_annotation_box(frame, pose[0], pose[1],camera_matrix,dist_coeffs)

        #################################
        #################################
        #Blinks
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            eye_counter += 1
        else:
            if eye_counter > EYE_AR_CONSEC_FRAMES:
                blink_counter += 1
            eye_counter = 0

        cv.putText(frame, "Blinks: {}".format(blink_counter), (10, 30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ####################################
        ####################################
        ####################################

    cv.imshow('frame',frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
