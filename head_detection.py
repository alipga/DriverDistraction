import numpy as np
import cv2 as cv
import imutils
from imutils import face_utils
import dlib

class headDetection:

    def __init__(self,landmark_model='shape_predictor_68_face_landmarks.dat'):
        self.landmark_detector = dlib.shape_predictor(landmark_model)
        self.face_detector = dlib.get_frontal_face_detector()
        self.model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner

                                ])

    def detect_face(self,image):
        gray = image#cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        faces = self.face_detector(gray,1)
        return faces

    def face_landmark(self,image,face):
        gray = image#cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        shape = self.landmark_detector(gray,face)
        shape = face_utils.shape_to_np(shape)
        return shape

    def camera_calibration(self,dist_coefs,focal_length,center):
        self.dist_coefs = dist_coefs
        self.camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )

    def eye_detector(self,image,face):
        shape = self.face_landmark(image,face)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[36:42]
        rightEye = shape[42:48]
        return (leftEye,rightEye)

    def eye_box(self,eye):
        top = eye[1][1]
        left = eye[0][0]
        bottom = eye[5][1]
        right = eye[3][0]
        width = right-left
        lenght = bottom-top
        return [left,top,width,lenght]




    def face_pose_detector(self,image,face):
        shape = self.face_landmark(image,face)
        image_points=np.array([shape[33]],dtype='double')
        image_points=np.append(image_points,[shape[8]],axis=0)
        image_points=np.append(image_points,[shape[36]],axis=0)
        image_points=np.append(image_points,[shape[45]],axis=0)
        image_points=np.append(image_points,[shape[48]],axis=0)
        image_points=np.append(image_points,[shape[54]],axis=0)
        (success, rotation_vector, translation_vector) = cv.solvePnP(self.model_points,image_points,self.camera_matrix, self.dist_coefs)
        return (rotation_vector, translation_vector)

    def draw_annotation_box(self,image, rotation_vector, translation_vector,color=(255, 255, 255), line_width=2):
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
                                      self.camera_matrix,
                                      self.dist_coefs)
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
