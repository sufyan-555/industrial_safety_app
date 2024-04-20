import math
import cv2
import numpy as numpy
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
def detectpose(image):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ =image.shape
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))
    return output_image,landmarks
  
def calculateangle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle +=360
    return angle

def alert(frame,flag=True):
    if not flag:
        return [False,[]]
    try:
        frame, landmarks = detectpose(frame)
        left_elbow_angle = calculateangle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
      
        left_shoulder_angle = calculateangle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
      
        right_elbow_angle = calculateangle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
      
        right_shoulder_angle = calculateangle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
      
        if left_elbow_angle > 150 and left_elbow_angle < 210 and right_elbow_angle > 150 and right_elbow_angle < 210:
            if left_shoulder_angle > 60 and left_shoulder_angle < 150 and right_shoulder_angle > 150 and right_shoulder_angle < 220:
                return [True,[]]
          
        return [False,[]]
    except:
       return [False,[]]




    
    
# """camera_video = cv2.VideoCapture(0)
# while camera_video.isOpened():
#   ok, frame = camera_video.read()
#   if not ok:
#     continue

#   frame = cv2.flip(frame, 1)
#   frame_height, frame_width, _ = frame.shape
#   frame = cv2.resize(frame,(int(frame_width * (640 / frame_height)), 640))
#   result=alert(frame=frame)
#   print(result)
#   cv2.imshow('Pose Classification', frame)

#   k = cv2.waitKey(1) & 0xFF
#   if(k == 27):
#     break

# camera_video.release()
#cv2.destroyAllWindows()"""