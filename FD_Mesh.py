#얼굴 감지와 3d 랜드마크 감지하는 파일

import numpy as np
import cv2
import mediapipe as mp
from detectFaceMeshImage import draw_face_mesh
from keras.models import load_model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection # 얼굴 검출을 위한 face_detection 모듈을 사용
img_size = 128

COLOR = (255,255,0) # BGR순서임 : Yellow
THICKNESS = 3 # 두께

#input 파일 리스트 이름 불러오기 (모델이 예측한 값의 라벨이 필요하기 때문)
import os
data_path = './Input/CK+48'
file_list = os.listdir(data_path)
file_list.remove('.DS_Store')
print("file list:",file_list)

#argmax방식으로 리턴    
def getLabel_ByArgmax(preds):
    index = np.argmax(preds[0])
    return file_list[index]
     
#학습 시킨 모델 불러오기 현재 weights_best_emotion_detect5 모델이 가장 인식률이 좋음.

emotion_detect_model = load_model('face_mesh_image_detect_emotion_model1.h5')

#영상 위에 투명도 정보가 존재하는 이미지 그리기
def overlay(image, x,y,overlay_image):
    """
    배경 이미지에 내가 원하는 이미지를 넣을때 사용하는 함수.
    image: 배경 이미지 (3채널)
    y: 집어넣을 이미지의 위치 y좌표
    x: 집어넣을 이미지의 위치 x좌표
    overlay_image: 덮어 씌울 이미지(4채널)
    """

    alpha = overlay_image[:, :, 3] # BGRA 채널의 이미지 중. Alpha 즉, 투명 영역의 값을 가져옴.
    mask_image = alpha/255 # 0~255 사이의 값을 가진 데이터를 255로 나누면, 0 ~ 1 사이의 값을 가지게 됨. (1: 불투명, 0: 완전)
    
    for c in range(0,3): # channel BGR
        #투명한 영역은 배경 이미지로 나오게, 불투명한 영역은 overlay_image의 것이 나오게
        image[y:y+300, x:x+300, c] = (overlay_image[:,:,c] * mask_image) + image[y:y+300, x:x+300,c] * (1 - mask_image)

def predict_emotion(image):
          preds = emotion_detect_model.predict(image.reshape(-1,img_size,img_size,3))
          text = getLabel_ByArgmax(preds)
          
          return text

count = 0
#Face Detection Start
text = 'happy'

# For webcam input: 웹캠의 경우
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    #감지할 얼굴의 최대 개수
    max_num_faces=1,
    refine_landmarks=True,
    #얼굴 감지에 대한 모델의 최소 신뢰도
    min_detection_confidence=0.5,
    #얼굴 특징(눈, 입꼬리, 등등) 감지에 대한 모델의 최소 신뢰도
    min_tracking_confidence=0.5) as face_mesh,mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    a = 0
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.resize(image,(1920,1080),cv2.INTER_AREA)
        empty_image = np.zeros((1080,1920,3),np.uint8)
        a+=1
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
    #성능을 향상시키기 위해서 ,선택적으로 이미지를 참조할 수 없는 것으로 마킹한다.
    #성능 향상을 위해 임시적으로 이미지를 변경한 부분.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        face_detect_results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Draw the face mesh annotations on the image.
        #변경한 이미지 원상 복구
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if face_detect_results.detections:                   
        #얼굴의 특징과 얼굴 객체가 감지되는 경우.
            if results.multi_face_landmarks:
                for detection in face_detect_results.detections:
                    image_h, image_w,_ = image.shape
                    box = detection.location_data.relative_bounding_box        
                    x = int(box.xmin * image_w)
                    y = int(box.ymin * image_h)
                    w = int(box.width * image_w)
                    h = int(box.height * image_h) 
                for face_landmarks in results.multi_face_landmarks:
                #얼굴 메쉬 웹캠에서 캡쳐되는 이미지에 그리기
                    draw_face_mesh(image,face_landmarks)
                    #얼굴 메쉬 검은 이미지에 그리기
                    draw_face_mesh(empty_image,face_landmarks)
                
                if a %19 == 0:
                    crop = empty_image[y-100:y+w+100,x-100:x+w+100]
                    resize_img = cv2.resize(crop,(img_size,img_size),cv2.INTER_AREA)
                    cv2.imwrite(f'./image/croped_image{a}.jpg', resize_img)
                    text = predict_emotion(resize_img)
                    print("predict emotion:",text)
     
                #화면에 감정과, 이모티콘 출력
                image = cv2.flip(image,1)
                cv2.putText(image,text,(y,x),cv2.FONT_HERSHEY_SIMPLEX,1,COLOR,THICKNESS)
                image = cv2.flip(image,1)
                # cv2.rectangle(image, (x,y),(x+300,y+300),COLOR,THICKNESS) #속이 빈 사각형
                emoticon = cv2.imread(f'./emoticon/{text}.png',cv2.IMREAD_UNCHANGED)
                #overlay 함수 호출
                overlay(image, x,y, emoticon)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Image Face Mesh', cv2.flip(image, 1))
        cv2.imshow('Only Face Mesh', cv2.flip(empty_image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)