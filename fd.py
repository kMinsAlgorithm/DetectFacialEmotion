import cv2
import mediapipe as mp
import time 
from keras.models import load_model
import tensorflow as tf
import numpy as np

img_size =300

#얼굴을 찾고 찾은 얼굴에 표시를해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection # 얼굴 검출을 위한 face_detection 모듈을 사용
mp_drawing = mp.solutions.drawing_utils #얼굴의 특징을 그리기 위한 Drawing_utils 모듈을 사용 

cap = cv2.VideoCapture(0)
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

emotion_detect_model = load_model('weights_best_emotion_detect5.h5')

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
#model_selection : 0또는 1의 옵션을 가질 수 있음. 0의 경우 카메라로부터 2미터 이내의 근거리 1의 경우 5미터 이내의 거리에 적합
#min_detection_confidence : 임계치와 같은 의미, 0.5라고 적은 경우 모델이 얼굴일 것이다 라고 50퍼센트 확신하면 얼굴로 인정한다는 의미.
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
  #동영상이 잘 열렸는지 확인
  a = 0
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image,(1920,1080),cv2.INTER_AREA)
    original_img = image.copy()
    a +=1
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
   
    # 성능항샹을 위해 임시적으로 이미지를 변경한 부분.
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #BGR 이미지를 RGB로 변경 
    #image로 부터 얼굴을 검출하여 results 변수에 저장.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #임시로 변경한 이미지를 다시 원상복구
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
    #검출된 얼굴이 있는 경우
    if results.detections:
    #검출된 사람만큼 사각형을 그려줌.
      for detection in results.detections:
        image_h, image_w,_ = image.shape
        box = detection.location_data.relative_bounding_box
        
        x = int(box.xmin * image_w)
        y = int(box.ymin * image_h)
        w = int(box.width * image_w)
        h = int(box.height * image_h)
        
        #감정 분석
        if a %19 == 0:
          # crop = image[y-100:y+300,x-100:x+300]
          crop = image[y:y+w,x:x+w]
          resize_img = cv2.resize(crop,(img_size,img_size),cv2.INTER_AREA)
          # gray_img = cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)
          # print(resize_img.shape)
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
        
    # Flip the image horizontally for a selfie-view display. # 좌우 전환하여 화면을 송출
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    cv2.imshow('Original', cv2.flip(original_img, 1))               
    if cv2.waitKey(10) == ord('q'): 
      break
    
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)