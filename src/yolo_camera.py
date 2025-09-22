# import torch
# import cv2

# # Model​
# model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# # Video capture
# cap = cv2.VideoCapture(0)

# # TODO: Loop for camera frames

# while True:
#     # Read frame (BGR to RGB)
#     ret, frame = cap.read()
#     # TODO: break the loop on error
#     if not ret:
#         print("Error: 카메라를 불러올 수 없습니다.")
#         break
#     # 추론 실행 (BGR -> RGB)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = model(rgb_frame)

#     # TODO: Boudning box 그리기
#     for i, obj in enumerate(results.xyxy[0]):
#         # TODO: 인식결과를 표시하기 위한 좌표를 얻음
#         x1,y1,x2,y2,_,cls = map(int, obj[:6])

#         # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
#         conf = obj[4]
#         # TODO: OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
#         cv2.rectangle(rgb_frame, x1, y1, x2, y2)
#         cv2.putText(rgb_frame, "1")
#         obj_info = list(map(int, obj))
#         print(f"Object {i}: {model.names[obj_info[5]]}")

#     # TODO: 화면 표시
#     cv2.imshow("test",rgb_frame)
#     # TODO: 종료를 위한 key 처리
#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:  # ESC
#         break
# cap.release()
# cv2.destroyAllWindows()

import torch
import cv2

# 모델 로드
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# 카메라 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: 카메라를 열 수 없습니다.")
    exit()

while True:
    # 카메라 프레임 읽기
    ret, frame = cap.read() # frame이 바로 그림을 그릴 '도화지'입니다.
    if not ret:
        print("Error: 카메라로부터 프레임을 읽을 수 없습니다.")
        break

    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    # Boudning box 그리기
    # results.xyxy[0]에 탐지된 객체 정보가 담겨 있습니다.
    for obj in results.xyxy[0]:
        # obj 텐서에서 필요한 값들을 추출합니다.
        x1, y1, x2, y2, conf, cls = obj
        
        # 정수형으로 변환
        x1, y1, x2, y2, cls = map(int, [x1, y1, x2, y2, cls])
        
        # 클래스 이름과 신뢰도를 라벨로 만듭니다.
        label = f"{model.names[cls]} {conf:.2f}"
        
        # !! 중요 !!
        # 'results'가 아닌 원본 이미지 'frame'에 사각형과 텍스트를 그립니다.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 초록색 사각형
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # !! 중요 !!
    # 'results'가 아닌, 그림이 모두 그려진 'frame'을 화면에 표시합니다.
    cv2.imshow("YOLOv5 Camera Feed", frame)

    # 'ESC' 키를 누르면 루프 종료
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()