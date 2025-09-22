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

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Video capture
cap = cv2.VideoCapture(0)

# Loop for camera frames
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Error: 카메라를 불러올 수 없습니다.")
        break

    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    # Boudning box 그리기 (사용자 코드 유지 및 수정)
    for i, obj in enumerate(results.xyxy[0]):
        # 인식결과를 표시하기 위한 좌표를 얻음
        # obj 텐서에서 x1, y1, x2, y2, 신뢰도, 클래스 ID를 직접 추출
        x1, y1, x2, y2, conf, cls_id = obj
        
        # 좌표와 클래스 ID를 정수형으로 변환
        x1, y1, x2, y2, cls_id = map(int, [x1, y1, x2, y2, cls_id])

        # 인식된 정확도(confidence)와 클래스를 label로 구성
        label = f"{model.names[cls_id]} {conf:.2f}"
        
        # [수정 1] OpenCV를 이용해서 'results'가 아닌 'frame'에 사각형과 text를 출력
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 기존 print문 유지
        print(f"Object {i}: {model.names[cls_id]}")

    # [수정 2] 화면 표시: 'results' 객체가 아닌, 그림이 그려진 'frame'을 표시
    cv2.imshow("test", frame)
    
    # 종료를 위한 key 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()