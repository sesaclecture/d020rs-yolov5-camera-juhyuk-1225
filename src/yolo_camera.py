import torch
import cv2

# Model​
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Video capture
cap = cv2.VideoCapture(0)

# TODO: Loop for camera frames
while True:
    # Read frame (BGR to RGB)
    ret, frame = cap.read()
    # TODO: break the loop on error
    if not ret:
        print("Error: 카메라를 불러올 수 없습니다.")
        break
    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    # TODO: Boudning box 그리기
    for i, obj in enumerate(results.xyxy[0]):
        # TODO: 인식결과를 표시하기 위한 좌표를 얻음
        x1,y1,x2,y2,_,cls = map(int, obj[:6])

        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        conf = obj[4]
        # [추가 주석] 화면에 표시할 라벨 텍스트를 생성합니다. (클래스 이름 + 신뢰도)
        label = f"{model.names[cls]} {conf:.2f}"

        # TODO: OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        # x,y,w,h = cv2.boundingRect()
        # cv2.rectangle(results, (x, y), (x+w, y+h), (0,255,0),2)
        # cv2.putText(results, f"Rect: ({x}, {y}, {w}, {h})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        # [추가 주석] 위 주석 코드처럼 'results'에 그리면 안 되고, 원본 이미지인 'frame'에 그려야 합니다.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        obj_info = list(map(int, obj))
        print(f"Object {i}: {model.names[obj_info[5]]}")

    # TODO: 화면 표시
    # cv2.imshow("test",results)
    # [추가 주석] 위 코드처럼 'results' 객체를 직접 표시하면 오류가 발생합니다.
    # [추가 주석] for 루프에서 사각형과 텍스트를 그린 'frame'을 화면에 표시해야 합니다.
    cv2.imshow("test", frame)
    
    # TODO: 종료를 위한 key 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()