import cv2
import numpy as np

# load model pre-trained

net = cv2.dnn.readNetFromCaffe(
    'C:/Users/Trongdz/Desktop/Project Computer Vision/b28_face detection with openCV DNN/models/deploy.prototxt.txt',
    'C:/Users/Trongdz/Desktop/Project Computer Vision/b28_face detection with openCV DNN/models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
)

# open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # chuẩn bị dữ liệu đầu vào
    # 1.0: tỉ lệ co dãn
    # (300,300): Kích thước mô hình yêu cầu đầu vào
    # (104.0, 177.0, 123.0): màu sắc trung bình của ảnh
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)

    # đặt dữ liệu đầu vào cho mạng
    # nếu có nhiều ảnh thì đưa vào vòng for
    net.setInput(blob)

    # chạy mạng để nhận dạng khuôn mặt
    faces = net.forward()

    # lấy kích thước của ảnh đầu vào
    h = frame.shape[0]
    w = frame.shape[1]
    # print(faces.shape)

    # vẽ các khuôn mặt nhận dạng được
    for i in range(faces.shape[2]):
        confident = faces[0, 0, i, 2] # độ tin cậy
        # nếu độ tin cậy > 0.5 thì vẽ
        if confident > 0.5:
            # lấy tọa độ của khuôn mặt
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            # vẽ hình chữ nhật
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2) 
            # hiển thị độ tin cậy
            text = 'Face: {:.2f}%'.format(confident * 100)
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # show the orginal img

    cv2.imshow('Result', frame)
    if (cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()


        
        