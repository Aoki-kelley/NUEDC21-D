# 摄像头录制视频

import cv2

cap = cv2.VideoCapture(1)  # 打开摄像头

fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
out = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))  # 写入视频

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)  # 写入帧
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # q退出
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
