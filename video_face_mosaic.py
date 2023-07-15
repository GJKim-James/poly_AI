# 설치한 OpenCV 패키지 불러오기
import cv2

# 학습된 얼굴 정면 검출기 사용하기
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

cam = cv2.VideoCapture("video/poly.mp4")

# 동영상은 시작부터 종료될 때까지 프레임을 지속적으로 받아야 하기 때문에 while문을 계속 반복함
while True:
    # ret 에는 프레임을 잘 가져왔으면 True, 그렇지 않으면 False 반환됨
    # video_image 에는 프레임의 좌표 값이 반환됨
    ret, video_image = cam.read()

    # 동영상으로부터 프레임(이미지)을 잘 받았으면 실행함
    if ret is True:

        # 동영상의 프레임 얼굴 인식율을 높이기 위해 흑백으로 변경함
        gray = cv2.cvtColor(video_image, cv2.COLOR_BGR2GRAY)

        # 변환한 흑백 사진으로부터 히스토그램 평활화
        gray = cv2.equalizeHist(gray)

        # 얼굴 인식하기
        faces = face_cascade.detectMultiScale(gray, 1.5, 5, 0, (20, 20))

        for face in faces:
            # 얼굴 위치 값 가져오기
            x, y, w, h = face

            # 원본 이미지로부터 얼굴 영역 가져오기
            face_image = video_image[y:y + h, x:x + w]

            # 모자이크 비율(픽셀 크기 증가로 모자이크 만들기)
            mosaic_rate = 30

            # 얼굴 영역의 픽셀을 mosaic_rate에 따라 나눠서 픽셀 확대
            face_image = cv2.resize(face_image, (w // mosaic_rate, h // mosaic_rate))

            # 확대한 얼굴 이미지(픽셀)를 얼굴 크기에 덮어쓰기
            face_image = cv2.resize(face_image, (w, h), interpolation=cv2.INTER_AREA)

            # 원본 이미지에 모자이크 처리한 얼굴 이미지 붙이기
            # y부터 y + h까지, x부터 x + w까지
            video_image[y:y + h, x:x + w] = face_image

            # 얼굴 검출 사각형 그리기
            cv2.rectangle(video_image, face, (255, 0, 0), 4)  # (255, 0, 0)은 파란색을 의미

        # 인식한 얼굴을 사각형으로 표시한 이미지 출력하기
        cv2.imshow("video_mosaic", video_image)

    # 입력받는 것 대기하기, 작성 안하면 결과창이 바로 닫힘
    # 1은 esc 키를 누르는 것을 의미
    if cv2.waitKey(1) > 0:
        break
