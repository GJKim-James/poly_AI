import cv2

# 분석하기 위한 이미지 불러오기
image = cv2.imread("image/lesserafim.jpg", cv2.IMREAD_COLOR)

# 흑백 사진으로 변경
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 변환한 흑백 사진으로부터 히스토그램 평활화
gray = cv2.equalizeHist(gray)

if image is None: raise Exception("이미지 읽기 실패")

# 학습된 얼굴 정면 검출기 사용하기
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# 얼굴 검출 수행(정확도 높이는 방법으로 아래 파라미터를 조절함)
# 얼굴 검출은 히스토그램 평활화한 이미지 사용
# scaleFactor : 보통 1.0 ~ 1.5 사용
# minNeightbors : 인근 유사 픽셀 발견 비율이 5번 이상
# flags : 0 -> 더 이상 사용하지 않는 인자값
# 분석할 이미지의 최소 크기 : 가로 100, 세로 100
faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (100, 100))

# 인식된 얼굴의 수
facesCnt = len(faces)

# 인식된 얼굴의 수 출력
print("인식된 얼굴의 수 : ", len(faces))

# 얼굴이 검출되었다면,
if facesCnt > 0:

    # 검출된 얼굴의 수만큼 반복하여 실행함
    for face in faces:
        # 얼굴 위치 값 가져오기
        x, y, w, h = face

        # 원본 이미지로부터 얼굴 영역 가져오기
        face_image = image[y:y + h, x:x + w]

        # 모자이크 비율(픽셀 크기 증가로 모자이크 만들기)
        mosaic_rate = 30

        # 얼굴 영역의 픽셀을 mosaic_rate에 따라 나눠서 얼굴 이미지 크기 축소
        face_image = cv2.resize(face_image, (w // mosaic_rate, h // mosaic_rate))

        # 축소된 이미지를 원래 크기로 강제로 확대(강제 확대되면서 픽셀이 깨짐)
        # interpolation 부분은 얼굴 영역을 벗어나 깨지는 부분 없애는 작업
        face_image = cv2.resize(face_image, (w, h), interpolation=cv2.INTER_AREA)

        # 원본 이미지에 모자이크 처리한 얼굴 이미지 붙이기
        # y부터 y + h까지, x부터 x + w까지
        image[y:y + h, x:x + w] = face_image

    # 모자이크 처리된 이미지 파일 생성하기
    cv2.imwrite("result/my_face_mosaic.jpg", image)

    # 모자이크 처리된 이미지 보여주기
    cv2.imshow("mosaic_image", cv2.imread("result/my_face_mosaic.jpg", cv2.IMREAD_COLOR))

else:
    print("얼굴 미검출")

# 입력받는 것 대기하기, 작성 안하면 결과창이 바로 닫힘
cv2.waitKey(0)
