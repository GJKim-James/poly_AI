# 설치한 OpenCV 패키지 불러오기
import cv2

# 인식율을 높이기 위한 전처리
def preprocessing():

    # 분석하기 위한 이미지 불러오기
    image = cv2.imread("image/my_face.jpg", cv2.IMREAD_COLOR)

    # 이미지가 존재하지 않으면, 에러 반환
    if image is None:
        return None, None

    # 이미지 크기 사이즈 변경하기(이미지가 클 경우 바꿔주기 위함)
    image = cv2.resize(image, (700, 700))

    # 흑백 사진으로 변경
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray1", gray)

    # 변환한 흑백 사진으로부터 히스토그램 평활화
    gray = cv2.equalizeHist(gray)

    cv2.imshow("gray2", gray)

    return image, gray

# 학습된 얼굴 정면 검출기 사용하기
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# 학습된 눈 검출기 사용하기
eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye_tree_eyeglasses.xml")

# 인식율을 높이기 위한 전처리 함수 호출
image, gray = preprocessing()

if image is None:
    raise Exception("영상 파일 읽기 에러")

# 얼굴 검출 수행(정확도 높이는 방법으로 아래 파라미터를 조절함)
# 얼굴 검출은 히스토그램 평활화한 이미지 사용
# scaleFactor : 보통 1.0 ~ 1.5 사용
# minNeightbors : 인근 유사 픽셀 발견 비율이 2번 이상
# flags : 0 -> 더 이상 사용하지 않는 인자값
# 분석할 이미지의 최소 크기 : 가로 100, 세로 100
faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

# any() -> 값이 뭐라도 있으면
if faces.any():

    # 얼굴 위치 값 가져오기
    x, y, w, h = faces[0]

    # 원본 이미지로부터 얼굴 영역 가져오기
    face_image = image[y:y + h, x:x + w]

    # 눈 검출 수행하기(정확도 높이는 방법으로 아래 파라미터를 조절함)
    # 눈 검출은 얼굴 이미지 영역만 불러와 분석 수행
    # scaleFactor : 보통 1.0 ~ 1.5 사용
    # minNeightbors : 인근 유사 픽셀 발견 비율이 7번 이상
    # flags : 0 -> 더 이상 사용하지 않는 인자값
    # 분석할 이미지의 최소 크기 : 가로 25, 세로 20
    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))

    # 눈을 찾을 수 있다면
    if len(eyes) == 2:
        for ex, ey, ew, eh in eyes:
            # 눈 가운데 위치 값 가져오기
            center = (x + ex + ew // 2, y + ey + eh // 2)

            # 눈 중심에 원 그리기
            cv2.circle(image, center, 10, (0, 255, 0), 2)

    else:
        print("눈 미검출")

    # 얼굴 검출 사각형 그리기
    cv2.rectangle(image, faces[0], (255, 0, 0), 4)

    # 사이즈 변경된 이미지로 출력하기
    cv2.imshow("MyFace", image)

else:
    print("얼굴 미검출")

# 입력받는 것 대기하기, 작성 안하면 결과창이 바로 닫힘
cv2.waitKey(0)
