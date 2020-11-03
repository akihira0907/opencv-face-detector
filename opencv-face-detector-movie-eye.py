import cv2

# 動画の読込
# カメラ等でストリーム再生の場合は引数にデバイスID(0等)を記述
video = cv2.VideoCapture('C001_001_GP01.mp4')

# 分類機の読込
cascade_front_path = "haarcascade_frontalface_default.xml"
cascade_front = cv2.CascadeClassifier(cascade_front_path)
cascade_profile_path = "haarcascade_profileface.xml"
cascade_profile = cv2.CascadeClassifier(cascade_profile_path)
cascade_eye_path = "haarcascade_eye.xml"
cascade_eye = cv2.CascadeClassifier(cascade_eye_path)

# 顔検出のパラメータ
scaleFactor = 1.08
minNeighbors = 7
minSize = (50, 50)

while video.isOpened():
    # フレーム読込
    ret, frame = video.read()

    # フレーム読込失敗
    if not ret:
        break

    # フレームの高さと幅
    frame_h, frame_w, _ = frame.shape

    # フレームをグレースケールに変換
    gry_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出(正面)
    facerects_front = cascade_front.detectMultiScale(
        gry_frame,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
        )
    # 顔検出(横顔)
    facerects_profile = cascade_profile.detectMultiScale(
        gry_frame,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
        )
    # 横顔については左右反転も実行
    facerects_profile_reverse = cascade_profile.detectMultiScale(
        cv2.flip(gry_frame, 1),
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )

    # 矩形線の色
    rectangle_front_color = (0, 255, 0) # 緑色
    rectangle_profile_color = (255, 0, 0) # 青色
    rectangle_eye_color = (0, 0, 255) # 赤色
    
    # 顔を検出した場合矩形を描画
    if len(facerects_front) > 0:
        for rect in facerects_front:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_front_color, thickness=2)
            # 検出した顔に対して瞳の検出を実行
            face = frame[y:y+h, x:x+w]
            gry_face = gry_frame[y:y+h, x:x+w]
            eyes = cascade_eye.detectMultiScale(gry_face)
            for eye in eyes:
                ex, ey, ew, eh = eye
                cv2.rectangle(face, (ex, ey), (ex+ew, ey+eh), rectangle_eye_color, thickness=2)
    # 横顔の矩形を描画
    if len(facerects_profile) > 0:
        for rect in facerects_profile:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_profile_color, thickness=2)
    # 左右反転分
    if len(facerects_profile_reverse) > 0:
        for rect in facerects_profile_reverse:
            x, y, w, h = rect
            cv2.rectangle(frame, (frame_w-x-w, y), (frame_w-x, y+h), rectangle_profile_color, thickness=2)
            
    # フレームの描画
    cv2.imshow('frame', frame)

    # qキーの押下で処理を中止
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
# メモリの解放
video.release()
cv2.destroyAllWindows()
