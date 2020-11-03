import cv2

# 動画の読込
# カメラ等でストリーム再生の場合は引数にデバイスID(0等)を記述
video = cv2.VideoCapture('C001_001_GP01.mp4')

# 分類機の読込
cascade_path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

while video.isOpened():
    # フレーム読込
    ret, frame = video.read()

    # フレーム読込失敗
    if not ret:
        break

    # フレームをグレースケールに変換
    gry_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出
    facerects = cascade.detectMultiScale(
        gry_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
        )

    # 矩形線の色
    rectangle_color = (0, 255, 0) # 緑色

    # 顔を検出した場合
    if len(facerects) > 0:
        for rect in facerects:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, thickness=2)

    # フレームの描画
    cv2.imshow('frame', frame)

    # qキーの押下で処理を中止
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
# メモリの解放
video.release()
cv2.destroyAllWindows()
