import cv2

# 分類機の読込
cascade_path = "haarcascade_profileface.xml"

# 画像ファイルの読込(結果表示に使用)
# img = cv2.imread('C001_001_GP01.jpg') # カラーで読込
img = cv2.imread('profile_2.jpg') # カラーで読込

# 画像の高さと幅
img_h, img_w, _ = img.shape

# グレースケールに変換(顔検出に使用)
gry_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# カスケード検出器の特徴量を取得
cascade = cv2.CascadeClassifier(cascade_path)

# 顔検出の実行
# scaleFactor: 画像スケールにおける縮小量
# minNeighbors: 信頼性のパラメータ
# minSize: 物体が取り得る最小サイズ
facerects = cascade.detectMultiScale(
    gry_img,
    scaleFactor=1.1,
    minNeighbors=6,
    minSize=(30, 30)
    )

# 左右反転して実行
facerects_reverse = cascade.detectMultiScale(
    cv2.flip(gry_img, 1),
    scaleFactor=1.1,
    minNeighbors=6,
    minSize=(30, 30)
    )

# 矩形線の色
rectangle_color = (0, 255, 0) # 緑色

# 顔を検出した場合
if len(facerects) > 0:
    for rect in facerects:
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), rectangle_color, thickness=2)
# 左右反転分
if len(facerects_reverse) > 0:
    for rect in facerects_reverse:
        x, y, w, h = rect
        cv2.rectangle(img, (img_w-x-w, y), (img_w-x, y+h), rectangle_color, thickness=2)

# 画像の表示
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
