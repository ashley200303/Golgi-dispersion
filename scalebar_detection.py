import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image
import os
import sys
import subprocess

# 한글 경로 지원 이미지 로더
def load_image_unicode(path):
    pil_img = Image.open(path)
    return np.array(pil_img)

# 자동 스케일바 감지 함수
def detect_scale_bar(image, min_length=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > min_length and h < w * 0.2:
            return w
    return None

# 클릭/드래그 기반 수동 감지
def manual_scale_bar_detection(image):
    print("자동 감지 실패. 마우스로 스케일 바를 지정하세요.")
    click_point = []
    drag_points = []

    def click_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point.append((x, y))
            cv2.destroyWindow("1. Click near scale bar")

    def drag_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_points.clear()
            drag_points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drag_points.append((x, y))
            cv2.destroyWindow("2. Drag over scale bar")

    img_resized = cv2.resize(image, None, fx=0.7, fy=0.7)
    factor = image.shape[1] / img_resized.shape[1]

    # 1. 클릭
    cv2.imshow("1. Click near scale bar", img_resized)
    cv2.setMouseCallback("1. Click near scale bar", click_callback)
    cv2.waitKey(0)

    if not click_point:
        raise Exception("클릭하지 않음")

    x, y = [int(i * factor) for i in click_point[0]]
    r = 100
    crop = image[max(0, y-r):y+r, max(0, x-r):x+r]
    if crop.size == 0:
        raise Exception("빈 이미지 영역")
    # 확대 배율 설정
    zoom_factor = 2.0
    crop_zoomed = cv2.resize(crop, None, fx=zoom_factor, fy=zoom_factor)


    # 드래그된 좌표 저장
    cv2.imshow("2. Drag over scale bar", crop_zoomed)
    cv2.setMouseCallback("2. Drag over scale bar", drag_callback)
    cv2.waitKey(0)

    if len(drag_points) != 2:
        raise Exception("드래그 실패")

    # 픽셀 길이
    (sx, sy), (ex, ey) = drag_points
    sx, sy = sx / zoom_factor, sy / zoom_factor
    ex, ey = ex / zoom_factor, ey / zoom_factor
    pixel_length = ((ex - sx)**2 + (ey - sy)**2) ** 0.5

    real_length = float(input("스케일 바의 실제 길이 (µm): "))
    pixels_per_um = pixel_length / real_length
    return pixels_per_um, pixel_length, real_length

def main():
    global image_display

    # 이미지 파일 선택
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="이미지를 선택하세요")
    if not image_path:
        print("이미지를 선택하지 않았습니다.")
        sys.exit(1)

    image = load_image_unicode(image_path)

    # 1단계: 자동 감지
    scale_bar_px = detect_scale_bar(image)
    if scale_bar_px:
        real_length = float(input("스케일 바의 실제 길이 (µm): "))
        pixels_per_um = scale_bar_px / real_length
        print(f"자동 감지: {pixels_per_um:.4f} px/µm")
    # 2단계: 수동 감지
    else:
        pixels_per_um, scale_bar_px, real_length = manual_scale_bar_detection(image)
        print(f"수동 감지: {pixels_per_um:.4f} px/µm")

    # 결과 저장
    output_dir = os.path.join(os.path.dirname(image_path), "scale_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pixels_per_um.txt")
    with open(output_path, "w") as f:
        f.write(f"Pixels per micron: {pixels_per_um:.6f} px/µm\n")
        f.write(f"Scale bar length (in pixels): {scale_bar_px:.2f} px\n")
        f.write(f"Real scale bar length: {real_length:.2f} µm\n")
    
    print(f"\n Pixels per µm: {pixels_per_um:.3f} px/µm")
    print(f"\n 결과 저장 완료: {output_path}")


    # 탐색기 열기
    try:
        if sys.platform == "win32":
            os.startfile(output_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", output_dir])
        else:
            subprocess.Popen(["xdg-open", output_dir])
    except Exception as e:
        print(f"탐색기 열기 실패: {e}")

if __name__ == "__main__":
    main()

def get_scale_info(image):
    scale_bar_px = detect_scale_bar(image)
    if scale_bar_px:
        real_length = float(input("스케일 바의 실제 길이 (µm): "))
        pixels_per_um = scale_bar_px / real_length
        print(f"[AUTO] {pixels_per_um:.4f} px/µm")
    else:
        pixels_per_um, scale_bar_px, real_length = manual_scale_bar_detection(image)
        print(f"[MANUAL] {pixels_per_um:.4f} px/µm")
    return pixels_per_um, scale_bar_px, real_length