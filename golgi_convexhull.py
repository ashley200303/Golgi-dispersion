import tkinter as tk
import os
import cv2
import numpy as np
import pandas as pd
import zipfile
import sys
import csv
import io
from PIL import Image
from read_roi import read_roi_file, read_roi_zip
from roifile import ImagejRoi
from tkinter import filedialog, simpledialog, messagebox
from scalebar_detection import get_scale_info

# 분석되지 않은 ROI가 있을 시 log에 저장됨
LOG_PATH = "unsupported_rois.log"

# 이미지, ROI 파일 경로 및 저장폴더 선택
def select_files_and_folder():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="이미지 파일 선택", filetypes=[("Image files", "*.tif *.png *.jpg *.jpeg")])
    roi_zip_path = filedialog.askopenfilename(title="ROI zip 파일 선택", filetypes=[("ZIP files", "*.zip")])
    output_dir = filedialog.askdirectory(title="결과 저장 폴더 선택")
    return image_path, roi_zip_path, output_dir

def ask_real_scale_length():
    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askfloat("스케일 바 길이", "스케일 바의 실제 길이 (µm)?")
    if value is None:
        messagebox.showerror("입력 오류", "실제 스케일 바 길이를 입력하지 않았습니다.")
        sys.exit(1)
    return value

#ROI mask화
def roi_to_mask(roi, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    roi_type = roi.get("type", "").lower()
    if roi_type in ("polygon", "freehand") and "x" in roi and "y" in roi:
        points = np.array(list(zip(roi["x"], roi["y"])), np.int32)
        if roi_type == "polygon":
            cv2.fillPoly(mask, [points], 255)
        else:  # freehand인 경우
            cv2.polylines(mask, [points], isClosed=True, color=255, thickness=1)
            cv2.fillPoly(mask, [points], 255)
    elif roi_type in ("rect", "rectangle") and all(k in roi for k in ["left", "top", "width", "height"]):
        x, y, w, h = roi["left"], roi["top"], roi["width"], roi["height"]
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    elif roi_type == "oval" and all(k in roi for k in ["left", "top", "width", "height"]):
        x, y, w, h = roi["left"], roi["top"], roi["width"], roi["height"]
        center = (x + w // 2, y + h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    else:
        with open(LOG_PATH, "a") as f:
            f.write(f"지원되지 않는 ROI 타입 또는 오류: {roi_type}\n")
        return None
    return mask

# hull이 저장될 ROI zip file 생성 함수
def create_imagej_roi_bytes(contour, name="roi"):
    points = contour.squeeze()
    roi = ImagejRoi.frompoints(points, name=name)
    return roi.tobytes()

# convex hull 제작 및 저장
def analyze_particles_and_save(image_path, roi_zip_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image = np.array(Image.open(image_path).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Scale bar detection (auto or manual)
    pixels_per_um, scale_bar_px, real_length = get_scale_info(image)
    scale_ratio = 1 / pixels_per_um  # µm/px
    print(f"[INFO] 스케일 바: {scale_bar_px}px = {real_length}µm → {scale_ratio:.3f} µm/px")

    # 결과 저장
    scale_output_dir = os.path.join(output_dir, "scale_output")
    os.makedirs(scale_output_dir, exist_ok=True)
    scale_path = os.path.join(scale_output_dir, "pixels_per_um.txt")
    with open(scale_path, "w") as f:
        f.write(f"Pixels per micron: {pixels_per_um:.6f} px/µm\n")
        f.write(f"Scale bar length (in pixels): {scale_bar_px:.2f} px\n")
        f.write(f"Real scale bar length: {real_length:.2f} µm\n")

    roi_dict = read_roi_zip(roi_zip_path)
    results = []
    roi_zip_bytes = {}

    for idx, (name, roi) in enumerate(roi_dict.items()):
        mask = roi_to_mask(roi, gray.shape)
        if mask is None:
            continue

        # ROI 내부 이미지로부터 particle 추출
        roi_img = cv2.bitwise_and(gray, gray, mask=mask)
        _, bin_img = cv2.threshold(roi_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        area_px = cv2.contourArea(hull)
        area_um = area_px * (scale_ratio ** 2)

        # 저장용 이미지
        result_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result_img, [hull], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"hull_{idx+1}.png"), result_img)

        # ImageJ ROI로 변환
        roi_bytes = create_imagej_roi_bytes(hull, name=f"hull_{idx+1}")
        roi_zip_bytes[f"hull_{idx+1}.roi"] = roi_bytes

        results.append({
            "ROI Index": idx+1,
            "Convex Hull Area (px²)": area_px,
            "Convex Hull Area (µm²)": area_um
        })

    # CSV 저장
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "convex_hull_results.csv"), index=False)

    # ROI zip 저장
    zip_path = os.path.join(output_dir, "hull_rois.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for filename, data in roi_zip_bytes.items():
            zf.writestr(filename, data)

    # 파일 탐색기 열기
    try:
        if sys.platform == "win32":
            os.startfile(output_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", output_dir])
        else:
            subprocess.Popen(["xdg-open", output_dir])
    except Exception as e:
        print(f"탐색기 열기 실패: {e}")
        
    print(f"완료: {len(results)}개 ROI 분석 및 저장 완료 → {output_dir}")

if __name__ == "__main__":
    image_path, roi_zip_path, output_dir = select_files_and_folder()
    if image_path and roi_zip_path and output_dir:
        analyze_particles_and_save(image_path, roi_zip_path, output_dir)
    else:
        print("입력 또는 출력 경로가 선택되지 않았습니다.")
