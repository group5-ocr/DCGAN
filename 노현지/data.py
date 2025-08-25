import os
import zipfile
import subprocess
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login
import torch

# ------------------------------------------------
# 1. 환경 설정 및 데이터셋 압축 해제
# ------------------------------------------------
print("환경 설정 및 다운로드된 데이터셋 압축 해제를 시작합니다. 🖼️")

# Hugging Face 토큰 로그인 (환경 변수 또는 직접 입력)
hf_token = os.environ.get("HUGGINGFACE_TOKEN", "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")
login(token=hf_token)

# 직접 다운로드한 데이터셋 파일명과 압축 해제할 폴더명을 지정하세요.
zip_file_path = "portrait-paintings.zip"
extract_dir = "portrait_paintings"

# 데이터셋 압축 해제
try:
    if not os.path.exists(zip_file_path):
        print(f"❌ 오류: '{zip_file_path}' 파일을 찾을 수 없습니다. 직접 다운로드하여 현재 폴더에 넣어주세요.")
        exit(1)

    print(f"다운로드된 데이터셋({zip_file_path}) 압축을 해제합니다...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("압축 해제 완료. ✅")

except Exception as e:
    print(f"❌ 오류: 압축 해제 중 문제 발생 - {e}")
    exit(1)

# ------------------------------------------------
# 2. 이미지 리사이징
# ------------------------------------------------
def resize_images_with_progress(root_dir, target_size=(512, 512)): # 해상도를 512로 변경
    print("\n이미지 리사이징 작업을 시작합니다...")
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                all_files.append(os.path.join(dirpath, filename))

    for file_path in tqdm(all_files, desc="리사이징 진행 중"):
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                img_resized.save(file_path, quality=95)
        except Exception as e:
            tqdm.write(f"❌ 오류: {file_path} 처리 중 문제 발생: {e}")

    print("\n모든 이미지 리사이징 작업이 완료되었습니다. ✅")

image_directory = "portrait_paintings/Images"
if os.path.isdir(image_directory):
    resize_images_with_progress(image_directory)
else:
    print(f"오류: 이미지 디렉토리 '{image_directory}'를 찾을 수 없습니다. 다운로드 및 압축 해제 과정을 확인하세요.")
    exit(1)