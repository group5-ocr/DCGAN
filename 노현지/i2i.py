import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import cv2
import numpy as np
from diffusers.utils import load_image

# ------------------------------------------------
# 1. GPU 및 모델 로드
# ------------------------------------------------
print("GPU 및 모델을 로드합니다... ⏳")

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# ControlNet 모델 로드 (Canny Edge Detection)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
    use_safetensors=True
).to(device)

# SD 1.5 기본 파이프라인과 ControlNet을 함께 로드
# 여기서 StableDiffusionPipeline 대신 StableDiffusionControlNetPipeline을 사용합니다.
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True
).to(device)

# 학습된 LoRA 가중치를 로드하여 스타일을 적용합니다.
lora_path = "sd15_lora_masterpiece"
pipeline.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")

# ------------------------------------------------
# 2. 이미지 전처리 및 스타일 변환 (ControlNet + LoRA)
# ------------------------------------------------
print("\n테스트 이미지에 스타일을 적용합니다. 🖼️")

# 테스트 이미지 경로 설정
test_image_path = "portrait.jpg"
if not os.path.exists(test_image_path):
    print(f"❌ 오류: 테스트 이미지 '{test_image_path}'를 찾을 수 없습니다. 경로를 확인하세요.")
    exit(1)

# 원본 이미지 로드
init_image = load_image(test_image_path).convert("RGB")
init_image = init_image.resize((512, 512))

# Canny 엣지 맵 생성 (원본 이미지로부터 윤곽선 추출)
image_np = np.array(init_image)
image_canny = cv2.Canny(image_np, 100, 200)
image_canny = Image.fromarray(image_canny)

# 프롬프트 설정 (LoRA 학습 프롬프트와 유사하게)
prompt = "masterpiece oil painting, highly detailed, high quality"
negative_prompt = "unwanted,lowres, bad anatomy, bad hands, cropped, worst quality, deformed,undesired"

# 이미지 생성
# ControlNet은 image 인수에 Canny 엣지 맵을 받아서 윤곽선을 유지합니다.
generator = torch.Generator(device).manual_seed(100)
generated_images = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image_canny, # <--- Canny 엣지 맵을 사용하여 구도 제어
    num_inference_steps=25,
    generator=generator
).images[0]

# 결과 이미지 저장
output_image_path = "result_portrait_controlnet_combined.png"
generated_images.save(output_image_path)
print(f"\n스타일이 적용된 이미지가 '{output_image_path}'에 저장되었습니다. ✅")