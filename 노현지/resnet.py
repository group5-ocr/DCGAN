import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. 모델 로드 및 GPU 설정
print("사전 학습된 ResNet18 모델을 로드합니다... ⏳")

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# 사전 학습된 ResNet18 모델 로드
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 마지막 fully-connected 레이어를 2개의 클래스(AI, Real)를 분류하도록 수정
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

model.eval() # 추론 모드로 설정

# 2. 이미지 전처리
# ResNet 모델에 맞는 이미지 전처리 파이프라인 정의
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. 이미지 분류 및 결과 출력
print("\n생성된 이미지의 AI 여부를 분류합니다... 🕵️‍♀️")

# 생성된 이미지 경로
image_path = "result_portrait_controlnet_combined.png"
if not os.path.exists(image_path):
    print(f"❌ 오류: '{image_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    # 이미지 로드
    img = Image.open(image_path).convert("RGB")
    
    # 전처리 및 배치 차원 추가
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # 예측 수행
    with torch.no_grad():
        output = model(img_tensor)
        
    # 결과 해석
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # 임의로 클래스 0을 'real', 클래스 1을 'fake'로 가정
    real_prob = probabilities[0].item() * 100
    fake_prob = probabilities[1].item() * 100

    print(f"\n✅ 분류 결과:")
    print(f"  AI(Fake)일 확률: {fake_prob:.2f}%")
    print(f"  실제(Real)일 확률: {real_prob:.2f}%")

    # 최종 판단
    if fake_prob > real_prob:
        print("➡️ 최종 판단: AI가 생성한 이미지로 보입니다. 🤖")
    else:
        print("➡️ 최종 판단: 실제 이미지일 가능성이 높습니다. 📸")