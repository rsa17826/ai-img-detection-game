## Getting Started

1. install visual studio build tools with desktop development with c++, cmake, and python 3.9.13

2. run this in powershell

```pwsh
git clone https://github.com/rsa17826/ai-img-detection-game.git ./ai-img-detection-game
cd ./ai-img-detection-game

pip install uv

uv venv -p 3.9.13
.venv\Scripts\activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install face_recognition facenet-pytorch opencv-python flask pandas numpy eel

```

3. Run game.bat
4. Access web interface for interaction
