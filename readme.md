## Getting Started

1. install visual studio build tools with desktop development with c++, cmake, and python 3.9.13

2. run this in powershell

```pwsh
git clone https://github.com/rsa17826/ai-img-detection.git ./ai-img-detection
cd ./ai-img-detection/object-detection
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz -o ./zip.tar.gz
tar -xvzf zip.tar.gz --strip-components=1
rm ./zip.tar.gz
py -3.13 -m venv ./.venv
Set-ExecutionPolicy Bypass -Scope Process -Force
./.venv/Scripts/Activate.ps1
python -m pip install -r ./requirements.txt
python tf_text_graph_ssd.py --input frozen_inference_graph.pb --config pipeline.config --output graph.pbtxt

cd ..

cd ./person-recognition
py -3.9 -m venv ./.venv
./.venv/Scripts/Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install face_recognition facenet-pytorch opencv-python flask pandas numpy eel

cd ..

```

3. Run game.bat
4. Access web interface for interaction
