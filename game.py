# region imports
import os
class HaltonRNG:
    def __init__(self):
        self.index = 1

    def next(self):
        result = 0
        f = 1
        i = self.index
        base = 2

        while i > 0:
            f = f / base
            result += f * (i % base)
            i //= base

        self.index += 1
        return result

class JitteredHaltonRNG(HaltonRNG):
    def next(self):
        base_value = super().next()
        jitter = (random.random() - 0.5) * 0.05  # small noise
        return max(0.0, min(1.0, base_value + jitter))

print("changing dir to ", os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
import cv2
from typing import Any
import random, time, re
from biasRand import BalancedRand
import eel
from threading import Thread
import os, base64
import enroll_faces
from facenet_pytorch import MTCNN, InceptionResnetV1 # type: ignore
import numpy as np
from pathlib import Path
import pyttsx3 # type: ignore
# endregion
# wget https://github.com/pytorch/vision/archive/refs/tags/v0.17.2.zip
# unzip v0.17.2.zip
# cd vision-0.17.2
# python setup.py install
# 

# pip install --no-cache-dir \
# torch==2.8.0 torchvision==0.23.0 \
# --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# pip install facenet-pytorch --no-deps

# cd ~/ai-img-detection-game
# git clone https://github.com/opencv/opencv.git
# git clone https://github.com/opencv/opencv_contrib.git
# cd opencv
# git checkout 4.9.0   # or the latest stable
# cd ../opencv_contrib
# git checkout 4.9.0


# cd ~/ai-img-detection-game/opencv
# mkdir build
# cd build

# davidbabel.clever
# jeff-hykin.mario
# MurlocCra4ler.bettersearch

# pip uninstall -y numpy
# pip install "numpy<2"

# pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# cmake -D CMAKE_BUILD_TYPE=Release \
# -D CMAKE_INSTALL_PREFIX=$HOME/.venv \
# -D PYTHON3_EXECUTABLE=$HOME/.venv/bin/python \
# -D PYTHON3_INCLUDE_DIR=$HOME/.venv/include/python3.10 \
# -D PYTHON3_PACKAGES_PATH=$HOME/.venv/lib/python3.10/site-packages \
# -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
# -D BUILD_opencv_python3=ON \
# -D WITH_GSTREAMER=ON \
# -D WITH_V4L=ON \
# -D WITH_CUDA=ON \
# -D ENABLE_NEON=ON \
# -D WITH_QT=OFF \
# -D BUILD_TESTS=OFF \
# -D BUILD_EXAMPLES=OFF ..
# 
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
  print("Device:", torch.cuda.get_device_name(0))


# region start
# normalize known vectors for cosine sim
def l2norm(x):
  return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


# F
class f:
  @staticmethod
  def read(
    file,
    default="",
    asbinary=False,
    buffering: int = -1,
    encoding: Any = None,
    errors: Any = None,
    newline: Any = None,
    closefd: bool = True,
    opener=None,
  ):
    if Path(file).exists():
      with open(
        file,
        "r" + ("b" if asbinary else ""),
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
        closefd=closefd,
        opener=opener,
      ) as f:
        text = f.read()
      if text:
        return text
      return default
    else:
      with open(
        file,
        "w" + ("b" if asbinary else ""),
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
        closefd=closefd,
        opener=opener,
      ) as f:
        f.write(default)
      return default

  @staticmethod
  def writeCsv(file, rows):
    with open(file, "w", encoding="utf-8", newline="") as f:
      w = csv.writer(f)
      w.writerows(rows)
    return rows

  @staticmethod
  def write(
    file,
    text,
    asbinary=False,
    buffering: int = -1,
    encoding: Any = None,
    errors: Any = None,
    newline: Any = None,
    closefd: bool = True,
    opener=None,
  ):
    with open(
      file,
      "w" + ("b" if asbinary else ""),
      buffering=buffering,
      encoding=encoding,
      errors=errors,
      newline=newline,
      closefd=closefd,
      opener=opener,
    ) as f:
      f.write(text)
    return text

  @staticmethod
  def append(
    file,
    text,
    asbinary=False,
    buffering: int = -1,
    encoding: Any = None,
    errors: Any = None,
    newline: Any = None,
    closefd: bool = True,
    opener=None,
  ):
    with open(
      file,
      "a",
      buffering=buffering,
      encoding=encoding,
      errors=errors,
      newline=newline,
      closefd=closefd,
      opener=opener,
    ) as f:
      f.write(text)
    return text

  @staticmethod
  def writeline(
    file,
    text,
    buffering: int = -1,
    encoding: Any = None,
    errors: Any = None,
    newline: Any = None,
    closefd: bool = True,
    opener=None,
  ):
    with open(
      file,
      "a",
      buffering=buffering,
      encoding=encoding,
      errors=errors,
      newline=newline,
      closefd=closefd,
      opener=opener,
    ) as f:
      f.write("\n" + text)
    return text


# A blank image encoded in base64, used as a placeholder
BLANK_IMAGE = (
  "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="
)

# Initialize Eel, a Python library for creating simple Electron-like desktop apps
eel.init("web")

# Variable to hold the capture object; initially set to 0
cap: Any = 0
MATCH_THRESHOLD = 0.55
DB_PATH = "data/embeddings_db.npz"
TARGET_CONFIDENCE = 0.75
mtcnn: Any = None
# Flag to determine whether to save the current frame
saveFrame = False
# Index of the camera to use
capidx = 1


# Log messages to the console and the front end
def log(*msgs):
  print(*msgs)
  eel.print(*msgs)


# Expose a function to stop video capture from the camera
@eel.expose
def stopCapture():
  global cap
  if cap:
    log("stopping capture")
    cap = None # Release the camera resource


# Expose a JavaScript function to save the current frame
@eel.expose
def jsSaveFrame():
  global saveFrame
  saveFrame = True # Set the flag to save the frame


# Expose JavaScript function to set minimum confidence level for detection
@eel.expose
def jsSetminconfidence(val):
  global MATCH_THRESHOLD
  MATCH_THRESHOLD = float(val) # Update minimum confidence with the new value
  log("MATCH_THRESHOLD set to " + str(val))


# Expose function to request updated settings/data to be sent to JavaScript
@eel.expose
def requestUpdatedData():
  # Send current configuration to the front end
  eel.loadData(
    {
      "captureIdx": capidx,
      "setminconfidenceInput": MATCH_THRESHOLD,
      "gameSpeed": gameSpeed,
    }
  )
@eel.expose
def setGameSpeed(v):
  global gameSpeed
  gameSpeed=v
gameSpeed=1


def gstreamer_pipeline(sensor_id=0, width=640, height=480, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        # We target BGRx directly and let videoconvert do the final step 
        # with a small queue to prevent blocking
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=True max-buffers=1"
    )
    # sudo sh -c 'echo "/usr/lib/aarch64-linux-gnu/tegra-egl" > /etc/ld.so.conf.d/nvidia-tegra-egl.conf && ldconfig'   

# Expose a function to start capturing video from the specified camera
@eel.expose
def startCapture(idx):
  global cap, capidx
  stopCapture()
  # pipeline = (
  #   "nvarguscamerasrc sensor-id=0 ! "
  #   "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 ! "
  #   "nvvidconv ! "
  #   "video/x-raw,format=BGRx ! "
  #   "videoconvert ! "
  #   "video/x-raw,format=BGR ! "
  #   "appsink drop=1 sync=false"
  # )

  # cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
  pipeline = gstreamer_pipeline(sensor_id=0, framerate=30)
  cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


  # idx = int(idx) # Convert the input index to an integer
  # log(f"Attempting to start capture on camera index: {idx}")
  # capidx = idx # Set the camera index to the global variable
  # cap = cv2.VideoCapture(idx) # Initialize the VideoCapture object
  if not cap.isOpened():
    log(
      f"Failed to open camera with index {idx}. Please check the index and try again."
    ) # Log error if camera fails to open
  else:
    log(f"camera with index {idx} was successfully opened") # Log success


def match_identity(embedding_vec):
  """
  Compare embedding_vec (512,) to known embeddings via cosine similarity.
  Return (best_name, best_score) or (None, None)
  """
  # normalize the candidate to unit length
  cand = embedding_vec / (np.linalg.norm(embedding_vec) + 1e-10)
  # cosine sim = dot product since both normalized
  sims = known_norm.dot(cand) # shape (N,)
  best_idx = np.argmax(sims)
  best_score = sims[best_idx]
  best_name = known_labels[best_idx]
  if best_score >= MATCH_THRESHOLD:
    return best_name, float(best_score)
  else:
    return None, None
import queue
frame_queue = queue.Queue(maxsize=1)

def worker():
    while True:
        frame = frame_queue.get() # Wait for a frame
        if frame is None: break
        
        try:
            # All the heavy lifting happens here, outside the main loop
            small_frame = cv2.resize(frame, (480, 360)) # Lower res = Much faster
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
            _, buffer = cv2.imencode(".jpg", small_frame, encode_param)
            
            encoded_frame = base64.b64encode(buffer).decode("utf-8")
            eel.receive_frame("data:image/jpeg;base64," + encoded_frame)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            frame_queue.task_done()

# 2. Start the worker thread ONCE at the start of your app
Thread(target=worker, daemon=True).start()

# Function to send a blank frame to avoid blank display
def sendBlankFrame():
  eel.receive_frame(BLANK_IMAGE) # Send the blank image
  time.sleep(0.1) # Sleep briefly to reduce CPU load


# Start the Eel application in a new thread
Thread(
  target=lambda: eel.start(
    mode=None, port=15674, close_callback=lambda *x: os._exit(0), shutdown_delay=10
  )
).start()
os.system("xdg-open http://127.0.0.1:15674/gameWeb.html")


# Function to format numbers into a specific format
def toPlaces(num: Any, pre=0, post=0, func=round):
  """Function to format numbers into a specific format

  Args:
    num (Any): number to format
    pre (int, optional): about of places before .. Defaults to 0.
    post (int, optional): amount of places after .. Defaults to 0.
    func (func, optional): function to use for trimming decimal places. Defaults to round.

  Returns:
    str: of the number formatted to the desired place counts
  """
  # Split the number into integer and decimal parts
  num = str(num).split(".")

  if len(num) == 1:
    num.append("") # Add empty decimal part if not present

  if pre is not None:
    # Keep only the last 'pre' digits of the integer part
    num[0] = num[0][-pre:]
    while len(num[0]) < pre: # Pad with zeros
      num[0] = "0" + num[0]

  # Extract the relevant decimal digit based on 'post'
  temp = num[1][post : post + 1] if len(num[1]) > post else "0"
  num[1] = num[1][:post] # Keep only first 'post' digits

  # Pad decimal part with zeros
  while len(num[1]) < post:
    num[1] += "0"

  if post > 0:
    # Round the last digit of the decimal part
    temp = func(float(num[1][-1] + "." + temp))
    num[1] = list(num[1])
    num[1][-1] = str(temp)
    num[1] = "".join(num[1])
    num = ".".join(num) # Combine back into single string
  else:
    num = num[0]

  return num


def reset():
  global lastFace, deathBoxList, speed, score, deathPosRand, stopped, highScore, size, dirRand
  lastFace = None
  deathBoxList = []
  speed = 3
  gameScore = 0
  deathPosRand = JitteredHaltonRNG()
  # spawnNewDeathRand = BalancedRand(0, 1, 0.1, 0.5)
  dirRand = BalancedRand(0, 3, 0.1, 0.5)
  stopped = False
  try:
    highScore = int(f.read("./highScore.txt", "0"))
  except Exception as e:
    highScore = -1
  size = 35


autoReset = False
lastFace: Any = 0
deathBoxList: Any = 0
speed: Any = 0
gameScore: Any = 0
deathPosRand: Any = 0
# spawnNewDeathRand: Any = 0
stopped: Any = 0
highScore: Any = 0
size: Any = 0
dirRand: Any = 0

reset()


def collides(x, y, w, h, face):
  x2, y2, w2, h2 = face
  h2 -= y2
  w2 -= x2
  return not (x >= x2 + w2 or x + w <= x2 or y >= y2 + h2 or y + h <= y2)


import tempfile


def updateFacesList():
  global mtcnn, known_norm, resnet, device, known_embeddings, known_labels
  try:
    # enroll_faces.init(log, eel.setProg)
    log("started loading new file")
    # if not os.path.exists(DB_PATH) and os.path.exists(DB_PATH + ".backup"):
    #   os.rename(DB_PATH + ".backup", DB_PATH)
    # with tempfile.NamedTemporaryFile(delete=False) as temp_db:
    #   log(temp_db.name)
    #   # f.write(temp_db.name, f.read(DB_PATH, "", True), True)
    #   # if os.path.exists(DB_PATH + ".backup"):
    #   #   os.remove(DB_PATH + ".backup")
    #   # os.rename(DB_PATH, DB_PATH + ".backup")
    db = np.load(DB_PATH) # Load from the temporary location

    known_embeddings = db["embeddings"] # shape (N,512)
    known_labels = db["labels"] # shape (N,)
    # load models
    known_norm = l2norm(known_embeddings)
    print(torch.cuda.is_available(), "torch.cuda.is_available()")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=20,min_face_size=40, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    log("done loading new file")
    import gc

    # After loading the model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  except Exception as e:
    log(e)
  eel.hideProg()

import threading

# Global storage for the latest "known" face data
last_detected_faces = [] 
ai_is_busy = False

# def run_ai_inference(frame_rgb):
#     global last_detected_faces, ai_is_busy
#     try:
#         boxes, probs = mtcnn.detect(frame_rgb)
#         if boxes is not None:
#             face_crops = []
#             valid_boxes = []
#             for box, prob in zip(boxes, probs):
#                 if prob is None or prob < 0.9: continue
#                 x1, y1, x2, y2 = [max(0, int(v)) for v in box]
#                 face_crops.append(frame_rgb[y1:y2, x1:x2])
#                 valid_boxes.append((x1, y1, x2, y2))
            
#             if face_crops:
#                 embeddings = get_embeddings_batched(face_crops)
#                 results = []
#                 for i, emb in enumerate(embeddings):
#                     name, score = match_identity(emb)
#                     results.append({"name": name, "score": score, "box": valid_boxes[i]})
#                 last_detected_faces = results # Update global state
#     finally:
#         ai_is_busy = False # Unlock so the next frame can be processed


def comstr(item: Any) -> str:
  reg = [r"(?<=\d)(\d{3}(?=(?:\d{3})*(?:$|\.)))", r",\g<0>"]
  if item is float:
    return (
      re.sub(reg[0], reg[1], str(item).split(".")[0])
      + "."
      + str(item).split(".")[1]
    )
  return re.sub(reg[0], reg[1], str(item))


engine = pyttsx3.init()


def say(msg):
  def _say():
    # pythoncom.CoInitialize()
    if engine._inLoop:
      engine.endLoop()
    engine.say(msg)
    engine.runAndWait()
    engine.stop()

  log(msg)
  # Thread(target=_say).start()

def get_embeddings_batched(face_crops):
    """
    Processes all detected faces in a single GPU pass.
    face_crops: List of RGB numpy arrays
    """
    if not face_crops:
        return []

    # 1. Pre-process all crops (Resize & Normalize)
    tensors = []
    for crop in face_crops:
        # Resize to MTCNN expected size (160x160)
        img = cv2.resize(crop, (160, 160))
        # Convert to tensor and normalize (standard for InceptionResnetV1)
        img = (torch.tensor(img).permute(2, 0, 1).float() - 127.5) / 128.0
        tensors.append(img)
    
    # Create a batch: shape (N, 3, 160, 160)
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        # RUN INFERENCE ONCE for all faces
        embs = resnet(batch)
    
    return embs.cpu().numpy()

# endregion
highScoreOwner = f.read("./highScorename.txt", "")
faceName = None
updateFacesList()
prev_time: float = time.time()
gameScores: Any = {}
shouldSayNewHighScores: Any = {}
spawnCount = 0.0
lastActiveTimes: Any = {}
import threading
import queue

# Global State for Async AI
last_detected_faces = []
ai_is_busy = False
frame_queue = queue.Queue(maxsize=1)

# --- 1. THE WORKER: Encodes and sends to Eel ---
def eel_worker():
    while True:
        frame = frame_queue.get()
        if frame is None: break
        try:
            # Send at a fixed low-res for the web UI
            send_small = cv2.resize(frame, (640, 480))
            _, buffer = cv2.imencode(".jpg", send_small, [cv2.IMWRITE_JPEG_QUALITY, 35])
            encoded = base64.b64encode(buffer).decode("utf-8")
            eel.receive_frame("data:image/jpeg;base64," + encoded)
        except Exception as e:
            print(f"Eel Worker Error: {e}")
        finally:
            frame_queue.task_done()

threading.Thread(target=eel_worker, daemon=True).start()

# --- 2. THE AI WORKER: Processes embeddings without blocking the game ---
def run_ai_inference(frame_rgb_small, scale_w, scale_h):
    global last_detected_faces, ai_is_busy
    try:
        boxes, probs = mtcnn.detect(frame_rgb_small)
        results = []
        if boxes is not None:
            face_crops = []
            valid_boxes = []
            for box, prob in zip(boxes, probs):
                if prob is None or prob < 0.9: continue
                # Scale coordinates back to original frame size
                x1, y1, x2, y2 = [int(box[0] * scale_w), int(box[1] * scale_h), 
                                  int(box[2] * scale_w), int(box[3] * scale_h)]
                
                # Clip coordinates to avoid array errors
                crop = frame_rgb_small[max(0, int(box[1])):int(box[3]), max(0, int(box[0])):int(box[2])]
                if crop.size > 0:
                    face_crops.append(crop)
                    valid_boxes.append((x1, y1, x2, y2))

            if face_crops:
                embeddings = get_embeddings_batched(face_crops)
                for i, emb in enumerate(embeddings):
                    name, score = match_identity(emb)
                    results.append({"name": name, "score": score, "box": valid_boxes[i]})
        
        last_detected_faces = results
    finally:
        ai_is_busy = False

# --- 3. THE MAIN LOOP: Runs at max FPS ---
while True:
    if not cap or not cap.isOpened():
        sendBlankFrame()
        continue

    curr_time = time.time()
    delta = curr_time - prev_time
    fps = 1 / max(delta, 0.0001)
    prev_time = curr_time

    ret, frame = cap.read()
    if not ret: continue

    frame = cv2.flip(frame, 0)
    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # A. ASYNC AI TRIGGER: Only run if AI thread is free
    if not ai_is_busy:
        ai_is_busy = True
        # Speed hack: Detect on a tiny image
        ai_w, ai_h = 320, 240
        small_for_ai = cv2.resize(frame_rgb, (ai_w, ai_h))
        run_ai_inference(small_for_ai, width/ai_w, height/ai_h)
        # threading.Thread(target=run_ai_inference, 
        #                  args=(small_for_ai, width/ai_w, height/ai_h), 
        #                  daemon=True).start()

    # B. GAME LOGIC: Always runs, using 'last_detected_faces'
    # Update Red Death Boxes
    spawnCount += 0.1*gameSpeed # Adjust spawn rate as needed
    while spawnCount > 1:
      spawnCount -= 1
      diridx = int(round(dirRand.next()))
      dir = [[0, 1], [0, -1], [1, 0], [-1, 0]][diridx]
      deathBox = [
        0,
        0,
        int(size),
        int(size),
        dir,
        speed,
      ]
      if diridx == 0:
        deathBox[1] = 0
        deathBox[0] = int(deathPosRand.next()*height)
      elif diridx == 1:
        deathBox[1] = height - size
        deathBox[0] = int(deathPosRand.next()*width)
      elif diridx == 2:
        deathBox[0] = 0
        deathBox[1] = int(deathPosRand.next()*height)
      elif diridx == 3:
        deathBox[0] = width - size
        deathBox[1] = int(deathPosRand.next()*width)
      s = deathBox[4]
      s[0] *= deathBox[5]
      s[1] *= deathBox[5]
      deathBoxList.append(deathBox)
    
    # Update positions and Draw Death Boxes
    for db in deathBoxList:
        db[0] += db[4][0] * db[5]*gameSpeed # x += dir_x * speed
        db[1] += db[4][1] * db[5]*gameSpeed # y += dir_y * speed
        cv2.rectangle(frame, (int(db[0]), int(db[1])), (int(db[0]+db[2]), int(db[1]+db[3])), (0, 0, 255), 2)
    # C. COLLISION & SCORING: Process every face currently tracked
    for face in last_detected_faces:
        name = face["name"]
        score = face["score"]
        x1, y1, x2, y2 = face["box"]
        
        if name:
            lastActiveTimes[name] = curr_time
            collision = False
            graze = 0
            grazeSize = 5
            
            # Check collision against deathBoxList
            for x, y, w, h, dir, speed in deathBoxList:
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                facePos = (x1, y1, x2, y2)
                if collides(x, y, w, h, facePos):
                  if (
                    (
                      name not in shouldSayNewHighScores
                      or not shouldSayNewHighScores[name]
                    )
                    and highScoreOwner == name
                    and name in gameScores
                    and highScore == gameScores[name]
                  ):
                    say(
                      name
                      + " just lost with a new highscore of "
                      + str(int(highScore))
                    )
                  gameScores[name] = 0
                  shouldSayNewHighScores[name] = True
                  collision = True
                  break
                elif collides(
                  x - grazeSize,
                  y - grazeSize,
                  w + (grazeSize - 2),
                  h + (grazeSize * 2),
                  facePos,
                ):
                  graze = 2
                elif collides(
                  x - (grazeSize * 2),
                  y - (grazeSize * 2),
                  w + (grazeSize * 4),
                  h + (grazeSize * 4),
                  facePos,
                ):
                  graze = 0.3

            if not collision:
                gameScores[name] = gameScores.get(name, 0) + ((y2 / 3) * delta) * (graze + 1)
            color = (0, 255, 0)
            if collision:
              color = (0, 0, 255)
            elif graze == 0.3:
              color = (0, 192, 255)
            elif graze == 2:
              color = (0, 128, 255)

            textSize = 0.6
            hasHighScore = name == highScoreOwner
            gettingHighScore = hasHighScore and gameScores[name] >= highScore
            textColor = (255, 255, 255)
            if gettingHighScore or hasHighScore:
              textSize = 0.8
            if gettingHighScore:
              textColor = (192, 0, 255)
            elif hasHighScore:
              textColor = (255, 192, 0)
            # Draw border (black) in 4 directions
            cv2.putText(
              frame,
              str(int(gameScores[name])),
              (x1 - 1, y1 - 30),
              cv2.FONT_HERSHEY_SIMPLEX,
              textSize,
              (0, 0, 0),
              4,
              cv2.LINE_AA,
            )
            cv2.putText(
              frame,
              str(int(gameScores[name])),
              (x1 + 1, y1 - 30),
              cv2.FONT_HERSHEY_SIMPLEX,
              textSize,
              (0, 0, 0),
              4,
              cv2.LINE_AA,
            )
            cv2.putText(
              frame,
              str(int(gameScores[name])),
              (x1, y1 - 31),
              cv2.FONT_HERSHEY_SIMPLEX,
              textSize,
              (0, 0, 0),
              4,
              cv2.LINE_AA,
            )
            cv2.putText(
              frame,
              str(int(gameScores[name])),
              (x1, y1 - 29),
              cv2.FONT_HERSHEY_SIMPLEX,
              textSize,
              (0, 0, 0),
              4,
              cv2.LINE_AA,
            )

            # Draw main text on top
            cv2.putText(
              frame,
              str(int(gameScores[name])),
              (x1, y1 - 30),
              cv2.FONT_HERSHEY_SIMPLEX,
              textSize,
              textColor,
              2,
              cv2.LINE_AA,
            )
            # endregion

            # D. DRAWING: Render UI on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name}: {toPlaces(score, 1, 2)}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # region update scores and highscores
            for scorereName, gameScore in gameScores.items():
              if int(gameScore) > highScore:
                highScore = gameScore
                if highScoreOwner:
                  if scorereName != highScoreOwner:
                    say(
                      scorereName
                      + " overtook "
                      + highScoreOwner
                      + " with a score of "
                      + str(int(gameScore))
                    )
                  else:
                    if (
                      scorereName in shouldSayNewHighScores
                      and shouldSayNewHighScores[scorereName]
                    ):
                      say(
                        scorereName
                        + " got a new high score of "
                        + str(int(gameScore))
                      )
                      shouldSayNewHighScores[scorereName] = False
                highScoreOwner = str(scorereName)
                f.write("./highScore.txt", str(int(gameScore)))
                f.write("./highScorename.txt", str(scorereName))
          # endregion

    # E. CLEANUP & SEND
    deathBoxList = [db for db in deathBoxList if 0 <= db[0] <= width and 0 <= db[1] <= height]
      
    cv2.putText(
      frame,
      "FPS: " + toPlaces(fps, 2, 3),
      (20, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 255, 255),
      2,
    )
    eel.setHighscoreMessage( # type: ignore
      "HIGH SCORE: " + comstr(int(highScore)) + " by " + highScoreOwner,
    )

    # Push to queue for the Eel worker to pick up
    # if frame_queue.full():
    #     try: frame_queue.get_nowait()
    #     except: pass
    frame_queue.put(frame)