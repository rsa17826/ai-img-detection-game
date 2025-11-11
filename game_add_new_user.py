import os

print("changing dir to ", os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
import cv2
from typing import Any
import random, time, re
from biasRand import BalancedRand
import eel
from threading import Thread
import base64
import enroll_faces
from facenet_pytorch import MTCNN, InceptionResnetV1 # type: ignore
import numpy as np
from pathlib import Path
import time
from collections import deque
from typing import Dict


class RecentAverage:
  def __init__(self, max_size=20):
    self.values = deque(maxlen=max_size) # Store values with a maximum length
    self.total = 0 # Sum of values for average calculation

  def registerValue(self, value):
    if len(self.values) == self.values.maxlen: # Check if deque is full
      # Subtract the value that will be removed
      self.total -= self.values[0]
    self.values.append(value) # Register the new value
    self.total += value # Update total

  def getAverage(self):
    if not self.values: # Check if there are no values to average
      return 0
    return self.total / len(self.values) # Calculate the average


class DecayingAverage:
  def __init__(self):
    self.values = deque() # Store values as (timestamp, value) pairs
    self.total = 0 # Sum of values for average calculation
    self.count = 0 # Number of active values

  def registerValue(self, value):
    current_time = time.time()
    self.values.append((current_time, value)) # Register the current time and value
    self.total += value
    self.count += 1

  def decayValues(self):
    current_time = time.time()
    while self.values and (
      current_time - self.values[0][0] > 10
    ): # Check for decay
      old_time, old_value = self.values.popleft()
      self.total -= old_value
      self.count -= 1

  def getAverage(self):
    self.decayValues() # Clean up old values before calculating average
    if self.count == 0:
      return 0 # Return 0 if there are no values to average
    return self.total / self.count


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
MATCH_THRESHOLD = 0.65
DB_PATH = "data/embeddings_db.npz"
enableAutoCapture = False
TARGET_CONFIDENCE = 0.7
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


# Expose function to request updated settings/data to be sent to JavaScript
@eel.expose
def updateHtmlData():
  # Send current configuration to the front end
  eel.loadData({"captureIdx": capidx, "enableAutoCapture": enableAutoCapture})


# Expose JavaScript function to set minimum confidence level for detection
@eel.expose
def setenableAutoCapture(val):
  global enableAutoCapture
  enableAutoCapture = float(val) # Update minimum confidence with the new value
  log("enableAutoCapture set to " + str(val))


# Expose a function to start capturing video from the specified camera
@eel.expose
def startCapture(idx):
  global cap, capidx
  idx = int(idx) # Convert the input index to an integer
  log(f"Attempting to start capture on camera index: {idx}")
  capidx = idx # Set the camera index to the global variable
  cap = cv2.VideoCapture(idx) # Initialize the VideoCapture object
  if not cap.isOpened():
    log(
      f"Failed to open camera with index {idx}. Please check the index and try again."
    ) # Log error if camera fails to open
  else:
    log(f"camera with index {idx} was successfully opened") # Log success


def get_embedding(face_img_rgb):
  """
  face_img_rgb: np array (H,W,3) RGB cropped face region
  returns: (512,) embedding or None if detection failed
  """
  # detect/align just this face via mtcnn on the crop
  face_tensor = mtcnn(face_img_rgb)

  if face_tensor is None:
    return None

  # Convert to proper shape
  face_tensor = face_tensor.squeeze(0).to(device) # Remove the batch dimension

  with torch.no_grad():
    emb = resnet(
      face_tensor.unsqueeze(0)
    ) # Add the batch dimension back for processing
  emb = emb.squeeze(0).cpu().numpy()
  return emb


def match_identity(embedding_vec):
  """
  Compare embedding_vec (512,) to known embeddings via cosine similarity.
  Return (best_name, best_score) or (None, None)
  """
  # normalize the candidate to unit length
  cand = embedding_vec / (np.linalg.norm(embedding_vec) + 1e-10)
  # cosine sim = dot product since both normalized
  sims = known_norm.dot(cand) # shape (N,)
  if not sims.size:
    return None, None
  best_idx = np.argmax(sims)
  best_score = sims[best_idx]
  best_name = known_labels[best_idx]
  if best_score >= MATCH_THRESHOLD:
    return best_name, float(best_score)
  else:
    return None, None


# Function to send the current video frame to the front end
def send_frame(frame):
  # Convert the frame to a format suitable for web
  _, buffer = cv2.imencode(".jpg", frame) # Encode frame as JPEG
  frame_bytes = buffer.tobytes() # Get bytes from the buffer
  encoded_frame = base64.b64encode(frame_bytes).decode("utf-8") # Base64 encode
  # Send the encoded frame to the JavaScript frontend
  eel.receive_frame("data:image/jpeg;base64," + encoded_frame)


# Function to send a blank frame to avoid blank display
def sendBlankFrame():
  eel.receive_frame(BLANK_IMAGE) # Send the blank image
  time.sleep(0.1) # Sleep briefly to reduce CPU load


# Start the Eel application in a new thread
Thread(
  target=lambda: eel.start(
    mode=None, port=15675, close_callback=lambda *x: os._exit(0), shutdown_delay=10
  )
).start()

os.system("start http://127.0.0.1:15675/gameWebNewUser.html")


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


def collides(x, y, w, h, face):
  x2, y2, w2, h2 = face
  h2 -= y2
  w2 -= x2
  return not (x >= x2 + w2 or x + w <= x2 or y >= y2 + h2 or y + h <= y2)


known_labels: Any = []


@eel.expose
def updateFacesList():
  global mtcnn, known_norm, resnet, device, db, known_embeddings, known_labels
  try:
    enroll_faces.init(log, eel.setProg)
    db = np.load(DB_PATH)
    known_embeddings = db["embeddings"] # shape (N,512)
    known_labels = db["labels"] # shape (N,)
    # load models
    known_norm = l2norm(known_embeddings)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
  except Exception as e:
    log(e)
  eel.hideProg()


def comstr(item: Any) -> str:
  reg = [r"(?<=\d)(\d{3}(?=(?:\d{3})*(?:$|\.)))", r",\g<0>"]
  if item is float:
    return (
      re.sub(reg[0], reg[1], str(item).split(".")[0])
      + "."
      + str(item).split(".")[1]
    )
  return re.sub(reg[0], reg[1], str(item))


@eel.expose
def addFaceToList(val):
  global faceName
  faceName = val
  enableAutoCapture=True
  updateHtmlData()
  log("faceName set to " + val)


# endregion
faceName = None
avgs: Dict[str, RecentAverage] = {}
updateFacesList()
prev_time: float = time.time()
while True:
  if not cap or not cap.isOpened():
    sendBlankFrame()
    continue
  curr_time = time.time()
  fps = 1 / max(curr_time - prev_time, 0.0001)
  prev_time = curr_time
  # region Capture a frame from the camera
  ret, frame = cap.read()
  if not ret:
    log(frame, ret)
    continue
  frame = cv2.flip(frame, 1) # Flip the frame for a mirror effect
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  rawframe_bgr = frame.copy()
  facePos = None
  cv2.putText(
    frame,
    "FPS: " + toPlaces(fps, 2, 3),
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
  )
  # endregion
  if mtcnn:
    foundUnknownFace = False
    boxes, probs = mtcnn.detect(frame_rgb)
    if boxes is not None:
      for box, prob in zip(boxes, probs):
        # region what face is that
        if prob is None:
          continue
        x1, y1, x2, y2 = [int(v) for v in box]
        face_crop_rgb = frame_rgb[y1:y2, x1:x2]
        if face_crop_rgb.size == 0:
          continue
        emb = None
        try:
          emb = get_embedding(face_crop_rgb)
        except Exception as e:
          continue
        if emb is None:
          continue
        try:
          name, score = match_identity(emb)
        except Exception as e:
          log(e)
          continue
        # endregion
        # region draw face box and name
        label_text = "Unknown"
        color = (0, 0, 255) # red in BGR
        if name is not None:
          label_text = f"{name}:"
          color = (0, 255, 0) # green
        else:
          foundUnknownFace = True
        facePos = [x1, y1, x2, y2]
        # if name:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # endregion
        # region capture face
        if (
          faceName
          and foundUnknownFace
          and not name
          and faceName not in known_labels
        ):
          enableAutoCapture = True
          updateHtmlData()
        if (
          (
            enableAutoCapture
            and name
            and score
            and score < TARGET_CONFIDENCE
            and score > MATCH_THRESHOLD
          )
          or (
            faceName
            and foundUnknownFace
            and not name
            and faceName not in known_labels
          )
          or (faceName and name and faceName == name)
        ):
          send_frame(frame)
          if not name:
            name = faceName
          i = 0
          path = f"./players/{name}/{i}.png"
          os.makedirs(f"./players/{name}", exist_ok=True)
          while os.path.exists(path):
            i += 1
            path = f"./players/{name}/{i}.png"
          log("adding image for ", name, "idx: ", i)
          if facePos:
            frame_rgb_cropped = rawframe_bgr[
              max(0, facePos[1] - 15) : min(
                facePos[3] + 15, rawframe_bgr.shape[0]
              ),
              max(0, facePos[0] - 15) : min(
                facePos[2] + 15, rawframe_bgr.shape[1]
              ),
            ]
          else:
            frame_rgb_cropped = rawframe_bgr

          cv2.imwrite(
            path, frame_rgb_cropped
          ) # Save the current frame as an image
          faceName = None
          updateFacesList()
        # endregion
        if score and name:
          if name not in avgs:
            avgs[name] = RecentAverage()
          avgs[name].registerValue(score)
        if foundUnknownFace:
          break
        if len(avgs[name].values) < 10:
          color = (0, 0, 255)
          label_text += (
            " move your face around and show different angles "
            + str(len(avgs[name].values))
            + "/10"
          )
        elif avgs[name].getAverage() < 0.8:
          color = (0, 0, 255)
          label_text += (
            " get avg > .8 AVG: "
            + toPlaces(avgs[name].getAverage(), 1, 2)
            + " - CURRENT: "
            + toPlaces(score, 1, 2)
          )
        else:
          color = (0, 255, 0)
          label_text += (
            " ready to play ~"
            + toPlaces(avgs[name].getAverage() * 100, 3, 0)
            + "% detection rate"
            + " - CURRENT: "
            + toPlaces(score, 1, 2)
          )
          enableAutoCapture = False
        updateHtmlData()
        textSize = 0.55
        cv2.putText(
          frame,
          label_text,
          (5 - 1, y1 - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          textSize,
          (0, 0, 0),
          4,
          cv2.LINE_AA,
        )
        cv2.putText(
          frame,
          label_text,
          (5 + 1, y1 - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          textSize,
          (0, 0, 0),
          4,
          cv2.LINE_AA,
        )
        cv2.putText(
          frame,
          label_text,
          (5, y1 - 11),
          cv2.FONT_HERSHEY_SIMPLEX,
          textSize,
          (0, 0, 0),
          4,
          cv2.LINE_AA,
        )
        cv2.putText(
          frame,
          label_text,
          (5, y1 - 9),
          cv2.FONT_HERSHEY_SIMPLEX,
          textSize,
          (0, 0, 0),
          4,
          cv2.LINE_AA,
        )

        cv2.putText(
          frame,
          label_text,
          (5, y1 - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          textSize,
          color,
          2,
          cv2.LINE_AA,
        )
  send_frame(frame)
