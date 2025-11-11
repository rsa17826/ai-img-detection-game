import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1 # type:ignore
from typing import Any, Dict
import json
from pathlib import Path


class Cache:
  def __init__(self, name="cache"):
    self.__cache__ = {}
    self.name = name

  def has(self, thing):
    if hasattr(self, "lastThing"):
      raise Exception("[ERROR]: should not have last thing")
    self.lastThing = thing
    return thing in self.__cache__

  def get(self):
    if not hasattr(self, "lastThing"):
      raise Exception("[ERROR]: should have last thing")
    data = self.__cache__[self.lastThing]
    del self.lastThing
    return data

  def set(self, val):
    if not hasattr(self, "lastThing"):
      raise Exception("[ERROR]: should have last thing")
    self.__cache__[self.lastThing] = val
    del self.lastThing

  def saveToFile(self):
    os.makedirs("./.cache", exist_ok=True)

    def serialize(data):
      if isinstance(data, np.ndarray):
        return {"type": "ndarray", "data": data.tolist()}
      elif isinstance(data, dict):
        return {
          "type": "dict",
          "data": {key: serialize(value) for key, value in data.items()},
        }
      elif isinstance(data, list):
        return {"type": "list", "data": [serialize(item) for item in data]}
      else:
        return {"type": type(data).__name__, "data": data}

    formatted_cache = {key: serialize(val) for key, val in self.__cache__.items()}

    with open(f"./.cache/{self.name}", "w") as f:
      for key, item in formatted_cache.items():
        # Convert the item dictionary to a string representation
        f.write(
          f"{key}: {str(item)}\n"
        ) # Use str() to safely convert the dict to a string

  def loadFromFile(self) -> bool:
    os.makedirs("./.cache", exist_ok=True)
    self.__cache__ = {}

    cache_file = f"./.cache/{self.name}"
    if os.path.exists(cache_file):
      try:
        with open(cache_file, "r") as f:
          for line in f:
            key, value_str = line.strip().split(": ", 1)

            # Use safer parsing instead of eval
            value_data = eval(
              value_str
            ) # Still risky; consider a safer parsing method

            def deserialize(data):
              if isinstance(data, dict) and "type" in data:
                if data["type"] == "ndarray":
                  return np.array(data["data"])
                elif data["type"] == "dict":
                  return {
                    k: deserialize(v)
                    for k, v in data["data"].items()
                  }
                elif data["type"] == "list":
                  return [deserialize(item) for item in data["data"]]
              return data["data"]

            self.__cache__[key] = deserialize(value_data)

        return True
      except Exception as e:
        print(f"Loading error: {e}")
        return False
    return False


cache: Cache = Cache()
cache.loadFromFile()


def init(log, setProg=lambda *a: 1):
  DB_PATH = "data/embeddings_db.npz"

  os.makedirs("players", exist_ok=True)

  # 1. Load face detector + embedder
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
  resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

  all_embeddings: Any = []
  all_labels: Any = []
  prog = 0
  maxProg = 0

  for person_name in os.listdir("players"):
    person_folder = os.path.join("players", person_name)
    if not os.path.isdir(person_folder):
      continue

    innerList = os.listdir(person_folder)
    maxProg += len(innerList)

  setProg(prog, maxProg, "")
  for person_name in os.listdir("players"):
    person_folder = os.path.join("players", person_name)
    if not os.path.isdir(person_folder):
      continue

    innerList = os.listdir(person_folder)

    for img_file in innerList:
      img_path = os.path.join(person_folder, img_file)
      prog += 1
      setProg(prog, maxProg, person_name)

      # Check if the person is already players
      if cache.has(person_name + "/" + img_file):
        # log(
        #   f"[INFO] Skipping already processed image for: {person_name} ({img_file})"
        # )

        person_name, emb = cache.get()
        all_embeddings.append(emb)
        all_labels.append(person_name)
        continue

      # Read image with cv2 (BGR -> RGB)
      img_bgr = cv2.imread(img_path)
      if img_bgr is None:
        log(f"[WARN] Could not read {img_path}")
        del cache.lastThing
        continue
      img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

      # 2. Detect & crop face (returns PIL Image or None)
      face = mtcnn(img_rgb)
      if face is None:
        log(f"[WARN] No face found in {img_path}")
        del cache.lastThing
        os.remove(img_path)
        continue

      # Face is a torch tensor [3,160,160]
      face = face.unsqueeze(0).to(device) # [1,3,160,160]

      # 3. Get embedding (512-d vector)
      with torch.no_grad():
        emb = resnet(face) # [1,512]
      emb = emb.squeeze(0).cpu().numpy() # [512]

      all_embeddings.append(emb)
      all_labels.append(person_name)
      cache.set([person_name, emb])

  # Convert to arrays
  all_embeddings = np.array(all_embeddings) if all_embeddings else np.empty((0, 512))
  all_labels = np.array(all_labels) if all_labels else np.empty((0,))

  # Ensure the 'data' directory exists
  os.makedirs("data", exist_ok=True)
  np.savez(DB_PATH, embeddings=all_embeddings, labels=all_labels)

  # Log total faces players
  log(f"[INFO] Total faces players: {len(all_labels)}")
  log(f"[INFO] Saved database to {DB_PATH}")
  cache.saveToFile()
