#!/usr/bin/env fish

# ===========================
# OpenCV 4.9.0 install for Jetson + .venv
# ===========================

set BASE_DIR ~/ai-img-detection-game
set VENV_DIR $BASE_DIR/.venv
set OPENCV_DIR $BASE_DIR/opencv
set OPENCV_CONTRIB_DIR $BASE_DIR/opencv_contrib
set BUILD_DIR $OPENCV_DIR/build
set OPENCV_DIR $BASE_DIR/opencv

# ---------------------------
# Step 1: Download OpenCV & contrib if not present
# ---------------------------
if not test -d $OPENCV_DIR
    echo "Downloading OpenCV..."
    curl -L -o $BASE_DIR/opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.9.0.zip
    unzip $BASE_DIR/opencv.zip -d $BASE_DIR
    mv $BASE_DIR/opencv-4.9.0 $OPENCV_DIR
end

if not test -d $OPENCV_CONTRIB_DIR
    echo "Downloading OpenCV contrib..."
    curl -L -o $BASE_DIR/opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.9.0.zip
    unzip $BASE_DIR/opencv_contrib.zip -d $BASE_DIR
    mv $BASE_DIR/opencv_contrib-4.9.0 $OPENCV_CONTRIB_DIR
end

# ---------------------------
# Step 2: Prepare build directory
# ---------------------------
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# ---------------------------
# Step 3: Run CMake
# ---------------------------

cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=(python -c "import sys; print(sys.prefix)") \
-D WITH_GSTREAMER=ON \
-D PYTHON3_EXECUTABLE=(which python) ..

  

# ---------------------------
# Step 4: Build & install
# ---------------------------
make -j (nproc)
make install

# ---------------------------
# Step 5: Verify
# ---------------------------
source $VENV_DIR/bin/activate.fish
python -c "import cv2; print('OpenCV version:', cv2.__version__); print(cv2.getBuildInformation())"