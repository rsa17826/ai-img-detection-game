# activate venv 

source .venv/bin/activate.fish 

  

# install build tools 

sudo apt install build-essential cmake git pkg-config \ 

        libgstreamer1.0-dev gstreamer1.0-plugins-base \ 

        gstreamer1.0-plugins-good libgtk-3-dev 

  

# clone OpenCV 

git clone https://github.com/opencv/opencv.git 

cd opencv 

  

mkdir build 

cd build 

  

# configure build (Fish syntax) 

cmake -D CMAKE_BUILD_TYPE=Release \ 

          -D CMAKE_INSTALL_PREFIX=(python -c "import sys; print(sys.prefix)") \ 

          -D WITH_GSTREAMER=ON \ 

          -D PYTHON3_EXECUTABLE=(which python) .. 

  

# compile (can take a while) 

make -j(nproc) 

  

# install into your venv 

# activate venv 

source .venv/bin/activate 

  

# install build tools 

sudo apt install build-essential cmake git pkg-config \ 

        libgstreamer1.0-dev gstreamer1.0-plugins-base \ 

        gstreamer1.0-plugins-good libgtk-3-dev 

  

# clone OpenCV 

git clone https://github.com/opencv/opencv.git 

cd opencv 

  

mkdir build 

cd build 

  

# configure build (Fish syntax) 

cmake -D CMAKE_BUILD_TYPE=Release \ 

          -D CMAKE_INSTALL_PREFIX=(python -c "import sys; print(sys.prefix)") \ 

          -D WITH_GSTREAMER=ON \ 

          -D PYTHON3_EXECUTABLE=(which python) .. 

  

# compile (can take a while) 

make -j(nproc) 

  

# install into your venv 

make install 

  

  

  