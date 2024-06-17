FROM nvcr.io/nvidia/dgl:24.04-py3

RUN apt-get update && apt-get upgrade -y
RUN apt-get install libffi-dev

# make installs directory
RUN mkdir -p /installs
WORKDIR /installs
RUN git clone https://github.com/stepjam/PyRep.git

RUN wget https://downloads.coppeliarobotics.com/V4_5_1_rev4/CoppeliaSim_Player_V4_5_1_rev4_Ubuntu22_04.tar.xz
RUN tar -xvf CoppeliaSim_Player_V4_5_1_rev4_Ubuntu22_04.tar.xz
RUN rm CoppeliaSim_Player_V4_5_1_rev4_Ubuntu22_04.tar.xz

ENV COPPELIASIM_ROOT=/installs/CoppeliaSim_Player_V4_5_1_rev4_Ubuntu22_04
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

WORKDIR /installs/PyRep
RUN pip install -r requirements.txt
RUN pip install .

WORKDIR /installs
RUN git clone -b peract https://github.com/MohitShridhar/RLBench.git # note: 'peract' branch
WORKDIR /installs/RLBench
RUN pip install -r requirements.txt
RUN python setup.py develop

WORKDIR /installs
RUN git clone -b peract https://github.com/MohitShridhar/YARR.git # note: 'peract' branch
WORKDIR /installs/YARR
RUN pip install -r requirements.txt
RUN pip install .

WORKDIR /workspace
RUN pip install git+https://github.com/openai/CLIP.git

RUN apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev -y
# RUN nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
RUN wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb
RUN dpkg -i virtualgl_2.5.2_amd64.deb
RUN rm virtualgl_2.5.2_amd64.deb
RUN nohup sudo X &
RUN export DISPLAY=:0.0
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility


# Set the working directory
WORKDIR /workspace

# Specify the command to run on start
CMD [ "bash" ]
