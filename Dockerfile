FROM ubuntu:20.04

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install xvfb xorg git wget pip python3-pip libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev -y
RUN apt-get install curl -y
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b
RUN rm Miniforge3-$(uname)-$(uname -m).sh 

ENV PATH="/root/miniforge3/bin:${PATH}"

ADD conda_environment.yaml /tmp/conda_environment.yaml
RUN mamba env create -f /tmp/conda_environment.yaml

SHELL ["conda", "run", "-n", "se3diffuser", "/bin/bash", "-c"]
# make installs directory
RUN mkdir -p /installs
WORKDIR /installs
RUN git clone https://github.com/stepjam/PyRep.git

RUN wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
RUN tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
RUN rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

ENV COPPELIASIM_ROOT=/installs/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

RUN apt-get install python3-pip -y
RUN pip3 install setuptools

WORKDIR /installs/PyRep
RUN pip3 install -r requirements.txt
RUN python3 setup.py develop


WORKDIR /installs
RUN git clone -b peract https://github.com/MohitShridhar/RLBench.git # note: 'peract' branch
WORKDIR /installs/RLBench
RUN pip3 install -r requirements.txt
RUN pip3 install absl-py
RUN python3 setup.py develop

# Set the working directory
WORKDIR /workspace

RUN echo "source activate se3diffuser" > ~/.bashrc

RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get install -y g++-11

ENV DIFFUSION_POLICY_ROOT=/workspace/

# Specify the command to run on start
CMD [ "bash" ]
