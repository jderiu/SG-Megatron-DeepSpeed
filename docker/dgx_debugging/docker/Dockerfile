FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt install -y tzdata


# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      libsndfile1 \
	  libxext6 \
      ffmpeg \
	  nano \
      wget && \
    rm -rf /var/lib/apt/lists/*


# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh &&  /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh
	
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --use-deprecated=legacy-resolver --no-cache-dir -r requirements.txt 


RUN git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#RUN git clone https://github.com/microsoft/deepspeed && cd deepspeed && rm -rf build && DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check

RUN wget --quiet --no-check-certificate https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz 
RUN tar -zxf openmpi-4.1.5.tar.gz && cd openmpi-4.1.5 && ./configure --prefix=/usr/local/openmpi && make all && make install
ENV MPI_DIR=/usr/local/openmpi
RUN export LD_LIBRARY_PATH=$MPI_DIR/lib:$LD_LIBRARY_PATH
RUN conda install -y mpi4py  
RUN pip install numpy==1.23

# Install ssh
RUN apt-get update && apt-get install -y openssh-server openssh-client

RUN mkdir /var/run/sshd
RUN echo 'root:pw' | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

# Copy authorized keys
ADD ssh_keys.tar.gz /tmp/
RUN mkdir -p /root/.ssh \
    && mv /tmp/ssh_keys/authorized_keys /root/.ssh/authorized_keys \
    && chown root:root /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/authorized_keys \
    && stat /root/.ssh/authorized_keys \
&& rm -rf /tmp/ssh_keys ssh_keys.tar.gz
