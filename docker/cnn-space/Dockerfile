FROM nvcr.io/nvidia/pytorch:19.08-py3
# Note that you need one user account in nvcr to download this image.
MAINTAINER noname <noname@noaddress.aabbcc>

RUN "sh" "-c" "echo nameserver 8.8.8.8 >> /etc/resolv.conf"

##########################
#####  Docmuemts #########
##########################

# Build a a light environment with the correct user access to cvlabdata3
# No root access originally, can sudo to get it
# Create to support k8s deployment and pod commmand
# Hope it can work!

RUN "sh" "-c" "echo nameserver 8.8.8.8 >> /etc/resolv.conf"

## user: root permission
USER root

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Install python packages via conda.
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN conda install -y "conda>=4.4.11" && conda clean -ya
#################################################
# ADAPT this for your own project!!!
RUN conda install -y \
        h5py \
        matplotlib \
        numpy \
        pillow \
        graphviz \
        pyyaml \
        protobuf \
        scipy \
        tqdm \
        && \
    conda clean -ya
RUN pip install \
        tensorboardX

# Create user with UID=NB_UID and in the 'NB_GID' group
# and make sure these dirs are writable by the `NB_GID` group.
# Configure user and group

#################################################
# CHANGE THIS PART ACCORDINGLY !!!
ENV SHELL=/bin/bash \
    NB_USER=yourname \
    NB_UID=112233 \
    NB_GROUP=101 \
    NB_GID=1102
#################################################

ENV HOME=/home/$NB_USER

RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER}
RUN chown -R ${NB_USER}:${NB_GROUP} ${HOME}
#RUN chown $NB_USER:$NB_GID $CONDA_DIR
#################################################

# The user gets passwordless sudo
RUN echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers


## user: user permission
USER $NB_USER


RUN pip install networkx tensorflow==1.12.0
RUN cd /tmp && git clone https://github.com/google-research/nasbench && cd nasbench && pip install -e .
# install the pytorch-1.2 use cuda 10.0
RUN conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch \
    && conda clean -ya
