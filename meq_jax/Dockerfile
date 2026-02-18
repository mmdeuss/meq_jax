# Dockerfile for MEQ-JAX
# Uses an HTTPS Token for secure repository access during build.
# To build and run:
# 1. Get a GitLab token from https://gitlab.epfl.ch/-/user_settings/personal_access_tokens
# 2. Build: docker build --build-arg GITLAB_TOKEN=<your_token> -t meq_jax .
# 3. Run: docker run meq_jax

FROM debian:bookworm

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
# liboctave-dev is the standard name for Octave headers on Debian.
RUN apt-get update && apt-get install -y \
    octave \
    liboctave-dev \
    git \
    pkg-config \
    liblapacke-dev \
    libblas-dev \
    liblapack-dev \
    python3-tk \
    python3-pip \
    python3-venv \
    make \
    gcc \
    g++ \
    gfortran \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Download and build MEQ
ARG GITLAB_TOKEN
WORKDIR /app

# Clone via HTTPS using the token.
RUN git clone https://oauth2:${GITLAB_TOKEN}@gitlab.epfl.ch/spc/public/meq/meq.git meq && \
    cd meq && \
    git config --global url."https://".insteadOf git:// && \
    git submodule update --init --recursive && \
    # Paths in Debian are typically the same as Ubuntu for multi-arch
    make tbx USE_OCTAVE=1 USE_OPENMP=no BLASLIB=CBLAS \
    LAPACKPATH=/usr/lib/x86_64-linux-gnu/lapack \
    CBLASPATH=/usr/lib/x86_64-linux-gnu/blas \
    WERROR=no

# 3. Set environment variables
ENV MAT_ROOT=/app
ENV VIRTUAL_ENV=/app/meqvenv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Headless configuration for JAX and Matplotlib
ENV MPLBACKEND=Agg
ENV JAX_PLATFORMS=cpu
# Prevent threading issues that cause hangs. Only an issue with OpenBLAS.
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# 4. Install MEQ-JAX
COPY . /app/meq_jax
RUN cd /app/meq_jax && pip install -v .

# 5. Verify installation
RUN python /app/meq_jax/examples/meq_jax_example.py

WORKDIR /app/meq_jax
CMD ["python", "examples/meq_jax_example.py"]
