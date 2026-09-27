# Dockerfile for MEQ-JAX
# To build and run:
# 1. From the repository root, build: docker build -f meq_jax/Dockerfile -t meq_jax .
# 2. Run: docker run meq_jax

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
WORKDIR /app

# MEQ v6.10.0 removed L.ind.ira (upstream 32095be), which MEQ-JAX still reads.
# Override with: docker build --build-arg MEQ_VERSION=<tag-or-commit> ...
ARG MEQ_VERSION=crpptbx-release-v6.9.0

# Clone the public MEQ repository via HTTPS.
RUN git clone --branch ${MEQ_VERSION} https://gitlab.epfl.ch/spc/public/meq/meq.git meq && \
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
RUN cd /app/meq_jax && pip install --no-cache-dir --force-reinstall -v .

# 5. Verify installation
RUN python -c "import meq_jax; import meqpy"

WORKDIR /app/meq_jax
CMD ["python", "meq_jax/examples/meq_jax_example.py"]
