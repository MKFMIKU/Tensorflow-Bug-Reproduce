TF_CFLAGS := `python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'`
TF_LFLAGS := `python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'`
CUDA_HOME := /usr/local/cuda

SRC_DIR = ops
BUILD_DIR = build
LIB_DIR = lib

CC = g++ -std=c++11
NVCC = nvcc -std c++11
CFLAGS = -fPIC $(TF_CFLAGS) $(TF_LFLAGS)  -O2 -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart
NVFLAGS = -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG -expt-relaxed-constexpr -Wno-deprecated-gpu-targets -ftz=true

SRC = bilateral_slice.cc
CUDA_SRC = bilateral_slice.cu.cc
CUDA_OBJ = $(addprefix $(BUILD_DIR)/,$(CUDA_SRC:.cc=.o))
SRCS = $(addprefix $(SRC_DIR)/, $(SRC))

all: $(LIB_DIR)/ops.so

# Main library
$(LIB_DIR)/hdrnet_ops.so: $(CUDA_OBJ) $(LIB_DIR) $(SRCS)
	$(CC) -shared -o $@ $(SRCS) $(CUDA_OBJ) $(CFLAGS) $(LDFLAGS)

# Cuda kernels
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc $(BUILD_DIR)
	$(NVCC) -c $< -o $@ $(TF_CFLAGS) -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 $(TF_LFLAGS) $(NVFLAGS)

$(BUILD_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)