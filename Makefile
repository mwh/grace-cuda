
MINIGRACE_DIR=../minigrace
CUDA_DIR=/opt/cuda
CUDA_LIB_DIR=$(CUDA_DIR)/lib64
CUDA_INCLUDE_DIR=$(CUDA_DIR)/include
INSTALL_DIR=$(MINIGRACE_DIR)

SAMPLES = map generalcuda matrix
SAMPLES_EXECUTABLES = $(foreach s,$(SAMPLES),samples/$(s))

all: cudap.gso cuda.gso

cudap.gso: cudap.grace
	$(MINIGRACE_DIR)/minigrace --dynamic-module cudap.grace

cuda.gso: cuda.c
	gcc -I$(CUDA_INCLUDE_DIR) -I$(MINIGRACE_DIR) -g -std=c99 -o cuda.gso -L$(CUDA_LIB_DIR) -lcuda -shared -fPIC cuda.c

samples: all $(SAMPLES_EXECUTABLES)

samples/%: samples/%.grace cudap.gso cuda.gso
	$(MINIGRACE_DIR)/minigrace --make -XPlugin=cudap $<

install: cudap.gso cuda.gso
	cp $^ $(INSTALL_DIR)

clean:
	rm -f cudap.gso cuda.gso
	rm -f samples/*.c samples/*.gct samples/*.gcn
	rm -f _cuda/*
	rm -f $(SAMPLES_EXECUTABLES)

.PHONY: install clean samples
