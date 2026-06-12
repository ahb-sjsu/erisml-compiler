# ===============================================================
# Makefile for erisml-compiler silicon target
# Target: Xilinx Alveo U55C
# Toolchain: Vitis HLS / v++ (Vivado 2024.x recommended)
# NRP Coder template: "U55C FPGA Vitis Workflow"
# ===============================================================

VPP        := v++
PLATFORM   := xilinx_u55c_gen3x16_xdma_3_202210_1
TARGET     := hw
KERNEL     := erisml_evaluate
SOURCES    := erisml_fsm.cpp erisml_em_dag.cpp erisml_top.cpp
OBJ_DIR    := build_$(TARGET)

XO         := $(OBJ_DIR)/erisml.xo
XCLBIN     := $(OBJ_DIR)/erisml.xclbin

.PHONY: all hls hw_emu hw clean

all: $(XCLBIN)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(XO): $(SOURCES) | $(OBJ_DIR)
	$(VPP) -c -t $(TARGET) --platform $(PLATFORM) --kernel $(KERNEL) \
	       -o $@ $(SOURCES)

$(XCLBIN): $(XO)
	$(VPP) -l -t $(TARGET) --platform $(PLATFORM) \
	       -o $@ $<

hw_emu:
	$(MAKE) all TARGET=hw_emu

clean:
	rm -rf build_hw build_hw_emu build_sw_emu *.log *.jou
