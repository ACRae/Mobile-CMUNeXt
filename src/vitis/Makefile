# Makefile for HLS C++ project
CXX = g++
CXXFLAGS = -std=c++17 -g -O0
INCLUDES = -I./include -I./include/etc -I./include/utils -I./hls
DEFINES = -D__VITIS_HLS__

# Directories
SRCDIR = hls
BUILDDIR = build
INCDIR = include

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

# Testbench executables
TESTBENCHES = tb_layerC3D tb_layerDW tb_layerPW

# Main target
.PHONY: all clean testbenches

all: $(BUILDDIR) $(TESTBENCHES)

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Build testbench executables
tb_layerC3D: $(BUILDDIR)/tb_layerC3D
tb_layerDW: $(BUILDDIR)/tb_layerDW
tb_layerPW: $(BUILDDIR)/tb_layerPW

$(BUILDDIR)/tb_layerC3D: $(BUILDDIR)/tb_layerC3D.o $(BUILDDIR)/layerC3D.o $(BUILDDIR)/utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILDDIR)/tb_layerDW: $(BUILDDIR)/tb_layerDW.o $(BUILDDIR)/layerDW.o $(BUILDDIR)/utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILDDIR)/tb_layerPW: $(BUILDDIR)/tb_layerPW.o $(BUILDDIR)/layerPW.o $(BUILDDIR)/utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compile object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -c $< -o $@

# Run testbenches
run-tb-c3d: $(BUILDDIR)/tb_layerC3D
	./$(BUILDDIR)/tb_layerC3D

run-tb-dw: $(BUILDDIR)/tb_layerDW
	./$(BUILDDIR)/tb_layerDW

run-tb-pw: $(BUILDDIR)/tb_layerPW
	./$(BUILDDIR)/tb_layerPW

# Clean build artifacts
clean:
	rm -rf $(BUILDDIR)

# Debug information
debug:
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Include paths: $(INCLUDES)"
