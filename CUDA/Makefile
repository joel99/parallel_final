EXECUTABLE := out
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
CU_FILES   := cuda_solver.cu cuda_utils.cu
CU_DEPS    :=
CC_FILES   := main.cpp utils.cpp word.cpp mathutils.cpp
LOGS	   := logs

all: $(EXECUTABLE)

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/utils.o $(OBJDIR)/word.o $(OBJDIR)/mathutils.o \
	 $(OBJDIR)/cuda_solver.o $(OBJDIR)/cuda_utils.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) *.ppm


export: $(EXFILES)
	cp -p $(EXFILES) $(STARTER)


$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)


$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
