SCTL_INCDIR = ./SCTL/include/
ENABLE_CUDA ?= 1
ENABLE_MPI ?= 0

CXX = g++ # requires g++-8 or newer / icpc (with gcc compatibility 7.5 or newer) / clang++ with llvm-10 or newer
CXXFLAGS = -std=c++17 -fopenmp -Wall -Wfloat-conversion # need C++11 and OpenMP

NVCC = nvcc -O3
NVCCFLAGS = -Xcompiler -fopenmp -Xptxas -v -Xptxas -O3
NVCCFLAGS += -gencode arch=compute_70,code=sm_70
NVCCFLAGS += -gencode arch=compute_80,code=sm_80
NVCCFLAGS += -gencode arch=compute_90,code=sm_90
CUDA_LIBS += -L$(CUDA_ROOT)/lib64/ -lcudart

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -O0 -fsanitize=address,leak,undefined,pointer-compare,pointer-subtract,float-divide-by-zero,float-cast-overflow -fno-sanitize-recover=all -fstack-protector # debug build
	CXXFLAGS += -DSCTL_MEMDEBUG # Enable memory checks
else
	CXXFLAGS += -O3 -march=native -DNDEBUG # release build
endif

OS = $(shell uname -s)
ifeq "$(OS)" "Darwin"
	CXXFLAGS += -g -rdynamic -Wl,-no_pie # for stack trace (on Mac)
else
	CXXFLAGS += -gdwarf-4 -g -rdynamic # for stack trace
endif

CXXFLAGS += -DSCTL_PROFILE=5 -DSCTL_VERBOSE # Enable profiling
ifeq ($(ENABLE_MPI), 1)
	CXXFLAGS += -DSCTL_HAVE_MPI #use MPI
endif


RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./include
TESTDIR = ./test

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(filter %.o, $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o))

ifeq ($(ENABLE_CUDA), 1)
	SOURCES += $(wildcard $(SRCDIR)/*.cu)
	OBJECTS += $(filter %.o, $(SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o))
	LDLIBS += $(CUDA_LIBS)
	CXXFLAGS += -DHAVE_CUDA
endif

TARGET_BIN = \
       $(BINDIR)/fmm-near

all : $(TARGET_BIN)

$(BINDIR)/%: $(OBJDIR)/%.o $(OBJECTS)
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDLIBS) $^ -o $@
ifeq "$(OS)" "Darwin"
	/usr/bin/dsymutil $@ -o $@.dSYM
endif

$(OBJDIR)/%.o: $(TESTDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -I$(SCTL_INCDIR) -c $^ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -I$(SCTL_INCDIR) -c $^ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	-@$(MKDIRS) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) -c $^ -o $@

.PHONY: all check clean

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~ */*/*~
