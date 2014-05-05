.PHONY:	all clean 

CXX = g++ -std=c++11 
DEBUG = -g
STATIC = -static
ARCH = -march=corei7
OPT = -O3 
OPENMP = -fopenmp
CFLAGS += -g $(DEBUG) $(OPT) $(STATIC) $(OPENMP) -Wall -I..
CXXFLAGS += -g $(DEBUG) $(OPT) $(STATIC) $(OPENMP) -Wall -I..
LDFLAGS += -g $(STATIC) 
LDLIBS += -lboost_program_options -lboost_timer -lboost_chrono -lboost_system -lopenblas-sandybridge-openmp

HEADERS = argos.h array.h blas-wrapper.h neural.h combo.h io.h
COMMON = blas-wrapper.o argos.o register.o
PROGS = run #cifar train predict

all:	$(PROGS)

$(PROGS):	%:	%.o $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp $(COMMON) $(LDLIBS)

%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp 

clean:
	rm $(PROGS) *.o

