.PHONY:	all clean 

CXX = g++ -std=c++11 
DEBUG = -g
STATIC = -static
ARCH = -march=corei7
OPT = -O3 
OPENMP = -fopenmp
CFLAGS += $(DEBUG) $(OPT) $(STATIC) $(OPENMP) -Wall -I..
CXXFLAGS += $(DEBUG) $(OPT) $(STATIC) $(OPENMP) -Wall -I..
LDFLAGS += $(STATIC)
LDLIBS += -lboost_program_options -lboost_log -lboost_log_setup -lboost_timer -lboost_chrono -lboost_thread -lboost_system -lopenblas-sandybridge-openmp -ldl

HEADERS = argos.h array.h blas-wrapper.h neural.h combo.h io.h eval.h
COMMON = blas-wrapper.o argos.o library.o register.o library.o
PROGS = #argos #cifar train predict
SHARED = argos-basic.so

all:	argos 

argos:	main.o $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(PROGS):	%:	%.o $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)


%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp 

clean:
	rm $(PROGS) *.o

