.PHONY:	all clean 

CXX = g++ -std=c++11 
DEBUG = -g
STATIC = -static
ARCH = -march=corei7
OPT = -O3 
OPENMP = #-fopenmp
CFLAGS += $(DEBUG) $(OPT) $(STATIC) $(OPENMP) -Wall -I.. -I/opt/libjpeg-turbo/include -DMEM_SRCDST_SUPPORTED -Ihttp++
CXXFLAGS += $(DEBUG) $(OPT) $(STATIC) $(OPENMP) -Wall -I.. -I/opt/libjpeg-turbo/include -DMEM_SRCDST_SUPPORTED -Ihttp++
LDFLAGS += $(STATIC) -L/opt/libjpeg-turbo/lib -Lhttp++
LDLIBS += -lhttp++ -lopencv_imgproc -lopencv_core -lboost_regex -lboost_program_options -lboost_log -lboost_timer -lboost_chrono -lboost_thread -lboost_system -lturbojpeg -lopenblas-sandybridge -ldl -lz -lpthread -lrt
#LDLIBS += -lboost_program_options -lboost_log -lboost_timer -lboost_chrono -lboost_thread -lboost_system -lopenblas-sandybridge-openmp -ldl


HEADERS = argos.h array.h blas-wrapper.h
NODE_HEADERS = node-core.h node-utils.h node-combo.h node-image.h node-dream.h
COMMON = blas-wrapper.o argos.o library.o library.o 
PROGS = #argos #cifar train predict
SHARED = argos-basic.so

all:	argos 

argos:	main.o $(COMMON) register.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(PROGS):	%:	%.o $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

register.o:	register.cpp $(HEADERS) $(NODE_HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp 


%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp 

clean:
	rm $(PROGS) *.o

