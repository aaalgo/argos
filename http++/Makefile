CC = g++ 
CXXFLAGS += -std=c++11 -O3
OBJS = connection.o  connection_manager.o mime_types.o  reply.o  request_handler.o  static_url_handler.o request_parser.o  server.o	
LIB = libhttp++.a

LDLIBS += $(LIB) -lboost_regex -lboost_system -lpthread

all:	$(LIB)	main 

$(LIB):	$(OBJS)
	ar r $@ $^

clean:
	rm $(OBJS) $(LIB)
