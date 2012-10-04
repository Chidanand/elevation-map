CC=g++

DIR_LIBRARY=../libraries/

DIR_LIBELAS=$(DIR_LIBRARY)libelas/src/

SOURCES=generate_disparity_and_Q.cpp $(DIR_LIBELAS)elas.cpp $(DIR_LIBELAS)descriptor.cpp $(DIR_LIBELAS)filter.cpp $(DIR_LIBELAS)matrix.cpp $(DIR_LIBELAS)triangle.cpp

EXECUTABLE=generate_disparity_and_Q

CFLAGS := -DNDEBUG -O3 -march=native

# OpenCV
CFLAGS += `pkg-config --cflags --libs opencv`

all: generate_disparity_and_Q

generate_disparity_and_Q:
	$(CC) $(SOURCES) -o $(EXECUTABLE) $(CFLAGS)

clean:
	rm -rf $(EXECUTABLE)
