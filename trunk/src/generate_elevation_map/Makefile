CC=g++

#INCLUDEDIR= -I/usr/include/pcl-1.6/ -I/usr/include/eigen3

SOURCES=generate_elevation_map.cpp

EXECUTABLE=generate_elevation_map

CFLAGS := -DNDEBUG -O3 -march=native

# OpenCV
CFLAGS += `pkg-config --cflags --libs opencv`

#DEBUG
CFLAGS += -ggdb

all: generate_elevation_map

generate_elevation_map:
	$(CC) $(SOURCES) -o $(EXECUTABLE) $(CFLAGS)
#	$(CC) $(SOURCES) -o $(EXECUTABLE) $(INCLUDEDIR) $(CFLAGS)

clean:
	rm -rf $(EXECUTABLE)
