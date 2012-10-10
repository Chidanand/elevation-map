CC=g++

#INCLUDEDIR= -I/usr/include/pcl-1.6/ -I/usr/include/eigen3

SOURCES=compute_camera_rotation.cpp

EXECUTABLE=compute_camera_rotation

CFLAGS := -DNDEBUG -O3 -march=native

# OpenCV
CFLAGS += `pkg-config --cflags --libs opencv`

all: compute_camera_rotation

compute_camera_rotation:
	$(CC) $(SOURCES) -o $(EXECUTABLE) $(CFLAGS)
#	$(CC) $(SOURCES) -o $(EXECUTABLE) $(INCLUDEDIR) $(CFLAGS)

clean:
	rm -rf $(EXECUTABLE)
