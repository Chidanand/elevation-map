CC=g++

SOURCES=undistort_image.cpp

EXECUTABLE=undistort_image

CFLAGS := -Wall -DNDEBUG -O3 -march=native

# OpenCV
CFLAGS += `pkg-config --cflags --libs opencv`

all: undistort_image

undistort_image:
	$(CC) $(SOURCES) -o $(EXECUTABLE) $(CFLAGS)

clean:
	rm -rf $(EXECUTABLE)
