CC = mingw32-g++
CFLAGS = -mavx -g -O3 -DNDEBUG -fopenmp -fopenacc
TARGET = nn
all: $(TARGET)
	
$(TARGET): src/*.cpp
	$(CC) -I include/ $(CFLAGS) -o $(TARGET) src/*.cpp
