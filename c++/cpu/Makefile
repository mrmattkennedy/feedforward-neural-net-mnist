CC = x86_64-w64-mingw32-g++
CFLAGS = -mavx -g -O3 -DNDEBUG -fopenmp -fopenacc
TARGET = cpu
all: $(TARGET)
	
$(TARGET): src/*.cpp
	$(CC) -I include/ $(CFLAGS) -o $(TARGET) src/*.cpp || (echo "Compile failed"; exit 1)
	./cpu
