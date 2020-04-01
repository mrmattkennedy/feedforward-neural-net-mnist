CC = nvcc
CFLAGS = --x cu -Xptxas -O3,-v
TARGET = gpu
all: $(TARGET)

$(TARGET): src/*.cpp
	$(CC) -I include/ $(CFLAGS) -o $(TARGET) src/*.cpp
	mv $(TARGET).exp build/$(TARGET).exp
	mv $(TARGET).lib build/$(TARGET).lib

