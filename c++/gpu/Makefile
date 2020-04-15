CC = nvcc
CFLAGS = -I include/ --x cu -Xptxas -O3,-v -l cublas
TARGET = gpu
all: $(TARGET)

$(TARGET): src/*.cpp
	$(CC) $(CFLAGS) -o $(TARGET) src/*.cpp || (echo "Compile failed"; exit 1)
	mv $(TARGET).exp build/$(TARGET).exp
	mv $(TARGET).lib build/$(TARGET).lib
	./gpu
