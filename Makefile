# Makefile

# Compiler
CXX = /opt/homebrew/opt/llvm/bin/clang++
# CXX = /opt/homebrew/bin/g++-13

# Compiler flags
# CXXFLAGS = -g -fopenmp -Ofast -framework Accelerate -DACCELERATE_NEW_LAPACK 
CXXFLAGS = -g -fopenmp -Ofast -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
# CXXFLAGS = -g -fopenmp -Ofast

# Target executable name
TARGET = main

# Source files
SRC = *.cpp
HEADERS = *.h

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)