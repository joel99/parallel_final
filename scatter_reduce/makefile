# Name of the executable you want to produce
TARGET = sred

# Source files (adjust if you have more than one)
SRC = sred.cpp

# Default compiler
CXX = g++

# Compiler flags
CFLAGS = -std=c++17 -Wall -O3 -fopenmp

# Linking step - no additional libraries needed in your case
LDFLAGS = 

# The main rule to build your target with the default compiler
$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Rule to build with g++-13
$(TARGET)-special:
	$(MAKE) CXX=g++-13

# A rule for cleaning object files and the executable (optional)
clean:
	rm -f *.o $(TARGET) $(TARGET)-special

