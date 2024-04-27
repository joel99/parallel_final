# Name of the executable you want to produce
TARGET = sred

# Source files (adjust if you have more than one)
SRC = sred.cpp

# Compiler
CXX = g++-13

# Compiler flags
CFLAGS = -std=c++17 -Wall -O3 -fopenmp

# Linking step - no additional libraries needed in your case
LDFLAGS = 

# The main rule to build your target
$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# A rule for cleaning object files and the executable (optional)
clean:
	rm -f *.o $(TARGET) 