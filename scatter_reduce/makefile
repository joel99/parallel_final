# Name of the executable you want to produce
TARGET = sred
TARGET_2D = sred-2d

# Source files (adjust if you have more than one)
SRC = sred.cpp
SRC_2D = sred_2d.cpp

# Compiler
CXX = g++
CXX_13 = g++-13

# Compiler flags
CFLAGS = -std=c++17 -Wall -O3 -fopenmp

# Linking step - no additional libraries needed in your case
LDFLAGS = 

# The main rule to build your target
$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Rule to build with g++-13 for the main target
$(TARGET)-special:
	$(CXX_13) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Rule for sred_2d
$(TARGET_2D): $(SRC_2D)
	$(CXX) $(CFLAGS) $(SRC_2D) -o $(TARGET_2D) $(LDFLAGS)

# Rule to build sred_2d with g++-13
$(TARGET_2D)-special:
	$(CXX_13) $(CFLAGS) $(SRC_2D) -o $(TARGET_2D) $(LDFLAGS)

# A rule for cleaning object files and the executable (optional)
clean:
	rm -f *.o $(TARGET) $(TARGET)-special $(TARGET_2D) $(TARGET_2D)-special
