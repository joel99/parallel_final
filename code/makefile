CXX=g++
CXXFLAGS=-std=c++11 -Wall
SRC=wordle.cpp
OBJ=$(SRC:.cpp=.o)
EXEC=wordle

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(OBJ) $(EXEC)

.PHONY: all clean
