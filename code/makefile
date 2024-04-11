CXX=g++
CXXFLAGS=-std=c++11 -Wall
SRC=main.cpp wordle_utils.cpp wordle_solver.cpp
OBJ=$(SRC:.cpp=.o)
EXEC=wordle

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)

.PHONY: all clean
