CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3

# List of source files
SOURCES = main.cpp mathutils.cpp word.cpp utils.cpp
# Object files can be deduced from the source files list
OBJECTS = $(SOURCES:.cpp=.o)
# The name of the final executable
EXECUTABLE = wordle

# The first target is the default target, so this will be the one made when just typing "make"
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# A pattern rule that uses automatic variables:
# - $<: the name of the prerequisite of the rule (a .cpp file in this case)
# - $@: the name of the target of the rule (the corresponding .o file)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Phony target to clean the build files
.PHONY: clean
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

# Dependencies - you would fill these in with the specific header dependencies for each source
mathutils.o: mathutils.h
word.o: word.h
utils.o: utils.h
