CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra
OBJECTS = trace_formatter.o trace_capture.o

all: format-trace-capture

format-trace-capture: format_trace_capture.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp trace_formatter.h trace_capture.h ../trace_record.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) format-trace-capture
