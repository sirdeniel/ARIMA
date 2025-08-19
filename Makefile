CXX       = g++
CXXFLAGS  = -Wall -g -Iinclude -Ilib
LDFLAGS   = 
SRC_DIR   = src
OBJ_DIR   = build
BIN_DIR   = bin
TARGET    = $(BIN_DIR)/arima

SRCS := $(shell find $(SRC_DIR) -type f -name '*.cpp')
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)
