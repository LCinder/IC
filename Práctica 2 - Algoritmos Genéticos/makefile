
CXX = g++
CPPFLAGS = -std=c++11 -O2
BIN = .
OBJ = ./obj
DOC = ./doc
SRC = ./src
INCLUDE = ./include
BINF = ./bin

all: $(BINF)/QAP

# ************ Compilación de módulos ************

$(BINF)/QAP: $(OBJ)/random.o $(OBJ)/QAP.o $(INCLUDE)/QAP.h
	$(CXX) $(CPPFLAGS) $(INCLUDE)/QAP.h $(OBJ)/random.o $(OBJ)/QAP.o -o $(BINF)/QAP

$(OBJ)/random.o: $(SRC)/random.cpp $(INCLUDE)/random.h
	$(CXX) $(CPPFLAGS) -c $(SRC)/random.cpp -o $(OBJ)/random.o -I $(INCLUDE)/random.h

$(OBJ)/QAP.o: $(SRC)/QAP.cpp $(INCLUDE)/QAP.h
	$(CXX) $(CPPFLAGS) -c $(SRC)/QAP.cpp -o  $(OBJ)/QAP.o -I $(INCLUDE)/QAP.h


run:
	./bin/QAP ./data/
