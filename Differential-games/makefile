EFLAGS = -I /usr/include/eigen3
CFLAGS = -ffast-math

all: final

final: sine.o constructingmatrixes.o
	g++ $(CFLAGS)  sine.o constructingmatrixes.o -o final

constructingmatrixes.o: ConstructingMat.cpp
	@echo " Constructing Matrices ..."
	g++ -O3 $(EFLAGS) $(CFLAGS) -c ConstructingMat.cpp 

sine.o:sine.cpp
	@echo "Constructing the sine tragectory..."
	g++ $(CFLAGS)  -c sine.cpp -lsfml-graphics -lsfml-window -lsfml-system

clean:
	@echo"Cleaning..."
	rm sine.o constructingmatrixes.o final 