LINKLIB=  -L/usr/local/lib/ -fopenmp -lfftw3f_omp -lfftw3f -lpthread -lm

#LINKLIB= -lm -lc -L/usr/local/lib/ -lfftw3

INCLUDE=-I/usr/local/include/

CFLAGS=-g
CC=gcc


bispec: bispec.o  
	$(CC) $(CFLAGS) -o bispec  bispec.o $(LINKLIB)
	rm -rf *.o
	rm -rf *~


bispec.o:	bispec.c
	$(CC) -c $(CFLAGS) $(INCLUDE) bispec.c


clean:
	rm -rf *.o
	rm -rf *~










