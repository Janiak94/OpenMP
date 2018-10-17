CC=gcc
STD=-std=c11
OFLAGS=-O3 -ffast-math
LIBFLAGS=-lm -lgomp

.PHONY: all clean

all: cell_distance debug_cell

cell_distance:cell_distance.c
	$(CC) -fopenmp $(OFLAGS) $(STD) -o $@ $< $(LIBFLAGS)

debug_cell: cell_distance.c
	$(CC) -openmp $(STD) -g -o $@ $< $(LIBFLAGS)

clean:
	rm cell_distance
	rm debug_cell
