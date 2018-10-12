CC=gcc
STD=-std=c11
OFLAGS=-O2 -ffast-math
LIBFLAGS=-lm

.PHONY: all

all: cell_distance cell_debug

cell_distance:cell_distance.c
	$(CC) $(OFLAGS) $(STD) -o $@ $< $(LIBFLAGS)

cell_debug:cell_distance.c
	$(CC) $(STD) -g -o $@ $< $(LIBFLAGS)
