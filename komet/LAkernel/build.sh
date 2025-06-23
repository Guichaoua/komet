#!/bin/bash

# Build script for LAkernel shared object

echo "Compiling seqio.c..."
gcc -lm -O2 -Wall -Wshadow -fPIC -D__unix -c -o seqio.o seqio.c

echo "Compiling LAkernel.c..."
gcc -lm -O2 -Wall -Wshadow -fPIC -D__unix -c -o LAkernel.o LAkernel.c

echo "Linking to create LAkernel.so..."
gcc seqio.o LAkernel.o -shared -o LAkernel.so

echo "Build complete: LAkernel.so"
