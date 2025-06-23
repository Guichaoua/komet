#!/bin/bash

PYTHON_ARCH=$(python3 -c "import platform; print(platform.machine())")

echo "Python architecture: $PYTHON_ARCH"

if [ "$PYTHON_ARCH" = "x86_64" ]; then
    echo "Forcing compilation in x86_64 mode (Intel / Rosetta)..."
    arch -x86_64 clang -O2 -Wall -Wshadow -fPIC -D__unix -c -o seqio.o seqio.c
    arch -x86_64 clang -O2 -Wall -Wshadow -fPIC -D__unix -c -o LAkernel.o LAkernel.c
    arch -x86_64 clang seqio.o LAkernel.o -shared -o LAkernel.so
else
    echo "Compiling for native architecture ($PYTHON_ARCH)..."
    clang -O2 -Wall -Wshadow -fPIC -D__unix -c -o seqio.o seqio.c
    clang -O2 -Wall -Wshadow -fPIC -D__unix -c -o LAkernel.o LAkernel.c
    clang seqio.o LAkernel.o -shared -o LAkernel.so
fi

echo "Build complete: LAkernel.so"
