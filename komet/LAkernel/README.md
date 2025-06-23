# LAkernel Python Wrapper

This module provides a Python interface to the original LAkernel C implementation (v0.2), which computes local alignment string kernels.

## Usage

```python
from komet.LAkernel import LAkernel_eval
score = LAkernel_eval("AAA", "AA")
print(score)
```

Ensure that the compiled `LAkernel.so` is located in the same directory, or provide the correct path.

## Build Instructions

### Linux

```bash
gcc -lm -O2 -Wall -Wshadow -fPIC -D__unix -c -o seqio.o seqio.c
gcc -lm -O2 -Wall -Wshadow -fPIC -D__unix -c -o LAkernel.o LAkernel.c
gcc seqio.o LAkernel.o -shared -o LAkernel.so
```

### macOS (Intel or Apple Silicon)

Use the provided script:

```bash
chmod +x build_macOS.sh
./build_macOS.sh
```

This script auto-detects whether your Python is running under Intel (`x86_64`) or Apple Silicon (`arm64`) and compiles the `.so` accordingly using `clang`.

### Windows

On Windows, you can either:

- Use **WSL** (Windows Subsystem for Linux) and follow the Linux instructions above, or
- Compile a dynamic `.dll` library using `MinGW` or `MSYS2`:

```bash
gcc -O2 -Wall -Wshadow -D__windows -c -o seqio.o seqio.c
gcc -O2 -Wall -Wshadow -D__windows -c -o LAkernel.o LAkernel.c
gcc -shared -o LAkernel.dll seqio.o LAkernel.o
```

And in Python:

```python
ctypes.CDLL("LAkernel.dll")
```

## License

GPLv2 – See original LAkernel license.
