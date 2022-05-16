#!/usr/bin/env python3

import pathlib
import sys

assert(len(sys.argv) >= 3)

lib_name = sys.argv[1]
lib_search_paths = sys.argv[2:]

lib_paths = []

for lib_search_path in lib_search_paths:
    lib_paths += list(pathlib.Path(lib_search_path).glob(f"lib{lib_name}.so*"))

if not lib_paths:
    sys.exit(1)

print(":".join([str(lib_path) for lib_path in lib_paths]))
sys.exit(0)
