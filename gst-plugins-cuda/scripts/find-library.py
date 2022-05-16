#!/usr/bin/env python3

import pathlib
import sys

assert(len(sys.argv) >= 3)

lib_name = sys.argv[1]
lib_search_paths = sys.argv[2:]

for lib_search_path in lib_search_paths:
    found_libs = sorted(list(pathlib.Path(lib_search_path).glob(f"lib{lib_name}.so*")))
    if len(found_libs) > 0:
        print(found_libs[0])
        sys.exit(0)
sys.exit(1)
