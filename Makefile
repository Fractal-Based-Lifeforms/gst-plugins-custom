# This Makefile is intended to be used in conjunction with Dockerfile.jammy
# and is used by .github/workflows/docker.yml

build:
	cd gst-plugins-cuda && CC=gcc-10 CXX=g++-10 meson build -Dtests=enabled
	cd gst-plugins-cuda && ninja -C build
	cd gst-plugins-cuda && ninja install -C build

test:
	cd gst-plugins-cuda && ninja test -C build
