# SPDX-License-Identifier: MIT
# Bare-metal firmware build for the motorloop reference SoC (tier2-adoption §2).
# Modelled on the LiteX bare-metal demo: it pulls the SoC's generated build
# variables (toolchain, flags, include paths) so it links against the right
# libbase/libc for the CPU the SoC was built with. Build the SoC first:
#
#   python3 soc/motorloop_soc.py --build          # -> build/radiona_ulx3s/
#   make -C soc/firmware BUILD_DIR=$(pwd)/build/radiona_ulx3s
#
# Produces firmware.bin to load into the SoC's RAM (or pack into the gateware).

BUILD_DIR ?= ../../build/radiona_ulx3s
include $(BUILD_DIR)/software/include/generated/variables.mak
include $(SOC_DIRECTORY)/software/common.mak

OBJECTS = main.o

all: firmware.bin

firmware.bin: firmware.elf
	$(OBJCOPY) -O binary $< $@
	chmod -x $@

firmware.elf: $(OBJECTS)
	$(CC) $(LDFLAGS) -T linker.ld -N -o $@ \
	  $(BUILD_DIR)/software/libbase/crt0.o $(OBJECTS) \
	  -L$(BUILD_DIR)/software/libbase -L$(BUILD_DIR)/software/libc \
	  -lbase-nofloat -lc
	chmod -x $@

main.o: main.c
	$(CC) -c $(CFLAGS) -o $@ $<

clean:
	$(RM) $(OBJECTS) firmware.elf firmware.bin

.PHONY: all clean
