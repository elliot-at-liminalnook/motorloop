cmake_minimum_required(VERSION 3.22)
project(bldcsim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)

find_package(verilator HINTS $ENV{VERILATOR_ROOT} /usr/share/verilator REQUIRED)

set(RTL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../rtl)

pybind11_add_module(bldcsim
  src/bindings.cpp
  src/one_phase_plant.cpp
  src/three_phase_plant.cpp
  src/three_phase_scenario.cpp
  src/drv8301.cpp
  src/mcp3208.cpp
  src/as5600.cpp
  src/bench.cpp
)
target_include_directories(bldcsim PRIVATE src)
set_target_properties(bldcsim PROPERTIES POSITION_INDEPENDENT_CODE ON)

verilate(bldcsim
  SOURCES
    ${RTL_DIR}/controller_top.v
    ${RTL_DIR}/pwm_generator.v
    ${RTL_DIR}/commutation.v
    ${RTL_DIR}/spi_drv_master.v
    ${RTL_DIR}/adc_spi_master.v
    ${RTL_DIR}/divider32.v
    ${RTL_DIR}/as5600_pwm_capture.v
    ${RTL_DIR}/speed_meter.v
    ${RTL_DIR}/speed_pi.v
    ${RTL_DIR}/open_loop_ramp.v
    ${RTL_DIR}/drv_manager.v
    ${RTL_DIR}/adc_sequencer.v
  TOP_MODULE controller_top
  PREFIX Vcontroller_top
  TRACE
  INCLUDE_DIRS ${RTL_DIR} ${RTL_DIR}/gen
  VERILATOR_ARGS --x-assign fast --x-initial fast
)
