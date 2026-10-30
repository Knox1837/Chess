# Builds the C++ chess engine.   From the project root (the folder with main.py):
#
#     cmake -B build
#     cmake --build build
#
# The executable is written to engine/bin/chess_engine(.exe), where the Python code looks for it.
#
# Options (add to the first command, e.g.  cmake -B build -DCMAKE_BUILD_TYPE=Debug):
#     -DCMAKE_BUILD_TYPE=Release|Debug|RelWithDebInfo|MinSizeRel    (default: Release)
#     -DUSE_NATIVE=OFF       portable binary instead of one tuned for this CPU      (default: ON)
#     -DSTATIC_LINK=OFF      don't link the C++ runtime statically (MinGW only)     (default: ON)
#     -DWARNINGS=ON          extra compiler warnings                                (default: OFF)
#     -DENGINE_OUTPUT_DIR=D  put the executable somewhere else
#     -G "MinGW Makefiles" / -G Ninja / -G "Visual Studio 17 2022"   pick a toolchain
# Visual Studio (multi-config) generators choose the config at build time:
#     cmake --build build --config Release
cmake_minimum_required(VERSION 3.13)
project(chess_engine LANGUAGES CXX)

option(USE_NATIVE  "Tune for this CPU (-march=native); OFF gives a portable binary" ON)
option(STATIC_LINK "Link the C++ runtime statically (MinGW: no extra DLLs needed)"  ON)
option(WARNINGS    "Enable extra compiler warnings" OFF)
set(ENGINE_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/engine/bin" CACHE PATH "Where chess_engine(.exe) is written")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # build/compile_commands.json for VS Code IntelliSense

get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT IS_MULTI_CONFIG)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Release, Debug, RelWithDebInfo or MinSizeRel" FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug RelWithDebInfo MinSizeRel)
endif()

find_package(Threads REQUIRED)
add_executable(chess_engine engine/cpp/engine.cpp)
target_include_directories(chess_engine PRIVATE engine/cpp)
target_link_libraries(chess_engine PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(chess_engine PRIVATE /EHsc)
  if(WARNINGS)
    target_compile_options(chess_engine PRIVATE /W4)
  endif()
else()
  if(USE_NATIVE)
    target_compile_options(chess_engine PRIVATE -march=native)
  endif()
  if(WARNINGS)
    target_compile_options(chess_engine PRIVATE -Wall -Wextra)
  endif()
  if(MINGW AND STATIC_LINK)
    target_link_options(chess_engine PRIVATE -static)
  endif()
endif()

# One output folder for every configuration
set_target_properties(chess_engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${ENGINE_OUTPUT_DIR}")
foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(chess_engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${cfg} "${ENGINE_OUTPUT_DIR}")
endforeach()

add_custom_command(TARGET chess_engine POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Built: $<TARGET_FILE:chess_engine>")