# CMakeList.txt : CMake project for TestIntrin, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
#add_executable (TestIntrin "TestIntrin.cpp" "TestIntrin.h")

# TODO: Add tests and install targets if needed.
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable (main "main.cpp" "main.h")
target_link_libraries( main ${OpenCV_LIBS} )