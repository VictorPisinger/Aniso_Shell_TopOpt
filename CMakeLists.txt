cmake_minimum_required(VERSION 3.6.0)
project(shopt) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-fsanitize=address -Wall -Wextra -Wno-deprecated-copy -fopenmp ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -flto -fopenmp ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-fno-omit-frame-pointer -march=native -flto -fopenmp ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

# set include paths
include_directories( ${shopt_SOURCE_DIR} )

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set libararies to load
find_package(Catch2 REQUIRED)
list(APPEND LIBS "-lyaml-cpp")
list(APPEND LIBS "-lcholmod")
list(APPEND LIBS "-lexodus")
list(APPEND LIBS "-lbenchmark")

# define object macro
macro (add_object_by_name name)
if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${name}.cpp)
add_library(${name} OBJECT ${name}.cpp)
list(APPEND object_list ${name})
else()
MESSAGE(STATUS "\nLooking for file: " ${CMAKE_CURRENT_LIST_DIR}/${name} )
MESSAGE(STATUS "Error: cannot find file: " ${name}.cpp)
endif()
endmacro (add_object_by_name)

# define executable macro
macro (add_executable_by_name name)
if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${name}.cpp)
ADD_EXECUTABLE(${name} ${name}.cpp)
TARGET_LINK_LIBRARIES(${name} ${object_list} ${LIBS})
install(TARGETS ${name} DESTINATION ${Shell_SOURCE_DIR}/bin)
else()
MESSAGE(STATUS "\nLooking for file: " ${CMAKE_CURRENT_LIST_DIR}/${name} )
MESSAGE(STATUS "Error: cannot find file: " ${name}.cpp)
endif()
endmacro (add_executable_by_name)

# Add test
macro (add_test_by_name name)
if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${name}.cpp)
ADD_EXECUTABLE(${name} ${name}.cpp)
TARGET_LINK_LIBRARIES(${name} ${object_list} ${LIBS} Catch2::Catch2WithMain)
ADD_TEST(NAME ${name} COMMAND ${name})
else()
MESSAGE(STATUS "\nLooking for file: " ${CMAKE_CURRENT_LIST_DIR}/${name} )
MESSAGE(STATUS "Error: cannot find file: " ${name}.cpp)
endif()
endmacro (add_test_by_name)

add_object_by_name(elementIntegrator)
add_object_by_name(finiteElementAnalysis)
add_object_by_name(manifold)
add_object_by_name(mesh)
add_object_by_name(options)
add_object_by_name(exodusWriter) 
add_object_by_name(volumeConstraint) 
add_object_by_name(MMAseq) 
add_object_by_name(pdeFilter) 
add_object_by_name(thicknessFilter) 
add_object_by_name(qualityConstraint) 

add_test_by_name(testElementIntegrator)
add_test_by_name(testFiniteElementAnalysis)
add_test_by_name(testOptimization)
add_test_by_name(testFiniteDifference)

add_executable_by_name(main)
add_executable_by_name(benchmark)