cmake_minimum_required( VERSION 3.20)


add_subdirectory(sympiler)

set(CMAKE_CXX_STANDARD 11)

if(NOT TARGET MKL::MKL)
    set(MKL_INTERFACE "lp64")
    find_package(MKL CONFIG REQUIRED)
    add_definitions( "-DMKL")
endif()

project(psc_example )
add_executable(psc_example
        main.cpp
        )
target_compile_options(psc_example PRIVATE
        -march=native
        )
target_include_directories(psc_example PRIVATE
        sympiler/include
        sympiler/aggregation/include
        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
        )
target_link_libraries(psc_example PRIVATE
        aggregation
        $<LINK_ONLY:MKL::MKL>
        )
