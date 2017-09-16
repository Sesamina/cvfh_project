cmake_minimum_required(VERSION 3.9)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "possible configurations" FORCE)

# In case the user does not setup CMAKE_BUILD_TYPE, assume it's RelWithDebInfo
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "build type default to RelWithDebInfo, set to Release to improve performance" FORCE)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

project(cvfh_project)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

### ---[ Find universal dependencies
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

# ---[ Release/Debug specific flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  add_definitions("-DBOOST_DISABLE_ASSERTS -DEIGEN_NO_DEBUG")
endif()
if(WIN32 AND NOT MINGW)
  if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "_debug")
  endif()
  if(NOT DEFINED CMAKE_RELEASE_POSTFIX)
    set(CMAKE_RELEASE_POSTFIX "_release")
  endif()
  if(NOT DEFINED CMAKE_RELWITHDEBINFO_POSTFIX)
    set(CMAKE_RELWITHDEBINFO_POSTFIX "_release")
  endif()
  if(NOT DEFINED CMAKE_MINSIZEREL_POSTFIX)
    set(CMAKE_MINSIZEREL_POSTFIX "_release")
  endif()
endif()

# Compiler identification
# Define a variable CMAKE_COMPILER_IS_X where X is the compiler short name.
# Note: CMake automatically defines one for GNUCXX, nothing to do in this case.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_CLANG 1)
elseif(__COMPILER_PATHSCALE)
  set(CMAKE_COMPILER_IS_PATHSCALE 1)
elseif(MSVC)
  set(CMAKE_COMPILER_IS_MSVC 1)
endif()

 if(WIN32)
    if(PCL_SHARED_LIBS)
      SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--export-all-symbols -Wl,--enable-auto-import")
      if (MINGW)
        add_definitions("-DBOOST_THREAD_USE_LIB")
        SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-multiple-definition")
      endif()
    else(PCL_SHARED_LIBS)
      add_definitions("-DBOOST_LIB_DIAGNOSTIC -DBOOST_THREAD_USE_LIB")
    endif(PCL_SHARED_LIBS)
  endif()

if(CMAKE_COMPILER_IS_MSVC)
  add_definitions("-DBOOST_ALL_NO_LIB -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -DPCL_ONLY_CORE_POINT_TYPES /bigobj ${SSE_DEFINITIONS}")
  if("${CMAKE_CXX_FLAGS}" STREQUAL " /DWIN32 /D_WINDOWS /W3 /GR /EHsc")	# Check against default flags
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /EHsc /fp:precise /wd4800 /wd4521 /wd4251 /wd4275 /wd4305 /wd4355 ${SSE_FLAGS}")

    # Add extra code generation/link optimizations
    if(CMAKE_MSVC_CODE_LINK_OPTIMIZATION)
      SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL")
      SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG /OPT:REF")
      SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    endif(CMAKE_MSVC_CODE_LINK_OPTIMIZATION)
    # /MANIFEST:NO") # please, don't disable manifest generation, otherwise crash at start for vs2008

    if( MSVC_VERSION GREATER 1500 AND ${CMAKE_VERSION} VERSION_GREATER "2.8.6")
      include(ProcessorCount)
      ProcessorCount(N)
      if(NOT N EQUAL 0)
        SET(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /MP${N} ")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP${N} ")
      endif()
    endif()
  endif()
endif()

if(CMAKE_COMPILER_IS_PATHSCALE)
  if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    SET(CMAKE_CXX_FLAGS "-Wno-uninitialized -zerouv -pthread -mp")
  endif()
  if("${CMAKE_SHARED_LINKER_FLAGS}" STREQUAL "")
    SET(CMAKE_SHARED_LINKER_FLAGS "-mp")
  endif()
endif()

if(CMAKE_COMPILER_IS_CLANG)
  if("${CMAKE_C_FLAGS}" STREQUAL "")
    SET(CMAKE_C_FLAGS "-Qunused-arguments")
  endif()
  if("${CMAKE_CXX_FLAGS}" STREQUAL "")
    SET(CMAKE_CXX_FLAGS "-ftemplate-depth=1024 -Qunused-arguments -Wno-invalid-offsetof ${SSE_FLAGS}") # Unfortunately older Clang versions do not have this: -Wno-unnamed-type-template-args
    if(APPLE AND WITH_CUDA AND CUDA_FOUND)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    endif()
  endif()
  SET(CLANG_LIBRARIES "stdc++")
endif()

### ---[ Find universal dependencies
# the gcc-4.2.1 coming with MacOS X is not compatible with the OpenMP pragmas we use, so disabling OpenMP for it
if((NOT APPLE) OR (NOT CMAKE_COMPILER_IS_GNUCXX) OR (GCC_VERSION VERSION_GREATER 4.2.1) OR (CMAKE_COMPILER_IS_CLANG))
  find_package(OpenMP)
endif()
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message (STATUS "Found OpenMP")
  if(MSVC)
    if(MSVC_VERSION EQUAL 1500)
      set(OPENMP_DLL VCOMP90)
    elseif(MSVC_VERSION EQUAL 1600)
      set(OPENMP_DLL VCOMP100)
    elseif(MSVC_VERSION EQUAL 1700)
      set(OPENMP_DLL VCOMP110)
    elseif(MSVC_VERSION EQUAL 1800)
      set(OPENMP_DLL VCOMP120)
    elseif(MSVC_VERSION EQUAL 1900)
      set(OPENMP_DLL VCOMP140)
    elseif(MSVC_VERSION EQUAL 1910)
      set(OPENMP_DLL VCOMP140)
    endif()
    if(OPENMP_DLL)
      set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DELAYLOAD:${OPENMP_DLL}D.dll")
      set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DELAYLOAD:${OPENMP_DLL}.dll")
    else(OPENMP_DLL)
      message(WARNING "Delay loading flag for OpenMP DLL is invalid.")
    endif(OPENMP_DLL)
  endif(MSVC)
else(OPENMP_FOUND)
  message (STATUS "Not found OpenMP")
endif()

# Eigen (required)
find_package(Eigen REQUIRED)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})

# Find VTK
option(WITH_VTK "Build VTK-Visualizations" TRUE)
if(WITH_VTK AND NOT ANDROID)
  find_package(VTK)
  if(VTK_FOUND)
    if(NOT DEFINED VTK_RENDERING_BACKEND)
      # On old VTK versions this variable does not exist. In this case it is
      # safe to assume OpenGL backend
      set(VTK_RENDERING_BACKEND "OpenGL")
    endif()
    message(STATUS "VTK_MAJOR_VERSION ${VTK_MAJOR_VERSION}, rendering backend: ${VTK_RENDERING_BACKEND}")
    if (PCL_SHARED_LIBS OR
        (NOT (PCL_SHARED_LIBS) AND NOT (VTK_BUILD_SHARED_LIBS)))
      set(VTK_FOUND TRUE)
      find_package (QVTK)
      if (${VTK_MAJOR_VERSION} VERSION_LESS "6.0")
         message(STATUS "VTK found (include: ${VTK_INCLUDE_DIRS}, lib: ${VTK_LIBRARY_DIRS})")
         link_directories(${VTK_LIBRARY_DIRS})
      else(${VTK_MAJOR_VERSION} VERSION_LESS "6.0")
         include (${VTK_USE_FILE})
         message(STATUS "VTK found (include: ${VTK_INCLUDE_DIRS}, lib: ${VTK_LIBRARIES}")
      endif (${VTK_MAJOR_VERSION} VERSION_LESS "6.0")
      if (APPLE)
          option (VTK_USE_COCOA "Use Cocoa for VTK render windows" ON)
          MARK_AS_ADVANCED (VTK_USE_COCOA)
      endif (APPLE)
      if(${VTK_RENDERING_BACKEND} STREQUAL "OpenGL")
        set(VTK_RENDERING_BACKEND_OPENGL_VERSION "1")
      elseif(${VTK_RENDERING_BACKEND} STREQUAL "OpenGL2")
        set(VTK_RENDERING_BACKEND_OPENGL_VERSION "2")
      endif()
      set(HAVE_VTK ON)
    else ()
      set(VTK_FOUND OFF)
      set(HAVE_VTK OFF)
      message ("Warning: You are to build PCL in STATIC but VTK is SHARED!")
      message ("Warning: VTK disabled!")
    endif ()
  endif(VTK_FOUND)
else(WITH_VTK AND NOT ANDROID)
  set(VTK_FOUND OFF)
  set(HAVE_VTK OFF)
endif(WITH_VTK AND NOT ANDROID)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

if (VTK_FOUND)
  include("${VTK_USE_FILE}")	
endif()

find_package(PCL 1.8 REQUIRED)
if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  
  # For Use Not PreCompiled Features 
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )
endif()

include_directories("./include/")

add_executable(cvfh_project src/global_classification.cpp src/global_nn_classifier.cpp src/global_nn_recognizer_crh.cpp 
src/global_nn_recognizer_cvfh.cpp src/local_recognizer.cpp src/openni_frame_source.cpp src/render_views_tesselated_sphere.cpp)

target_link_libraries(cvfh_project ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${OpenCV_LIBS})


