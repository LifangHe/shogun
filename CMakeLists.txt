# Instructions:
# $ mkdir build
# $ cd build
# $ cmake ..
# $ make

project(shogun)
cmake_minimum_required(VERSION 2.8)

################# VARIABLES #####################
SET(DEFINES "-DSWIG_TYPE_TABLE=shogun;-DSHOGUN")
SET(EXT_SRC_HEADER "h")
SET(EXT_SRC_C "c")
SET(EXT_SRC_CPP "cpp")
SET(EXT_SRC_SWIG "_wrap.cxx")
SET(EXT_SRC_TEST "_unittest.cc")
SET(EXT_INP_SWIG "i")
SET(EXT_IF_SWIG "py")
SET(EXT_CPP_TMP "${EXT_SRC_CPP}.templ")
SET(EXT_CPP_SH "${EXT_SRC_CPP}.sh")
SET(EXT_CPP_PY "${EXT_SRC_CPP}.py")

SET(LIBSHOGUN_SRC_DIR ${CMAKE_SOURCE_DIR}/src/shogun)
SET(COMMON_MODULAR_SRC_DIR ${CMAKE_SOURCE_DIR}/src/interfaces/modular/)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

################ COMPILER #######################
# g++ version needs to be => 4.3
IF(CMAKE_COMPILER_IS_GNUCXX)
	# in order to support cmake 2.8.7 and older
	IF(NOT CMAKE_CXX_COMPILER_VERSION)
		include(CheckCompiler)
	ENDIF()
	IF("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "4.3.0")
		MESSAGE(FATAL_ERROR "g++ version is too old")
	ENDIF()
ENDIF()

############### VERSION #####################
FILE(STRINGS "@CMAKE_SOURCE_DIR@/src/NEWS" NEWS LIMIT_COUNT 5)
STRING(REGEX REPLACE ".*SHOGUN Release version ([0-9.]*).*" "\\1" VERSION "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*\\(libshogun ([0-9.]*).*" "\\1" LIBSHOGUNVER "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*\\(libshogun ([0-9]*).*" "\\1" LIBSHOGUNSO "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*data ([0-9.]*).*" "\\1" DATAVER "${NEWS}")

# Detect OS
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	SET(DARWIN 1)
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	SET(LINUX 1)
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
	SET(FREEBSD 1)
ENDIF()

# Get processor type, sets MACHINE macro
execute_process(COMMAND uname -m
    OUTPUT_VARIABLE MACHINE_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
SET(MACHINE ${MACHINE_OUTPUT})

#Build type
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug CACHE STRING
       "Choose the type of build, options are: Debug Release."
       FORCE )
endif()

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
	#		COMP_OPTS="-g $COMP_OPTS";
	#	COMPFLAGS_SWIG_CPP="-g $COMPFLAGS_SWIG_CPP"
ENDIF()

################# DATATYPES #################
LIST(APPEND DEFINES -DUSE_CHAR)
LIST(APPEND DEFINES -DUSE_BOOL)
LIST(APPEND DEFINES -DUSE_UINT8)
LIST(APPEND DEFINES -DUSE_UINT16)
#LIST(APPEND DEFINES -DUSE_UINT32)
LIST(APPEND DEFINES -DUSE_UINT64)
#LIST(APPEND DEFINES -DUSE_INT8)
#LIST(APPEND DEFINES -DUSE_INT16)
LIST(APPEND DEFINES -DUSE_INT32)
LIST(APPEND DEFINES -DUSE_INT64)
LIST(APPEND DEFINES -DUSE_FLOAT32)
LIST(APPEND DEFINES -DUSE_FLOAT64)
#LIST(APPEND DEFINES -DUSE_FLOATMAX)

# detect word size
IF(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT DARWIN)
	SET(CMAKE_SWIG_FLAGS -DSWIGWORDSIZE64)
ENDIF()

#interfaces
OPTION(PythonModular "Python Modular" OFF)
OPTION(LuaModular "Lua Modular" OFF)
OPTION(RModular "R Modular" OFF)
OPTION(OctaveModular "Octave Modular" OFF)
OPTION(JavaModular "Java Modular" OFF)
OPTION(PerlModular "Perl Modular" OFF)
OPTION(RubyModular "Ruby Modular" OFF)
OPTION(CSharpModular "CSharp Modular" OFF)

# SVMLight
OPTION(USE_SVMLIGHT "SVMLight" ON)
SET(SVMLightWarning "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
SET(SVMLightWarning "${SVMLightWarning}\nWARNING: SHOGUN is built using SVMlight which was written")
SET(SVMLightWarning "${SVMLightWarning}\nby Thorsten Joachims and uses a different non GPL compatible license.")
SET(SVMLightWarning "${SVMLightWarning}\nTo build a fully GPL'd SHOGUN use")
SET(SVMLightWarning "${SVMLightWarning}\nThe SVMlight license can be found in LICENSE.SVMlight.  In case")
SET(SVMLightWarning "${SVMLightWarning}\nyou do not know or cannot agree to the licensing terms expressed in")
SET(SVMLightWarning "${SVMLightWarning}\nLICENSE.SVMlight press ctrl+c to abort configure now.")
SET(SVMLightWarning "${SVMLightWarning}\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
IF(USE_SVMLIGHT)
	LIST(APPEND DEFINES -DUSE_SVMLIGHT)
ENDIF()

#Trace memory allocs
OPTION(TRACE_MEMORY_ALLOCS "Memory allocation tracing" OFF)
IF(TRACE_MEMORY_ALLOCS)
	LIST(APPEND DEFINES -DTRACE_MEMORY_ALLOCS)
ENDIF()

# HMM
OPTION(USE_HMMDEBUG "HMM cache" OFF)
IF(USE_HMMDEBUG)
	LIST(APPEND DEFINES -DUSE_HMMDEBUG)
ENDIF()

OPTION(USE_HMMCACHE "HMM cache" ON)
IF(USE_HMMCACHE)
	LIST(APPEND DEFINES -DUSE_HMMCACHE)
ENDIF()

OPTION(USE_HMMPARALLEL "Parallel structures in hmm training" OFF)
IF(USE_HMMPARALLEL)
	SET(USE_HMMPARALLELSTRUCTURES 1)
	LIST(APPEND DEFINES -DUSE_HMMPARALLELSTRUCTURES -DUSE_HMMPARALLEL)
ENDIF()

# Viterbi path debug
OPTION(USE_PATHDEBUG "Viterbi path debugging" OFF)
IF(USE_PATHDEBUG)
	LIST(APPEND DEFINES -DUSE_PATHDEBUG)
ENDIF()

# big states
OPTION(USE_BIGSTATES "Big (16bit) state" ON)
IF(USE_BIGSTATES)
	LIST(APPEND DEFINES -DUSE_BIGSTATES)
ENDIF()

# Large file
OPTION(HAVE_LARGEFILE "Large file support" ON)
IF(HAVE_LARGEFILE)
	LIST(APPEND DEFINES -DHAVE_LARGEFILE)
ENDIF()

#kernelcache to use 4-byte-floating-point values instead of 8-byte-doubles
OPTION(USE_SHORTREAL_KERNELCACHE "Kernelcache to use 4-byte-floating-point values instead of 8-byte-doubles" ON)
IF(USE_SHORTREAL_KERNELCACHE)
	LIST(APPEND DEFINES -DUSE_SHORTREAL_KERNELCACHE)
ENDIF()

# PRNG settings
SET(MEXP "19937" CACHE STRING "PRNG Mersenne exponent")
SET(SFMT_MEXP ${MEXP})
SET(DSFMT_MEXP ${MEXP})
LIST(APPEND DEFINES -DDSFMT_MEXP=${MEXP} -DSFMT_MEXP=${MEXP})

# Reference counting
OPTION(USE_REFERENCE_COUNTING "Reference Counting" ON)
IF(USE_REFERENCE_COUNTING)
	LIST(APPEND DEFINES -DUSE_REFERENCE_COUNTING)
ENDIF()

######################### LIBRARIES #########################

# check for math functions
include(CheckFunctionExists)
FIND_LIBRARY(M_LIB m)
CHECK_FUNCTION_EXISTS(log2 HAVE_LOG2)
IF(HAVE_LOG2)
	LIST(APPEND DEFINES -DHAVE_LOG2)
ENDIF()

CHECK_FUNCTION_EXISTS(powl HAVE_POWL)
IF(HAVE_POWL)
	LIST(APPEND DEFINES -DHAVE_POWL)
ENDIF()

CHECK_FUNCTION_EXISTS(lgammal HAVE_LGAMMAL)
IF(HAVE_LGAMMAL)
	LIST(APPEND DEFINES -DHAVE_LGAMMAL)
ENDIF()

CHECK_FUNCTION_EXISTS(sqrtl HAVE_SQRTL)
IF(HAVE_SQRTL)
	LIST(APPEND DEFINES -DHAVE_SQRTL)
ENDIF()

# check SSE and SSE2 intrinsics header
include(CheckIncludeFile)
CHECK_INCLUDE_FILE(xmmintrin.h HAVE_BUILTIN_VECTOR)
CHECK_INCLUDE_FILE(emmintrin.h HAVE_SSE2)
IF(HAVE_SSE2)
	LIST(APPEND DEFINES -DHAVE_SSE2)
ENDIF()

FIND_PACKAGE(Threads)
IF (CMAKE_USE_PTHREADS_INIT)
	SET(HAVE_PTHREAD 1)
	LIST(APPEND DEFINES -DHAVE_PTHREAD)
ENDIF()

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

FIND_PACKAGE(PkgConfig)

FIND_PACKAGE(LAPACK)
IF (LAPACK_FOUND)
	SET(HAVE_LAPACK 1)
	LIST(APPEND DEFINES -DHAVE_LAPACK)
	SET(POSTLINKFLAGS ${LAPACK_LIBRARIES})
ENDIF()

FIND_PACKAGE(GLPK)
IF (GLPK_FOUND)
	SET(USE_GLPK 1)
	LIST(APPEND DEFINES -DUSE_GLPK)
	INCLUDE_DIRECTORIES(${GLPK_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${GLPK_LIBRARY})
ENDIF()

FIND_PACKAGE(CPLEX)
IF (CPLEX_FOUND)
	SET(USE_CPLEX 1)
	LIST(APPEND DEFINES -DUSE_CPLEX)
	INCLUDE_DIRECTORIES(${CPLEX_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${CPLEX_LIBRARY})
ENDIF()

FIND_PACKAGE(ARPACK)
IF (ARPACK_FOUND)
	SET(HAVE_ARPACK 1)
	LIST(APPEND DEFINES -DHAVE_ARPACK)
	INCLUDE_DIRECTORIES(${ARPACK_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${ARPACK_LIB})
ENDIF()

FIND_PACKAGE(Eigen3)
IF (EIGEN_FOUND)
	SET(HAVE_EIGEN3 1)
	LIST(APPEND DEFINES -DHAVE_EIGEN3)
	INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})
ENDIF()

FIND_PACKAGE(SuperLU)
IF (SUPERLU_FOUND)
	SET(HAVE_SUPERLU 1)
	LIST(APPEND DEFINES -DHAVE_SUPERLU)
	INCLUDE_DIRECTORIES(${SUPERLU_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${SUPERLU_LIBRARIES})
ENDIF()

FIND_PACKAGE(NLopt)
IF (NLOPT_FOUND)
	SET(HAVE_NLOPT 1)
	LIST(APPEND DEFINES -DHAVE_NLOPT)
	INCLUDE_DIRECTORIES(${NLOPT_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${NLOPT_LIBRARIES})
ENDIF()

FIND_PACKAGE(LpSolve)
IF (LPSOLVE_FOUND)
	SET(USE_LPSOLVE 1)
	LIST(APPEND DEFINES -DUSE_LPSOLVE -DYY_NEVER_INTERACTIVE -DPARSER_LP -DINVERSE_ACTIVE=INVERSE_LUSOL -DRoleIsExternalInvEngine)
	INCLUDE_DIRECTORIES(${LPSOLVE_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LPSOLVE_LIBRARIES})
ENDIF()

FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
	SET(HAVE_DOXYGEN 1)
	LIST(APPEND DEFINES -DHAVE_DOXYGEN)
ENDIF()

# detect PYTHON
FIND_PACKAGE(PythonInterp)
IF(NOT PYTHONINTERP_FOUND)
	MESSAGE(FATAL_ERROR "Currently we require python interpreter for build")
ENDIF()

# JSON
IF (PKG_CONFIG_FOUND)
	pkg_search_module(JSON libjson json json-c)
	IF (JSON_FOUND)
		SET(HAVE_JSON 1)
		LIST(APPEND DEFINES -DHAVE_JSON)
		INCLUDE_DIRECTORIES(${JSON_INCLUDE_DIRS})
		SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${JSON_LDFLAGS})
	ENDIF()
ENDIF()

FIND_PACKAGE(LibXml2)
IF (LIBXML2_FOUND)
	SET(HAVE_XML 1)
	LIST(APPEND DEFINES -DHAVE_XML)
	INCLUDE_DIRECTORIES(${LIBXML2_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LIBXML2_LIBRARIES})
ENDIF()

FIND_PACKAGE(HDF5)
IF (HDF5_FOUND)
	SET(HAVE_HDF5 1)
	LIST(APPEND DEFINES -DHAVE_HDF5)
	INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${HDF5_LIBRARIES})
ENDIF()

FIND_PACKAGE(ZLIB)
IF (ZLIB_FOUND)
	SET(USE_GZIP 1)
	LIST(APPEND DEFINES -DUSE_GZIP)
	INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${ZLIB_LIBRARIES})
ENDIF()

FIND_PACKAGE(BZip2)
IF (BZIP2_FOUND)
	SET(USE_BZIP2 1)
	LIST(APPEND DEFINES -DUSE_BZIP2)
	INCLUDE_DIRECTORIES(${BZIP_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${BZIP2_LIBRARIES})
ENDIF()

FIND_PACKAGE(LibLZMA)
IF (LIBLZMA_FOUND)
	SET(USE_LZMA 1)
	LIST(APPEND DEFINES -DUSE_LZMA)
	INCLUDE_DIRECTORIES(${LIBLZMA_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LIBLZMA_LIBRARIES})
ENDIF()

FIND_PACKAGE(SNAPPY)
IF (SNAPPY_FOUND)
	SET(USE_SNAPPY 1)
	LIST(APPEND DEFINES -DUSE_SNAPPY)
	INCLUDE_DIRECTORIES(${SNAPPY_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${SNAPPY_LIBRARIES})
ENDIF()

FIND_PACKAGE(LZO)
IF (LZO_FOUND)
	SET(USE_LZO 1)
	LIST(APPEND DEFINES -DUSE_LZO)
	INCLUDE_DIRECTORIES(${LZO_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LZO_LIBS})
ENDIF()

FIND_PACKAGE(Spinlock)
IF (SPINLOCK_FOUND)
	SET(USE_SPINLOCKS 1)
	LIST(APPEND DEFINES -DUSE_SPINLOCKS)
ENDIF()

# set DEFINES for defintions
add_definitions(${DEFINES})

#SWIG Interfaces
SET(CMAKE_SWIG_FLAGS "${CMAKE_SWIG_FLAGS};-w473;-w454;-w312;-w325;-fvirtual")
SET(CMAKE_SWIG_FLAGS "${CMAKE_SWIG_FLAGS};${DEFINES}")

OPTION(USE_SWIG_DIRECTORS "" OFF)

FIND_PACKAGE(SWIG 2.0.4)
IF (SWIG_FOUND)
	FIND_PACKAGE(PythonLibs)
	IF (PYTHONLIBS_FOUND)
		FIND_PACKAGE(NumPy)
		IF (NUMPY_FOUND)

			LIST(APPEND DEFINES -DHAVE_PYTHON)
			add_definitions(-DHAVE_PYTHON)
			SET(CMAKE_SWIG_FLAGS "${CMAKE_SWIG_FLAGS};-DHAVE_PYTHON")
			SET(PythonModular OFF CACHE BOOL ON)
			
			#custom swig flags for python modular interface
			SET(TARGET_SWIGFLAGS "-builtin\;-modern\;-modernargs")
			add_subdirectory(src/interfaces/python_modular)
		ENDIF()
	ENDIF()

	FIND_PACKAGE(Java)
	IF (Java_JAVAC_EXECUTABLE)
		FIND_PACKAGE(JNI)
		IF (JNI_FOUND)
			SET(JavaModular OFF CACHE BOOL ON)
			SET(TARGET_SWIGFLAGS "-package\;org.shogun")
			add_subdirectory(src/interfaces/java_modular)
		ENDIF()
	ENDIF()

	FIND_PACKAGE(Ruby)
	FIND_PACKAGE(Lua50)
	FIND_PACKAGE(Matlab)
	FIND_PACKAGE(Octave)
	#FIND_PACKAGE(R)
ENDIF()

IF (SVMLight)
	MESSAGE(STATUS ${SVMLightWarning})
ENDIF()

# Find GTEST and GMOCK frameworks
SET(GOOGLE_MOCK_SOURCE_DIR "" CACHE PATH "Path to the GMock source")
IF ("${GOOGLE_MOCK_SOURCE_DIR}" STREQUAL "" OR NOT EXISTS "${GOOGLE_MOCK_SOURCE_DIR}/CMakeLists.txt")
	MESSAGE("Unit testing is disabled")
ELSE()
	enable_testing()
	add_subdirectory(tests/unit)
ENDIF()

# save configuration options
SET(CONFIGURE_OPTIONS "TODO")
SET(COMPFLAGS_CPP "TODO")
SET(LINKFLAGS "TODO")

add_subdirectory(src/shogun)