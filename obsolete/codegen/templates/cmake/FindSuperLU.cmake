SET(SUPERLU_DIR    "$ENV{SUPERLU_DIR}")
SET(SUPERLU_INCLUDE "${SUPERLU_DIR}/include")
SET(SUPERLU_LIB_DIR "${SUPERLU_DIR}/lib")

FIND_LIBRARY(SUPERLU_LIB NAMES libsuperlu.a
             HINTS ${SUPERLU_LIB_DIR})

FIND_LIBRARY(BLAS_LIB NAMES libblas.a
             HINTS ${SUPERLU_LIB_DIR})

SET(SUPERLU_LIBRARIES ${SUPERLU_LIB} ${BLAS_LIB})
SET(SUPERLU_INCLUDES ${SUPERLU_INCLUDE})
  
SET (SUPERLU_INCLUDES  ${SUPERLU_INCLUDES}  CACHE STRING "SUPERLU include path" FORCE)
SET (SUPERLU_LIBRARIES ${SUPERLU_LIBRARIES} CACHE STRING "SUPERLU libraries"    FORCE)

MARK_AS_ADVANCED(SUPERLU_INCLUDES SUPERLU_LIBRARIES)

INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args (SUPERLU
                 "SUPERLU could not be found.  Be sure to set SUPERLU_DIR."
                 SUPERLU_INCLUDES SUPERLU_LIBRARIES)

IF (SUPERLU_FOUND)
  INCLUDE_DIRECTORIES(${SUPERLU_INCLUDES})
  MESSAGE(STATUS "SUPERLU Has been found")
  MESSAGE(STATUS "SUPERLU_DIR: ${SUPERLU_DIR}")
  MESSAGE(STATUS "SUPERLU_INCLUDES: ${SUPERLU_INCLUDES}")
  MESSAGE(STATUS "SUPERLU_LIBRARIES: ${SUPERLU_LIBRARIES}")
ENDIF(SUPERLU_FOUND)
