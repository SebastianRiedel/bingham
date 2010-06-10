# include(FindLibraryWithDebug)

if (GOTO_INCLUDES AND GOTO_LIBRARIES)
  set(GOTO_FIND_QUIETLY TRUE)
endif (GOTO_INCLUDES AND GOTO_LIBRARIES)

find_path(GOTO_INCLUDES
  NAMES
  cblas.h
  PATHS
  $ENV{GOTODIR}/include
  ${INCLUDE_INSTALL_DIR}
)

find_file(GOTO_LIBRARIES libgotoblas.so PATHS /usr/lib $ENV{GOTODIR} ${LIB_INSTALL_DIR})
find_library(GOTO_LIBRARIES gotoblas PATHS $ENV{GOTODIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GOTO DEFAULT_MSG
                                  GOTO_INCLUDES GOTO_LIBRARIES)

mark_as_advanced(GOTO_INCLUDES GOTO_LIBRARIES)
