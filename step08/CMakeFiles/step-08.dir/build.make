# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cs0uth/Documents/github/P2.3_seed/step08

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cs0uth/Documents/github/P2.3_seed/step08

# Include any dependencies generated for this target.
include CMakeFiles/step-08.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/step-08.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/step-08.dir/flags.make

CMakeFiles/step-08.dir/step-08.cc.o: CMakeFiles/step-08.dir/flags.make
CMakeFiles/step-08.dir/step-08.cc.o: step-08.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cs0uth/Documents/github/P2.3_seed/step08/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/step-08.dir/step-08.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/step-08.dir/step-08.cc.o -c /home/cs0uth/Documents/github/P2.3_seed/step08/step-08.cc

CMakeFiles/step-08.dir/step-08.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/step-08.dir/step-08.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cs0uth/Documents/github/P2.3_seed/step08/step-08.cc > CMakeFiles/step-08.dir/step-08.cc.i

CMakeFiles/step-08.dir/step-08.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/step-08.dir/step-08.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cs0uth/Documents/github/P2.3_seed/step08/step-08.cc -o CMakeFiles/step-08.dir/step-08.cc.s

# Object files for target step-08
step__08_OBJECTS = \
"CMakeFiles/step-08.dir/step-08.cc.o"

# External object files for target step-08
step__08_EXTERNAL_OBJECTS =

step-08: CMakeFiles/step-08.dir/step-08.cc.o
step-08: CMakeFiles/step-08.dir/build.make
step-08: /usr/lib/libdeal_II.g.so.9.1.1
step-08: /usr/lib/openmpi/libmpi_cxx.so
step-08: /usr/lib/openmpi/libmpi_usempif08.so
step-08: /usr/lib/openmpi/libmpi_usempi_ignore_tkr.so
step-08: /usr/lib/openmpi/libmpi_mpifh.so
step-08: /usr/lib/libz.so
step-08: /usr/lib/libboost_iostreams.so
step-08: /usr/lib/libboost_serialization.so
step-08: /usr/lib/libboost_system.so
step-08: /usr/lib/libboost_thread.so
step-08: /usr/lib/libboost_regex.so
step-08: /usr/lib/libboost_chrono.so
step-08: /usr/lib/libboost_date_time.so
step-08: /usr/lib/libboost_atomic.so
step-08: /usr/lib/libumfpack.so
step-08: /usr/lib/libcholmod.so
step-08: /usr/lib/libccolamd.so
step-08: /usr/lib/libcolamd.so
step-08: /usr/lib/libcamd.so
step-08: /usr/lib/libsuitesparseconfig.so
step-08: /usr/lib/libamd.so
step-08: /usr/lib/libmetis.so
step-08: /usr/lib/openmpi/libmpi.so
step-08: /usr/lib/liblapack.so
step-08: /usr/lib/libblas.so
step-08: CMakeFiles/step-08.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cs0uth/Documents/github/P2.3_seed/step08/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable step-08"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/step-08.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/step-08.dir/build: step-08

.PHONY : CMakeFiles/step-08.dir/build

CMakeFiles/step-08.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/step-08.dir/cmake_clean.cmake
.PHONY : CMakeFiles/step-08.dir/clean

CMakeFiles/step-08.dir/depend:
	cd /home/cs0uth/Documents/github/P2.3_seed/step08 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cs0uth/Documents/github/P2.3_seed/step08 /home/cs0uth/Documents/github/P2.3_seed/step08 /home/cs0uth/Documents/github/P2.3_seed/step08 /home/cs0uth/Documents/github/P2.3_seed/step08 /home/cs0uth/Documents/github/P2.3_seed/step08/CMakeFiles/step-08.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/step-08.dir/depend

