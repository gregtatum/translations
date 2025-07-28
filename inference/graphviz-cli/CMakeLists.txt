add_executable(graphviz main.cpp)
target_link_libraries(graphviz PRIVATE bergamot-translator-source)

# Place the executable in the top-level build directory
set_target_properties(graphviz PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
