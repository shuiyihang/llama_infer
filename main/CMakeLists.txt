
aux_source_directory(${CMAKE_SOURCE_DIR}/main DIR_MAIN)

add_executable(chat ${DIR_MAIN})
target_link_libraries(chat llama)
target_link_libraries(chat absl::base re2::re2 nlohmann_json::nlohmann_json)