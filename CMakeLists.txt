cmake_minimum_required(VERSION 3.22)

set(CXX_DISABLE_WERROR 1)
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_NAME NewRLQPController)
set(PROJECT_DESCRIPTION "NewRLQPController")
set(PROJECT_URL "")

project(
  ${PROJECT_NAME}
  LANGUAGES CXX
  VERSION 1.0.0)
include(CTest)
enable_testing()

if(NOT TARGET mc_rtc::mc_control)
  find_package(mc_rtc REQUIRED)
endif()
find_package(mc_joystick_plugin REQUIRED)

add_subdirectory(src)

# if(BUILD_TESTING)
#   add_subdirectory(tests)
# endif()

# Bundled policies, installed so the configured yaml can reference them with
# a machine-independent path.
set(RL_POLICY_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/policy")
file(GLOB RL_POLICY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/policy/*.onnx")
install(FILES ${RL_POLICY_FILES} DESTINATION "${RL_POLICY_INSTALL_PREFIX}")

configure_file(etc/NewRLQPController.in.yaml
               "${CMAKE_CURRENT_BINARY_DIR}/etc/NewRLQPController.yaml")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/etc/NewRLQPController.yaml"
        DESTINATION "${MC_CONTROLLER_RUNTIME_INSTALL_PREFIX}/etc")
