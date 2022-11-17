cmake_minimum_required(VERSION 3.18)

project(cv_resize_impl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable (main main.cc)

if (UNIX)
  target_link_directories(main PUBLIC /usr/local/lib)
  set(OPENCV_LIBS opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc opencv_calib3d opencv_features2d)
  target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/includes/opencv-3.4)
else() # win
  target_link_directories(main PUBLIC D:/opencv/opencv/build/lib)
  set(OPENCV_LIBS opencv_core455d opencv_imgcodecs455d opencv_highgui455d opencv_imgproc455d opencv_calib3d455d opencv_features2d455d)
  target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/includes/opencv-4.6)
endif()

target_link_libraries(main PUBLIC ${OPENCV_LIBS})
#target_compile_options(main PUBLIC -Wall -Werror)

