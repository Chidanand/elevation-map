cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project( reproject_image_to_point_cloud)

find_package( PCL 1.2 REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED )
add_executable( reproject_image_to_point_cloud reproject_image_to_point_cloud.cpp )
target_link_libraries( reproject_image_to_point_cloud ${OpenCV_LIBS} ${PCL_LIBRARIES})

