cmake_minimum_required(VERSION 2.8.3)
project(uuv_simulation_evaluation)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

if(CATKIN_ENABLE_TESTING)
  find_package(rostest)
  foreach(T
      test/test_all.launch
      )
     add_rostest(${T})
  endforeach()

  foreach(UNIT_TESTS
    test/test_constraint.py
    test/test_cost_function.py
    test/test_evaluation_trajectory.py
    test/test_kpis.py
    catkin_add_nosetests(${UNIT_TESTS}))
  endforeach()

endif()
