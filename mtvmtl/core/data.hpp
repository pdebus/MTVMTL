#ifndef TVTML_DATA_HPP
#define TVTML_DATA_HPP

// system includes
#include <cassert>
#include <limits>
#include <cmath>
#include <random>

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef TV_DATA_DEBUG
    #include <opencv2/highgui/highgui.hpp>
#endif

//Eigen includes
#include <Eigen/Core>
#include <Eigen/Geometry>

// video++ includes
#include <vpp/vpp.hh>
#include <vpp/utils/opencv_bridge.hh>

// own includes 
#include "manifold.hpp"

namespace tvmtl{

// Primary Template
template <typename MANIFOLD, int DIM >
class Data {
};


}// end namespace tvmtl

#include "data2d.hpp"
#include "data3d.hpp"

#endif
