#ifndef TVMTL_VISUALIZATION_HPP
#define TVMTL_VISUALIZATION_HPP

//System includes
#include <iostream>
#include <cmath>
#include <string>

//Eigen includes
#include <Eigen/Geometry>

//OpenCV includes
#include <opencv2/opencv.hpp>

//OpenGL includes
#include <GL/freeglut.h>

// video++ includes
#include <vpp/vpp.hh>

// own includes
#include "enumerators.hpp"
#include "func_ptr_utils.hpp"
#include "manifold.hpp"
#include "data.hpp"


namespace tvmtl{

// Primary Template
template < enum MANIFOLD_TYPE MF, int N, class DATA, int dim=2>
class Visualization {
};
} // end namespace tvmtl

#include "visualization2d.hpp"
#include "visualization3d.hpp"

#endif  
