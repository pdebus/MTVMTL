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
#include <GL/glew.h>
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

struct Camera{
  
    Camera();
    Camera(float x, float y, float z, float xd, float yd, float zd, float xa, float ya, float step):
	xPos_(x), yPos_(y), zPos_(z), xDir_(xd), yDir_(yd), zDir_(zd), xAngle_(xa), yAngle_(ya), step_(step)
	{}

    void  printParams();

    float xPos_, yPos_, zPos_;
    float xDir_, yDir_, zDir_;
    float xAngle_, yAngle_;
    float step_;
};


Camera::Camera(){
    xAngle_= 0.0;
    yAngle_ = 0.0;

    xPos_ = yPos_ = 0.0;
    zPos_ = 1.0;
    xDir_ = std::sin(yAngle_);
    zDir_ = - std::cos(yAngle_);
    yDir_ = 0.0;
    step_ = 0.05;
}

void Camera::printParams(){
    std::cout << "Position: (" << xPos_ << ", " << yPos_ << ", " << zPos_ << "), Direction: (" << xDir_ << ", " << yDir_ << ", " << zDir_ << "), Angles: (" << xAngle_ << ", " << yAngle_ << ")\n"; 
}


} // end namespace tvmtl

#include "visualization2d.hpp"
#include "visualization3d.hpp"

#endif  
