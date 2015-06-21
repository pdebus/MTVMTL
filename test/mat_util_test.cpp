    #include <Eigen/Dense>
    #include <unsupported/Eigen/MatrixFunctions>

    #include <iostream>
    #include <cmath>

    #include "../core/matrix_utils.hpp"

    int main(){

    using namespace Eigen;

    Matrix3d R = Matrix3d::Random();    
    Matrix3d M = R + R.transpose() + R.rows()*Matrix3d::Identity();
    
    Matrix3d E = Matrix3d::Zero();
    E(0,0)=1.0;

    std::cout << "Matrix M :\n" << M << std::endl;
    std::cout << "\nlog(M) :\n" << M.log() << std::endl;
    tvmtl::MatrixLogarithmFrechetDerivative(M, E, R);
    std::cout << "\ndlog(M,E_11) :\n" << R << std::endl;

    return 0;
    }
