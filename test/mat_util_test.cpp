    #include <Eigen/Dense>
    #include <Eigen/Sparse>
    #include <unsupported/Eigen/MatrixFunctions>
    #include <unsupported/Eigen/KroneckerProduct>

    #include <iostream>
    #include <cmath>

    #define TVMTL_MATRIX_UTILS_DEBUG

    #include <mtvmtl/core/matrix_utils.hpp>

    int main(){

    using namespace Eigen;

    Matrix3d R = Matrix3d::Random();    
    Matrix3d M = R + R.transpose() + R.rows()*Matrix3d::Identity();
    
    std::cout << "Matrix M :\n" << M << std::endl;
    std::cout << "\nlog(M) :\n" << M.log() << std::endl;

    for (int i = 0; i < 3; i++) {
    	for (int j = 0; j < 3; j++) {
	    Matrix3d E = Matrix3d::Zero();
	    MatrixXd D = MatrixXd::Zero(3,3);
	    E(i,j)=1.0;
	    tvmtl::MatrixLogarithmFrechetDerivative(M, E, D);
	    std::cout << "\ndlog(M,E_" << i+1 << j+1 << ") :\n" << D << std::endl;
	    Map<VectorXd> V(D.data(), D.size());
	    std::cout << "\nRowwise Vectorized:\n" << V << std::endl;
    	}
    }

    Matrix<double, 9 ,9>  Result;
    tvmtl::KroneckerDLog(M, Result);
    std::cout << "Full Kronecker Representation of DLog:\n" << Result << std::endl; 


    std::cout << "\n\nPermutation Matrix Test:\n" << std::endl;

    PermutationMatrix<9, 9, int> P;
    P.setIdentity();
    for(int i=0; i<3; i++)
	for(int j=0; j<i; j++)
	    P.applyTranspositionOnTheRight(j*3+i,i*3+j);
	
    std::cout << P*Matrix<double, 9,9>::Identity() << std::endl;

    std::cout << "\n\n Simple Rotation Matrix Test\n" << std::endl;

    Matrix3d s1 = Matrix3d::Identity();
    Matrix3d s2;
    s2 << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    Matrix3d m2 = s1.transpose()*s2; 

    std::cout << "\nMatrix 1\n" << s1 << std::endl;
    std::cout << "\nMatrix 2\n" << s2 << std::endl;
    std::cout << "\nDlog Argument\n" << m2 << std::endl;
    

    tvmtl::KroneckerDLog(m2, Result);
    std::cout << "\n\nFull Kronecker Representation of DLog:\n" << Result << std::endl; 


    Matrix3d m3;
    m3 << 4102350, -2446420, 2209640, -2446420,  1458920, -1317710, 2209640, -1317710, 1190180;

    tvmtl::KroneckerDLog(m3, Result);
    std::cout << "\n\nPathological Matrix test:\n" << Result << std::endl;
    



    return 0;
    }
