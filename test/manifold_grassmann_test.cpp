#include <iostream>
#include <cstdlib>

#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

//#define TVMTL_MANIFOLD_DEBUG
//#define TVMTL_MANIFOLD_DEBUG_GRASSMANN
#include <mtvmtl/core/manifold.hpp>

using namespace tvmtl;

const int N=3;
const int P=2;

typedef Manifold<GRASSMANN, N, P> mf_t;

typedef typename mf_t::value_type mat;


template <class T>
void test(T& vec1, T& vec2){

	std::cout << mf_t::MyType << std::endl;
	
	std::cout << "Vector 1:\n" << vec1 << std::endl;
	std::cout << "\nVector 2:\n" << vec2 << std::endl;

	std::cout<< "\n\n==========DISTANCES TEST==========" << std::endl;
	std::cout << "Geodesic distance: " << mf_t::distGeod_squared(vec1, vec2) << std::endl;
	std::cout << "Projection F distance: " << mf_t::distPF_squared(vec1, vec2) << std::endl;

	std::cout<< "\n\n==========PERMUTATION MATRIX TEST==========" << std::endl;

	auto vec1t = vec1.transpose();
	Eigen::VectorXd vecvec1 = Eigen::Map<Eigen::VectorXd>(vec1.data(), vec1.size());
	Eigen::VectorXd vecvec1t = Eigen::Map<Eigen::VectorXd>(vec1t.data(), vec1t.size());

	std::cout << "\n\nTest of the Permutation Matrix Knp:" <<std::endl;
	std::cout << "s1:\n" << vec1 << std::endl;
	std::cout << "s1^t:\n" << vec1t << std::endl;
	std::cout << "\nVectorized s1:\n" << vecvec1 << std::endl;
	std::cout << "\nVectorized s1^t:\n" << vecvec1t << std::endl;
	std::cout << "\nPermutation Matrix:\n" << mf_t::permutation_matrix.toDenseMatrix() << std::endl;
	std::cout << "\nPermutation Matrix Indices:\n" << mf_t::permutation_matrix.indices() << std::endl;
	std::cout << "\nP * Vectorized s1:\n" << mf_t::permutation_matrix * vecvec1 << std::endl;

	std::cout<< "\n\n==========DERIVATIVE COMPUTATION TEST==========" << std::endl;
	typedef typename mf_t::deriv2_type mat9x9;

	mat d1x,d1y;
	mf_t::deriv1x_dist_squared(vec1, vec2, d1x);
	mf_t::deriv1y_dist_squared(vec1, vec2, d1y);
	mat9x9 d2xx, d2xy, d2yy; 
	mf_t::deriv2xx_dist_squared(vec1, vec2, d2xx);
	mf_t::deriv2xy_dist_squared(vec1, vec2, d2xy);
	mf_t::deriv2yy_dist_squared(vec1, vec2, d2yy);
	
	std::cout << "\nFirst Derivative:" << std::endl;
	std::cout << d1x << std::endl;
	std::cout << std::endl;
	std::cout << d1y << std::endl;

	std::cout << "\nSecond Derivative:" << std::endl;
	std::cout << d2xx << std::endl;
	std::cout << std::endl;
	std::cout << d2xy << std::endl;
	std::cout << std::endl;
	std::cout << d2yy << std::endl;

	std::cout<< "\n\n==========TANGENT SPACE BASIS TEST==========" << std::endl;
	mf_t::tm_base_type t, t2;
	mf_t::tangent_plane_base(vec1, t);
	std::cout << "\nTangent Base Restriction:" << std::endl;
	std::cout << t << std::endl;
	std::cout << "\nTangent Base Restriction alternative calculatiopn: " << std::endl;
	Eigen::HouseholderQR<mf_t::value_type> qr(vec1);
	Eigen::Matrix<mf_t::scalar_type, N, N> Q = qr.householderQ();
	Eigen::Matrix<mf_t::scalar_type, N, N - P> vec1orth = Q.rightCols(N-P);
	t2 = Eigen::kroneckerProduct(Eigen::Matrix<mf_t::scalar_type, P, P>::Identity(), vec1orth.transpose());
	std::cout << t2 << std::endl;

	std::cout<< "\n\n==========RESTRICTED DERIVIATIVES COMPUTATION TEST==========" << std::endl;
	mf_t::restricted_deriv2_type rd2xx, rd2xy, rd2yy;
	rd2xx = t.transpose() * d2xx * t;
	rd2xy = t.transpose() * d2xy * t;
	rd2yy = t.transpose() * d2yy * t;
	std::cout << "\nRestricted second Derivatives:" << std::endl;
	std::cout << rd2xx << std::endl;
	std::cout << std::endl;
	std::cout << rd2xy << std::endl;
	std::cout << std::endl;
	std::cout << rd2yy << std::endl;

	std::cout<< "\n\n==========EXPONENTIAL LOGARITHM CONSISTENCY CHECK==========" << std::endl;
	mf_t::value_type u, z, v;
	std::cout << "\n\nExponential map test:" << std::endl;
	mf_t::exp(vec1, vec2, z);
	std::cout << "Exp(s1,s2) = \n" << z << std::endl;


	mf_t::log(vec1, vec2, u);
	mf_t::exp(vec1, u, z);
	mf_t::log(vec1, z, v);
	std::cout << "\nLogarithm map test:" << std::endl;
	std::cout << "U = Log(s1, s2) = \n" << u << std::endl;
	std::cout << "\nThe next two expression should be the same:" << std::endl;
	std::cout << "distGeod_squared(s1, s2) = " << mf_t::distGeod_squared(vec1,vec2) << std::endl;
	std::cout << "tr U^TU = " << (u.transpose()*u).trace() << std::endl;
    

	std::cout << "\nZ = Exp(s1, U) = \n"  << z << "\n and s2  =\n" << vec2 << "\nshould have distance close to zero for geodesic and projection F-norm :\n";
	std::cout << "\ndistGeod_squared(Z, s2) = " << mf_t::distGeod_squared(z,vec2) << std::endl;
	std::cout << "distPF_squared(Z, s2) = " << mf_t::distPF_squared(z,vec2) << std::endl;
//	std::cout << "\nV = Log(s1, Z) = \n" << v << std::endl;


	std::cout<< "\n\n==========KARCHER MEAN CONSISTENCY TEST==========" << std::endl;
	mf_t::value_type kmean, d1, d2, sum; 
	mf_t::karcher_mean(kmean, vec1, vec2);
	
	std::cout << "\nK = K(s1, s2) =\n " << kmean << std::endl;
	mf_t::log(kmean, vec1, d1);
	mf_t::log(kmean, vec2, d2);
	sum = d1 + d2;
	std::cout << "\n||Sum_i Log(K, si)|| should be close to zero:" << (sum.transpose()*sum).trace() << std::endl;
	std::cout << "\nThe following four distances should all be approximately equal, the upper pair and the lower pair should be exactly equal:\n";
	std::cout << "Geodesic squared distance between s1 and K: " << mf_t::distGeod_squared(vec1, kmean) << std::endl;
	std::cout << "Geodesic squared distance between s2 and K: " << mf_t::distGeod_squared(vec2, kmean) << std::endl;
	std::cout << "Projection F squared distance between s1 and K: " << mf_t::distPF_squared(vec1, kmean) << std::endl;
	std::cout << "Projection F squared distance between s2 and K: " << mf_t::distPF_squared(vec2, kmean) << std::endl;

	std::cout<< "\n\n==========KARCHER MEAN NEWTON TEST==========" << std::endl;
	mf_t::value_type grad, X, Y, s;
	mat9x9 H;
	Eigen::VectorXd G, S;
	
	typedef Eigen::Matrix<mf_t::scalar_type, N, N> matn;
	typedef Eigen::Matrix<mf_t::scalar_type, P, P> matp;
	typedef Eigen::Matrix<mf_t::scalar_type, P, N> matpn;

	matn I = Eigen::Matrix<mf_t::scalar_type, N, N>::Identity();
	matp Ip = Eigen::Matrix<mf_t::scalar_type, P, P>::Identity();

	matn XorthP, H1;
	matp H2;

	
	X=vec1;

	for (int i=0; i<8; ++i){
	    XorthP = I - X * X.transpose();
	    grad = -XorthP * vec1 * vec1.transpose() * X - XorthP * vec2 * vec2.transpose() * X;
	    H1 = XorthP * vec1 * vec1.transpose() + XorthP * vec2 * vec2.transpose();
	    H2 = X.transpose() * vec1 * vec1.transpose() * X +  X.transpose() * vec2 * vec2.transpose() * X;
	    H = kroneckerProduct(Ip, H1) - kroneckerProduct(H2.transpose(), I);
	    G = Eigen::Map<Eigen::VectorXd>(grad.data(), grad.size());
	    S = H.fullPivLu().solve(G);
	    s = Eigen::Map<mf_t::value_type>(S.data());
	    mf_t::exp(X,s,X);
	}
	std::cout << "\nK = K(s1, s2) =\n " << X << std::endl;
	
	std::cout << "\n||grad|| should be close to zero:" << (grad.transpose()*grad).trace() << std::endl;
	std::cout << "\nThe following four distances should all be approximately equal, the upper pair and the lower pair should be exactly equal:\n";
	std::cout << "Geodesic squared distance between s1 and K: " << mf_t::distGeod_squared(vec1, X) << std::endl;
	std::cout << "Geodesic squared distance between s2 and K: " << mf_t::distGeod_squared(vec2, X) << std::endl;
	std::cout << "Projection F squared distance between s1 and K: " << mf_t::distPF_squared(vec1, X) << std::endl;
	std::cout << "Projection F squared distance between s2 and K: " << mf_t::distPF_squared(vec2, X) << std::endl;

	std::cout << "\n\n------WITH TANGENT SPACE RESTRICTION-----"    << std::endl;
	X=vec1;

	mf_t::restricted_deriv2_type HR;
	Eigen::VectorXd GR, SR;
	mf_t::tm_base_type tb;

	for (int i=0; i<8; ++i){
	    XorthP = I - X * X.transpose();
	    grad = -XorthP * vec1 * vec1.transpose() * X - XorthP * vec2 * vec2.transpose() * X;
	    H1 = XorthP * vec1 * vec1.transpose() + XorthP * vec2 * vec2.transpose();
	    H2 = X.transpose() * vec1 * vec1.transpose() * X +  X.transpose() * vec2 * vec2.transpose() * X;
	    H = kroneckerProduct(Ip, H1) - kroneckerProduct(H2.transpose(), I);
	    G = Eigen::Map<Eigen::VectorXd>(grad.data(), grad.size());
	    mf_t::tangent_plane_base(X, tb);
	    HR = tb.transpose() * H * tb;
	    GR = tb.transpose() * G;
	    SR = HR.fullPivLu().solve(GR);
	    S = tb * SR;
	    s = Eigen::Map<mf_t::value_type>(S.data());
	    mf_t::exp(X,s,X);
	}

	std::cout << "\nK = K(s1, s2) =\n " << X << std::endl;
	
	std::cout << "\n||grad|| should be close to zero:" << (grad.transpose()*grad).trace() << std::endl;
	std::cout << "\nThe following four distances should all be approximately equal, the upper pair and the lower pair should be exactly equal:\n";
	std::cout << "Geodesic squared distance between s1 and K: " << mf_t::distGeod_squared(vec1, X) << std::endl;
	std::cout << "Geodesic squared distance between s2 and K: " << mf_t::distGeod_squared(vec2, X) << std::endl;
	std::cout << "Projection F squared distance between s1 and K: " << mf_t::distPF_squared(vec1, X) << std::endl;
	std::cout << "Projection F squared distance between s2 and K: " << mf_t::distPF_squared(vec2, X) << std::endl;




	std::cout<< "\n\n==========2ND ORDER TAYLOR DERIVATIVE CONSISTENCY CHECK==========" << std::endl;
	double h = 1e-4;
	std::cout << "\n\nTaylor expansion Derivative Tests with perturbation O(h) = O(" << h << ")" <<std::endl;
	mf_t::value_type dx, dy;

	matn HXproj = I - vec1 * vec1.transpose();
	matn HYproj = I - vec2 * vec2.transpose();
	
	std::cout << "\n------Tangent vector tests-----"    << std::endl;
	mat Xpdx, Ypdy;
	dx = mat::Random(); mf_t::projector(dx); dx = h * HXproj * dx;
	mf_t::exp(vec1, dx, Xpdx);
	Eigen::VectorXd vecdx = Eigen::Map<Eigen::VectorXd>(dx.data(), dx.size());
	std::cout << "\ndx should be in the horizontal space at X, i.e. X^Tdx=0:\n "<< (vec1.transpose()*dx).norm() << std::endl;
	
	dy = mat::Random(); mf_t::projector(dy); dy = h * HYproj * dy;
	mf_t::exp(vec2, dy, Ypdy);
	Eigen::VectorXd vecdy = Eigen::Map<Eigen::VectorXd>(dy.data(), dy.size());
	std::cout << "\ndy should be in the horizontal space at Y, i.e. Y^Tdy=0:\n "<< (vec2.transpose()*dy).norm() << std::endl;
	
	double exact = mf_t::dist_squared(vec1 + dx, vec2 + dy);
	double exact2 = mf_t::dist_squared(Xpdx, Ypdy);
	double taylor_order1 = mf_t::dist_squared(vec1, vec2) + d1x.cwiseProduct(dx).sum() + d1y.cwiseProduct(dy).sum();
	double taylor_order2 = taylor_order1 + 0.5 * d2xx.cwiseProduct(vecdx * vecdx.transpose()).sum() + 0.5 * d2yy.cwiseProduct(vecdy * vecdy.transpose()).sum() + d2xy.cwiseProduct(vecdx * vecdy.transpose()).sum();

	std::cout << "\n\nError of first order Taylor " << std::abs(taylor_order1 - exact) << " = O(h^"<< std::log10(std::abs(taylor_order1 - exact))/std::log10(h)  << ") " <<std::endl; 
	std::cout << "Error of first order Taylor " << std::abs(taylor_order1 - exact2) << " = O(h^"<< std::log10(std::abs(taylor_order1 - exact2))/std::log10(h)  << ") " <<std::endl; 
	std::cout << "\nError of second order Taylor " << std::abs(taylor_order2 - exact) << " = O(h^"<< std::log10(std::abs(taylor_order2 - exact))/std::log10(h)  << ") " <<std::endl; 
	std::cout << "Error of second order Taylor " << std::abs(taylor_order2 - exact2) << " = O(h^"<< std::log10(std::abs(taylor_order2 - exact2))/std::log10(h)  << ") " <<std::endl; 


	std::cout<< "\n\n==========2ND ORDER TAYLOR DERIVATIVE ALTERNATIVE CHECK==========" << std::endl;
	h = 1e-4;
	std::cout << "\n\nTaylor expansion Derivative Tests with perturbation O(h) = O(" << h << ")" <<std::endl;
	std::cout << "\n------Single variable, fixed Y-----"    << std::endl;

	X=vec1;
	Y=vec2;

	XorthP = I - X * X.transpose();
	grad = -XorthP * Y * Y.transpose() * X;

	H1 = XorthP * Y * Y.transpose();
	H2 = X.transpose() * Y * Y.transpose() * X;
	H = kroneckerProduct(Ip, H1) - kroneckerProduct(H2.transpose(), I);

	double exact1, exact3, exact4;
	double taylor_firstorder = mf_t::dist_squared(X, Y) + grad.cwiseProduct(dx).sum();

	exact1 = mf_t::distGeod_squared(Xpdx,Y);
	exact2 = mf_t::distGeod_squared(X+dx,Y);
	exact3 = mf_t::dist_squared(Xpdx,Y);
	exact4 = mf_t::dist_squared(X+dx,Y);

	

	std::cout << "\nGeodesic distances at (X,Y): " << mf_t::distGeod_squared(X,Y) << std::endl;
	std::cout << "1)Geodesic distances at (exp_X(dx),Y): " << mf_t::distGeod_squared(Xpdx,Y) << std::endl;
	std::cout << "2)Geodesic distances at (X+dx,Y): " << mf_t::distGeod_squared(X+dx,Y) << std::endl;
	std::cout << "\nProjection F distance at (X,Y): " << mf_t::dist_squared(X,Y) << std::endl;
	std::cout << "3)Projection F distance at (exp_X(dx),Y): " << mf_t::dist_squared(Xpdx,Y) << std::endl;
	std::cout << "4)Projection F distance at (X+dx,Y): " << mf_t::dist_squared(X+dx,Y) << std::endl;


	std::cout << "\n1)Error of first order Taylor " << std::abs(taylor_firstorder - exact1) << " = O(h^"<< std::log10(std::abs(taylor_firstorder - exact1))/std::log10(h)  << ") " <<std::endl; 
	std::cout << "2)Error of first order Taylor " << std::abs(taylor_firstorder - exact2) << " = O(h^"<< std::log10(std::abs(taylor_firstorder - exact2))/std::log10(h)  << ") " <<std::endl; 
	std::cout << "\n3)Error of first order Taylor " << std::abs(taylor_firstorder - exact3) << " = O(h^"<< std::log10(std::abs(taylor_firstorder - exact3))/std::log10(h)  << ") " <<std::endl; 
	std::cout << "4)Error of first order Taylor " << std::abs(taylor_firstorder - exact4) << " = O(h^"<< std::log10(std::abs(taylor_firstorder - exact4))/std::log10(h)  << ") " <<std::endl; 

}

int main(int argc, const char *argv[])
{
	srand(42);

	mat s1, s2;
	s1 = mat::Random();
	mf_t::projector(s1);

	s2 = mat::Random();
	mf_t::projector(s2);

	std::cout << "s1=\n" << s1 << std::endl;

	std::cout << "\n\nRANDOM Matrices" << std::endl;
	test(s1,s2);

	return 0;
}
