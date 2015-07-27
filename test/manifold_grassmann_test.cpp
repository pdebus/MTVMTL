#include <iostream>
#include <cstdlib>

#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

#define TVMTL_MANIFOLD_DEBUG
#include "../core/manifold.hpp"

using namespace tvmtl;

typedef Manifold<GRASSMANN, 3, 2> mf_t;

typedef typename mf_t::value_type mat;


template <class T>
void test(T& vec1, T& vec2){

	std::cout << mf_t::MyType << std::endl;
	
	std::cout << "Vector 1:\n" << vec1 << std::endl;
	std::cout << "\nVector 2:\n" << vec2 << std::endl;

	std::cout << "\nDistance:" << std::endl;
	std::cout << mf_t::dist_squared(vec1, vec2) << std::endl;

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


	mf_t::tm_base_type t;
	mf_t::tangent_plane_base(vec1, t);
	std::cout << "\nTangent Base Restriction:" << std::endl;
	std::cout << t << std::endl;

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

	mf_t::value_type u, z, v;
	std::cout << "\n\nExponential map test:" << std::endl;
	mf_t::exp(vec1, vec2, z);
	std::cout << "Exp(s1,s2) = \n" << z << std::endl;


	mf_t::log(vec1, vec2, u);
	mf_t::exp(vec1, u, z);
	mf_t::log(vec1, z, v);
	std::cout << "\n\nLogarithm map test:" << std::endl;
	std::cout << "U = Log(s1, s2) = \n" << u << std::endl;
	std::cout << "V = Log(s1, z) = \n" << v << std::endl;
	std::cout << "Z = Exp(s1, U) = \n"  << z << "\n and s2  =\n" << vec2 << "\nshould have distance close to zero:\n";
	std::cout << "dist(Z, vec2) = " << mf_t::dist_squared(z,v) << std::endl;

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

	double h = 1e-4;
	std::cout << "\n\nTaylor expansion Derivative Tests with perturbation O(h) = O(" << h << ")" <<std::endl;
	mf_t::value_type dx, dy;

	Eigen::Matrix<mf_t::scalar_type, 3, 3> HXproj = Eigen::Matrix<mf_t::scalar_type, 3, 3>::Identity() - vec1 * vec1.transpose();
	Eigen::Matrix<mf_t::scalar_type, 3, 3> HYproj = Eigen::Matrix<mf_t::scalar_type, 3, 3>::Identity() - vec2 * vec2.transpose();

	dx = mat::Random(); mf_t::projector(dx); dx = h * HXproj * dx;
	Eigen::VectorXd vecdx = Eigen::Map<Eigen::VectorXd>(dx.data(), dx.size());
	
	dy = mat::Random(); mf_t::projector(dy); dy = h * HYproj * dy;
	Eigen::VectorXd vecdy = Eigen::Map<Eigen::VectorXd>(dy.data(), dy.size());
	
	double exact = mf_t::dist_squared(vec1+dx, vec2+dy);
	double taylor_order1 = mf_t::dist_squared(vec1, vec2) + d1x.cwiseProduct(dx).sum() + d1y.cwiseProduct(dy).sum();
	double taylor_order2 = taylor_order1 + 0.5 * d2xx.cwiseProduct(vecdx * vecdx.transpose()).sum() + 0.5 * d2yy.cwiseProduct(vecdy * vecdy.transpose()).sum() + d2xy.cwiseProduct(vecdx * vecdy.transpose()).sum();

	std::cout << "Error of first order Taylor " << std::abs(taylor_order1 - exact) << " = O(h^"<< std::log10(std::abs(taylor_order1 - exact))/std::log10(h)  << ") " <<std::endl; 
	std::cout << "Error of second order Taylor " << std::abs(taylor_order2 - exact) << " = O(h^"<< std::log10(std::abs(taylor_order2 - exact))/std::log10(h)  << ") " <<std::endl; 
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
