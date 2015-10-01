#include <iostream>
#include <chrono>

#include <unsupported/Eigen/MatrixFunctions>

#define TVMTL_MANIFOLD_DEBUG
#include <mtvmtl/core/manifold.hpp>

using namespace tvmtl;

typedef Manifold<SPD, 3> mf_t;

typedef typename mf_t::value_type son_mat;


template <class T>
void test(T& vec1, T& vec2){

	std::cout << mf_t::MyType << std::endl;
	
	std::cout << "Vector 1:\n" << vec1 << std::endl;
	std::cout << "\nVector 2:\n" << vec2 << std::endl;

	std::cout << "\nDistance:" << std::endl;
	std::cout << mf_t::dist_squared(vec1, vec2) << std::endl;

	typedef typename mf_t::deriv2_type mat9x9;

	son_mat d1x,d1y;
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

}



int main(int argc, const char *argv[])
{
 
	son_mat s1, s2;
	s1 = son_mat::Random();
	s1 = (s1 + s1.transpose()).exp();
	s2 = son_mat::Random();
	s2 = (s2 + s2.transpose()).exp();

	std::cout << "s1=\n" << s1 << std::endl;

	std::cout << "\n\nRANDOM Matrices" << std::endl;
	test(s1,s2);

	mf_t::deriv2_type result;
	int evals =1000000;
	std::cout << "\n\nTest of Matrix square root implementations for N=3" << std::endl;

	std::cout << "\n First implementation using n^2 directional derivatives: " << std::endl;
	std::cout << "Test with " <<  evals << " evaluations started..." << std::endl;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> t = std::chrono::duration<double>::zero();
	start = std::chrono::system_clock::now();
	    for(int i=0; i<evals; ++i){
		KroneckerDSqrt2(s1,result);
	    }
	end = std::chrono::system_clock::now();
	t = end - start; 
	std::cout << "\t Elapsed time: " << t.count() << " seconds." << std::endl;
	std::cout << "\t " << t.count() / evals << " seconds per evaluation. " << std::endl;


	std::cout << "\n Second implementation using n^2 x n^2 kronecker product inverse: " << std::endl;
	std::cout << "Test with " <<  evals << " evaluations started..."<< std::endl;
	start = std::chrono::system_clock::now();
	    for(int i=0; i<evals; ++i){
		KroneckerDSqrt(s1,result);
	    }
	end = std::chrono::system_clock::now();
	t = end - start; 
	std::cout << "\t Elapsed time: " << t.count() << " seconds." << std::endl;
	std::cout << "\t " << t.count() / evals << " seconds per evaluation. " << std::endl;

	return 0;
}
