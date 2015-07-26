#include <iostream>

#define TVMTL_MANIFOLD_DEBUG
#include "../core/manifold.hpp"

using namespace tvmtl;

typedef Manifold<SO, 3> mf_t;

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


}



int main(int argc, const char *argv[])
{
 
	son_mat s1, s2;
	s1 = son_mat::Random();
	mf_t::projector(s1);
	s2 = son_mat::Random();
	mf_t::projector(s2);
	s2*=-1.0;
	
	//s1 = son_mat::Identity();
	//s2 << 0, -1, 0, 1, 0, 0, 0, 0, 1;

	std::cout << "\n\nOrthogonality test" << std::endl;
	std::cout << s1*s1.transpose() << std::endl;
	std::cout << s2*s2.transpose() << std::endl;

	std::cout << "\n\n Permutation matrix" << std::endl;
	std::cout << mf_t::permutation_matrix.toDenseMatrix()  << std::endl;


	std::cout << "\n\nRANDOM Matrices" << std::endl;
	test(s1,s2);



	return 0;
}
