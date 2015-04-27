#include <iostream>

#include "../core/manifold.hpp"


int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	typedef Manifold< EUCLIDIAN, 3 > mf_t;
	typedef typename mf_t::value_type vec3d;

	vec3d vec1, vec2;

	vec1 = vec3d::Random(); 
	vec2 = vec3d::Random();

	std::cout << mf_t::MyType << std::endl;
	
	std::cout << "Vector 1:\n" << vec1 << std::endl;
	std::cout << "\nVector 2:\n" << vec2 << std::endl;

	std::cout << "\nDistance:" << std::endl;
	std::cout << mf_t::dist_squared(vec1, vec2) << std::endl;

	typedef typename mf_t::deriv1_type vec6d;
	typedef typename mf_t::deriv2_type mat6x6;

	vec6d d1;
	mf_t::deriv1_dist_squared(vec1, vec2, d1);
	mat6x6 d2; 
	mf_t::deriv2_dist_squared(vec1, vec2, d2);
	
	std::cout << "\nFirst Derivative:" << std::endl;
	std::cout << d1 << std::endl;

	std::cout << "\nSecond Derivative:" << std::endl;
	std::cout << d2 << std::endl;

	return 0;
}
