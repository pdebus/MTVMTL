#include <iostream>

#define TVMTL_MANIFOLD_DEBUG
#include "../core/manifold.hpp"

using namespace tvmtl;

typedef Manifold< SPHERE, 3 > mf_t;
typedef typename mf_t::value_type vec3d;


template <class T>
void test(T& vec1, T& vec2){

	std::cout << mf_t::MyType << std::endl;
	
	std::cout << "Vector 1:\n" << vec1 << std::endl;
	std::cout << "\nVector 2:\n" << vec2 << std::endl;

	std::cout << "\nDistance:" << std::endl;
	std::cout << mf_t::dist_squared(vec1, vec2) << std::endl;

	typedef typename mf_t::deriv2_type mat3x3;

	vec3d d1x,d1y;
	mf_t::deriv1x_dist_squared(vec1, vec2, d1x);
	mf_t::deriv1y_dist_squared(vec1, vec2, d1y);
	mat3x3 d2xx, d2xy, d2yy; 
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
 
	std::cout << "\n\nNAN TEST" << std::endl;
	std::cout << std::boolalpha << "isfinite(NaN) = " << std::isfinite(NAN) << '\n'
		             << "isfinite(Inf) = " << std::isfinite(INFINITY) << '\n'
			     << "isfinite(0.0) = " << std::isfinite(0.0) << '\n'
			     << "isfinite(0.0/0.0) = " << std::isfinite(0.0/0.0) << '\n'
			     << "isfinite(1.0/0.0) = " << std::isfinite(1.0/0.0) << '\n'
			     << "isfinite(exp(800)) = " << std::isfinite(std::exp(800)) << '\n';


	vec3d vec1, vec2;
	vec1 = vec3d::Random().normalized(); 
	vec2 = vec3d::Random().normalized();

	std::cout << "\n\nRANDOM VECTORS" << std::endl;
	test(vec1,vec2);

	std::cout << "\n\nSAME VECTORS" << std::endl;
	vec2 = vec1;
	test(vec1, vec2);

	std::cout << "\n\nANTIPODAL VECTORS" << std::endl;
	vec2 *= -1;
	test(vec1, vec2);


	return 0;
}
