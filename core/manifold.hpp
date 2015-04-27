#ifndef TVMTL_MANIFOLD_HPP
#define TVMTL_MANIFOLD_HPP

#include <Eigen/Core>

#include "enumerators.hpp"

namespace tvmtl {

// Primary Template
template < enum MANIFOLD_TYPE MF, int N >
struct Manifold { };


// Specialization EUCLIDIAN
template < int N >
struct Manifold< EUCLIDIAN, N > {
    
    public:
	static const MANIFOLD_TYPE MyType;

	// Scalar type of manifold
	typedef double scalar_type;
	typedef double dist_type;
	
	// Value Typedef
	typedef Eigen::Matrix< scalar_type, N, 1>   value_type;
	typedef value_type&			    ref_type;
	typedef const value_type&		    cref_type;

	//TODO: Does Ref<> variant also work?
	//typedef Eigen::Ref< value_type >		   ref_type;
	//typedef const Eigen::Ref< const value_type > cref_type;

	// Derivative Typedefs
	typedef Eigen::Matrix<scalar_type, 2*N, 1>   deriv1_type;
	typedef deriv1_type&			     deriv1_ref_type;
	
	//typedef Eigen::Ref<deriv1_type>		     deriv1_ref_type;
	typedef Eigen::Matrix<scalar_type, 2*N, 2*N> deriv2_type;
	typedef deriv2_type&			     deriv2_ref_type;
	//typedef Eigen::Ref<deriv2_type>		     deriv2_ref_type;


	// Manifold distance functions (for IRLS)
	inline static dist_type dist_squared(cref_type x, cref_type y);
	inline static void deriv1_dist_squared(cref_type x, cref_type y, deriv1_ref_type result);
	inline static void deriv2_dist_squared(cref_type x, cref_type y, deriv2_ref_type result);

	// Manifold exponentials und logarithms ( for Proximal point)
	inline static void exp(cref_type x, cref_type y, ref_type result);
	inline static void log(cref_type x, cref_type y, ref_type result);
};


// Specialization SPD
template < int N >
struct Manifold< SPD, N > {
    public:
	static const MANIFOLD_TYPE MyType = SPD;
};

// Specialization SPHERE
template < int N>
struct Manifold< SPHERE, N> {

};

// Specialization SO(N)
template < int N>
struct Manifold< SO, N> {

};


/*-----IMPLEMENTATION EUCLIDIAN----------*/
template <int N>
const MANIFOLD_TYPE Manifold < EUCLIDIAN, N>::MyType = EUCLIDIAN; // Outside definition to avoid linker error

// Squared Euclidian distance function
template <int N>
inline typename Manifold < EUCLIDIAN, N>::dist_type Manifold < EUCLIDIAN, N>::dist_squared( cref_type x, cref_type y ){
    value_type v = x-y;
    return v.squaredNorm();
}

// Derivative of Squared Euclidian distance
template <int N>
inline void Manifold < EUCLIDIAN, N>::deriv1_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    result << 2 * (x-y), -2 * (x-y);
    //result.head<N>() = 2 * (x-y);
    //result.tail<N>() = -2 * (x-y);
}


// Second Derivative of Squared Euclidian distance
template <int N>
inline void Manifold < EUCLIDIAN, N>::deriv2_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    
    result << 2 * Eigen::Matrix<scalar_type, N, N>::Identity(), - 2 * Eigen::Matrix<scalar_type, N, N>::Identity(), -2 * Eigen::Matrix<scalar_type, N, N>::Identity(), 2 * Eigen::Matrix<scalar_type, N, N>::Identity();
    //result = 2 * deriv2_type::Identity();
    //result.bottomLeftCorner<N,N>() = -2 * Eigen::Matrix<scalar_type, N, N>::Identity();
    //result.topRightCorner<N,N>() = -2 * Eigen::Matrix<scalar_type, N, N>::Identity();
}

// Exponential and Logarithm Map
template <int N>
inline void Manifold <EUCLIDIAN, N>::exp(cref_type x, cref_type y, ref_type result){
    result=x+y;
}

template <int N>
inline void Manifold <EUCLIDIAN, N>::log(cref_type x, cref_type y, ref_type result){
    result = y-x;
}


} // end namespace tvmtl







#endif
