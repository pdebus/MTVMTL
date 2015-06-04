#ifndef TVMTL_MANIFOLD_EUC_HPP
#define TVMTL_MANIFOLD_EUC_HPP

#include <cmath>
#include <Eigen/Core>

#include "enumerators.hpp"

namespace tvmtl {

// Specialization EUCLIDIAN
template < int N >
struct Manifold< EUCLIDIAN, N > {
    
    public:
	static const MANIFOLD_TYPE MyType;
	static const int manifold_dim ;
	static const int value_dim; // TODO: maybe rename to embedding_dim 

	// Scalar type of manifold
	//typedef double scalar_type;
	typedef double scalar_type;
	typedef double dist_type;
	
	// Value Typedef
	typedef Eigen::Matrix< scalar_type, N, 1>   value_type;
	typedef value_type&			    ref_type;
	typedef const value_type&		    cref_type;
	//TODO: Does Ref<> variant also work?
	//typedef Eigen::Ref< value_type >	    ref_type;
	//typedef const Eigen::Ref< const value_type > cref_type;

	
	// Tangent space typedefs
	typedef Eigen::Matrix < scalar_type, N, N> tm_base_type;
	typedef tm_base_type& tm_base_ref_type;

	// Derivative Typedefs
	typedef value_type			     deriv1_type;
	typedef deriv1_type&			     deriv1_ref_type;
	
	//typedef Eigen::Ref<deriv1_type>	     deriv1_ref_type;
	typedef Eigen::Matrix<scalar_type, N, N>     deriv2_type;
	typedef deriv2_type&			     deriv2_ref_type;
	typedef	Eigen::Matrix<scalar_type, N, N>     restricted_deriv2_type;
	//typedef Eigen::Ref<deriv2_type>	     deriv2_ref_type;


	// Manifold distance functions (for IRLS)
	inline static dist_type dist_squared(cref_type x, cref_type y);
	inline static void deriv1x_dist_squared(cref_type x, cref_type y, deriv1_ref_type result);
	inline static void deriv1y_dist_squared(cref_type x, cref_type y, deriv1_ref_type result);

	inline static void deriv2xx_dist_squared(cref_type x, cref_type y, deriv2_ref_type result);
	inline static void deriv2xy_dist_squared(cref_type x, cref_type y, deriv2_ref_type result);
	inline static void deriv2yy_dist_squared(cref_type x, cref_type y, deriv2_ref_type result);

	// Manifold exponentials und logarithms ( for Proximal point)
	template <typename DerivedX, typename DerivedY>
	inline static void exp(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y, Eigen::MatrixBase<DerivedX>& result);
	inline static void log(cref_type x, cref_type y, ref_type result);

	// Basis transformation for restriction to tangent space
	inline static void tangent_plane_base(cref_type x, tm_base_ref_type result);

	// Projection to manifold
	inline static void projector(ref_type x);	
};

/*-----IMPLEMENTATION EUCLIDIAN----------*/

// Static constants, Outside definition to avoid linker error

template <int N>
const MANIFOLD_TYPE Manifold < EUCLIDIAN, N>::MyType = EUCLIDIAN; 

template <int N>
const int Manifold < EUCLIDIAN, N>::manifold_dim = N; 

template <int N>
const int Manifold < EUCLIDIAN, N>::value_dim = N; 



// Squared Euclidian distance function
template <int N>
inline typename Manifold < EUCLIDIAN, N>::dist_type Manifold < EUCLIDIAN, N>::dist_squared( cref_type x, cref_type y ){
    //value_type v = x-y;
    return (x-y).squaredNorm();
}



// Derivative of Squared Euclidian distance w.r.t. first argument
template <int N>
inline void Manifold < EUCLIDIAN, N>::deriv1x_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    result =  2 * (x-y); 
}
// Derivative of Squared Euclidian distance w.r.t. second argument
template <int N>
inline void Manifold < EUCLIDIAN, N>::deriv1y_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    result = 2 * (y-x); 
}




// Second Derivative of Squared Euclidian distance w.r.t first argument
template <int N>
inline void Manifold < EUCLIDIAN, N>::deriv2xx_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    result = 2 * deriv2_type::Identity();
}
// Second Derivative of Squared Euclidian distance w.r.t first and second argument
template <int N>
inline void Manifold < EUCLIDIAN, N>::deriv2xy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    result = -2 * deriv2_type::Identity();
}
// Second Derivative of Squared Euclidian distance w.r.t second argument
template <int N>
inline void Manifold < EUCLIDIAN, N>::deriv2yy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    result = 2 * deriv2_type::Identity();
}



// Exponential and Logarithm Map
template <int N>
template <typename DerivedX, typename DerivedY>
inline void Manifold <EUCLIDIAN, N>::exp(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y, Eigen::MatrixBase<DerivedX>& result){
    result=x+y;
}

template <int N>
inline void Manifold <EUCLIDIAN, N>::log(cref_type x, cref_type y, ref_type result){
    result = y-x;
}

// Tangent Plane restriction
template <int N>
inline void Manifold <EUCLIDIAN, N>::tangent_plane_base(cref_type x, tm_base_ref_type result){
    result = tm_base_type::Identity();
}

// Projector (not necessary for Euclidian)
template <int N>
inline void Manifold <EUCLIDIAN, N>::projector(ref_type x){
}

} // end namespace tvmtl








#endif
