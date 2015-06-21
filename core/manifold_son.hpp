#ifndef TVMTL_MANIFOLD_SON_HPP
#define TVMTL_MANIFOLD_SON_HPP

#include <cmath>
#include <complex>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>

#include "enumerators.hpp"

namespace tvmtl {

// Specialization SO(N)
template <int N>
struct Manifold< SO, N> {
    
    public:
	static const MANIFOLD_TYPE MyType;
	static const int manifold_dim ;
	static const int value_dim; // TODO: maybe rename to embedding_dim 

	// Scalar type of manifold
	//typedef double scalar_type;
	typedef double scalar_type;
	typedef double dist_type;
	typedef std::complex<double> complex_type;

	// Value Typedef
	typedef Eigen::Matrix< scalar_type, N, N>   value_type;
	typedef value_type&			    ref_type;
	typedef const value_type&		    cref_type;
	
	// Tangent space typedefs
	typedef Eigen::Matrix <scalar_type, N*N, N * (N - 1) / 2> tm_base_type;
	typedef tm_base_type& tm_base_ref_type;

	// Derivative Typedefs
	typedef value_type			     deriv1_type;
	typedef deriv1_type&			     deriv1_ref_type;
	
	typedef Eigen::Matrix<scalar_type, N*N, N*N>				deriv2_type;
	typedef deriv2_type&							deriv2_ref_type;
	typedef	Eigen::Matrix<scalar_type, N * (N - 1) / 2, N * (N - 1) / 2>	restricted_deriv2_type;


	// Manifold distance functions (for IRLS)
	inline static dist_type dist_squared(cref_type x, cref_type y);
	inline static void deriv1x_dist_squared(cref_type x, cref_type y, deriv1_ref_type result);
	inline static void deriv1y_dist_squared(cref_type x, cref_type y, deriv1_ref_type result);

	inline static void deriv2xx_dist_squared(cref_type x, cref_type y, deriv2_ref_type result);
	inline static void deriv2xy_dist_squared(cref_type x, cref_type y, deriv2_ref_type result);
	inline static void deriv2yy_dist_squared(cref_type x, cref_type y, deriv2_ref_type result);
	inline static const deriv2_type permutation_matrix;


	// Manifold exponentials und logarithms ( for Proximal point)
	template <typename DerivedX, typename DerivedY>
	inline static void exp(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y, Eigen::MatrixBase<DerivedX>& result);
	inline static void log(cref_type x, cref_type y, ref_type result);

	// Basis transformation for restriction to tangent space
	inline static void tangent_plane_base(cref_type x, tm_base_ref_type result);

	// Projection
	inline static void projector(ref_type x);


};


/*-----IMPLEMENTATION SO----------*/

// Static constants, Outside definition to avoid linker error

template <int N>
const MANIFOLD_TYPE Manifold < SO, N>::MyType = SO; 

template <int N>
const int Manifold < SO, N>::manifold_dim = N * (N - 1) / 2; 

template <int N>
const int Manifold < SO, N>::value_dim = N; 

template <int N>
const deriv2_type Manifold<SO, N>::permutation_matrix = deriv2_type::Identity(); // Change 

// Squared SO distance function
template <int N>
inline typename Manifold < SO, N>::dist_type Manifold < SO, N>::dist_squared( cref_type x, cref_type y ){
    return (x.transpose() * y).log().squaredNorm();
}


// Derivative of Squared SO distance w.r.t. first argument
template <int N>
inline void Manifold < SO, N>::deriv1x_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    result = -2 * x * (x.transpose() * y).log();
}
// Derivative of Squared SO distance w.r.t. second argument
template <int N>
inline void Manifold < SO, N>::deriv1y_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    result =  -2 * y * (y.transpose() * x).log();
}




// Second Derivative of Squared SO distance w.r.t first argument
template <int N>
inline void Manifold < SO, N>::deriv2xx_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    XtY = x.transpose()*y;
    deriv2_type dir = kroneckerProduct(y.transpose(),value_type::Identity());
    deriv2_type dlog = dlog(XtY, dir);
    result = -2.0 * (Eigen::kroneckerProduct(XtY.log(), value_type::Identity()) + kroneckerProduct(value_type::Identity(), x) * dlog * permutation_matrix );
}
// Second Derivative of Squared SO distance w.r.t first and second argument
template <int N>
inline void Manifold < SO, N>::deriv2xy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
   result = deriv2_type::Identity(); 
}
// Second Derivative of Squared SO distance w.r.t second argument
template <int N>
inline void Manifold < SO, N>::deriv2yy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
   result = deriv2_type::Identity(); 
}



// Exponential and Logarithm Map
template <int N>
template <typename DerivedX, typename DerivedY>
inline void Manifold <SO, N>::exp(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y, Eigen::MatrixBase<DerivedX>& result){
    result = x * (x.transpose() * y).exp();
}

template <int N>
inline void Manifold <SO, N>::log(cref_type x, cref_type y, ref_type result){
    result = x * (x.transpose() * y).exp();
}

// Tangent Plane restriction
template <int N>
inline void Manifold <SO, N>::tangent_plane_base(cref_type x, tm_base_ref_type result){
    result.setConstant(1.0); //TODO Placeholder! Change!
}


template <int N>
inline void Manifold <SO, N>::projector(ref_type x){
    Eigen::JacobiSVD<value_type> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
    x = svd.matrixU() * svd.matrixV().transpose();
}



} // end namespace tvmtl








#endif
