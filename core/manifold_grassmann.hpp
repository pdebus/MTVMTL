#ifndef TVMTL_MANIFOLD_GRASSMANN_HPP
#define TVMTL_MANIFOLD_GRASSMANN_HPP

#include <cmath>
#include <complex>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>

//own includes
#include "enumerators.hpp"
#include "matrix_utils.hpp"

namespace tvmtl {

// Specialization GRASSMANN
template <int N, int P>
struct Manifold< GRASSMANN, N, P> {
    
    public:
	static const MANIFOLD_TYPE MyType;
	static const int manifold_dim ;
	static const int value_dim; // TODO: maybe rename to embedding_dim 

	static const bool non_isometric_embedding;
	
	// Scalar type of manifold
	//typedef double scalar_type;
	typedef double scalar_type;
	typedef double dist_type;
	typedef std::complex<double> complex_type;

	// Value Typedef
	typedef Eigen::Matrix< scalar_type, N, P>   value_type;
	typedef value_type&			    ref_type;
	typedef const value_type&		    cref_type;
	
	// Tangent space typedefs
	typedef Eigen::Matrix <scalar_type, N*N, N * (N + 1) / 2> tm_base_type;
	typedef tm_base_type& tm_base_ref_type;

	// Derivative Typedefs
	typedef value_type			     deriv1_type;
	typedef deriv1_type&			     deriv1_ref_type;
	
	typedef Eigen::Matrix<scalar_type, N*N, N*N>				deriv2_type;
	typedef deriv2_type&							deriv2_ref_type;
	typedef	Eigen::Matrix<scalar_type, N * (N + 1) / 2, N * (N + 1) / 2>	restricted_deriv2_type;


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

	// Projection
	inline static void projector(ref_type x);

	// Interpolation pre- and postprocessing
	inline static void interpolation_preprocessing(ref_type x);
	inline static void interpolation_postprocessing(ref_type x);

};


/*-----IMPLEMENTATION SPD----------*/

// Static constants, Outside definition to avoid linker error

template <int N>
const MANIFOLD_TYPE Manifold < SPD, N>::MyType = SPD; 

template <int N>
const int Manifold < SPD, N>::manifold_dim = N * (N + 1) / 2; 

template <int N>
const int Manifold < SPD, N>::value_dim = N * N; 

template <int N>
const bool Manifold < SPD, N>::non_isometric_embedding = true; 


// Squared SPD distance function
template <int N>
inline typename Manifold < SPD, N>::dist_type Manifold < SPD, N>::dist_squared( cref_type x, cref_type y ){
    #ifdef TV_SPD_DEBUG
	std::cout << "\nDist2 function with x=\n" << x << "\nand y=\n" << y << std::endl;
    #endif
// NOTE: If x is not strictly spd, using LDLT completely halts the algorithm
/*    value_type sqrtX = x.sqrt();
    Eigen::LDLT<value_type> ldlt;
    ldlt.compute(sqrtX);

    value_type Z = ldlt.solve(y).transpose();	
    return ldlt.solve(Z).transpose().log().squaredNorm();	*/
    value_type invsqrt = x.sqrt().inverse();
    return (invsqrt * y * invsqrt).log().squaredNorm();
}


// Derivative of Squared SPD distance w.r.t. first argument
// TODO: Switch to solve() for N>4?
template <int N>
inline void Manifold < SPD, N>::deriv1x_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    value_type invsqrt = x.sqrt().inverse();
    result = -2.0 * invsqrt * (invsqrt * y * invsqrt).log() * invsqrt;
}
// Derivative of Squared SPD distance w.r.t. second argument
template <int N>
inline void Manifold < SPD, N>::deriv1y_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    deriv1x_dist_squared(y, x, result);
}


// Second Derivative of Squared SPD distance w.r.t first argument
template <int N>
inline void Manifold < SPD, N>::deriv2xx_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    value_type T2, T3, T4;
    T2 = x.sqrt().inverse();
    T3 = T2 * y * T2;
    T4 = T3.log();

    deriv2_type dlog, dsqrt;
    KroneckerDLog(T3, dlog);
    KroneckerDSqrt(x, dsqrt);

    deriv2_type T2T4tId, IdT2T4, T2T2, T2yId, IdT2y;
    Eigen::kroneckerProduct(T2 * T4.transpose(), value_type::Identity(), T2T4tId);
    Eigen::kroneckerProduct(value_type::Identity(), T2 * T4, IdT2T4);
    Eigen::kroneckerProduct(T2, T2, T2T2);
    Eigen::kroneckerProduct(T2 * y, value_type::Identity(), T2yId);
    Eigen::kroneckerProduct( value_type::Identity(), T2 * y, IdT2y);

    result =  2 * (T2T4tId + IdT2T4 + T2T2 * dlog * (T2yId + IdT2y) ) * T2T2 * dsqrt;
}
// Second Derivative of Squared SPD distance w.r.t first and second argument
template <int N>
inline void Manifold < SPD, N>::deriv2xy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    value_type isqrtX, T1;
    isqrtX = x.sqrt().eval().inverse();
    T1 = isqrtX * y * isqrtX;

    deriv2_type kp_isqrtX, dlog;
    Eigen::kroneckerProduct(isqrtX, isqrtX, kp_isqrtX);
    KroneckerDLog(T1, dlog);

    result = -2 * kp_isqrtX * dlog * kp_isqrtX;
}
// Second Derivative of Squared SPD distance w.r.t second argument
template <int N>
inline void Manifold < SPD, N>::deriv2yy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    deriv2xx_dist_squared(y, x, result);
}



// Exponential and Logarithm Map
template <int N>
template <typename DerivedX, typename DerivedY>
inline void Manifold <SPD, N>::exp(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y, Eigen::MatrixBase<DerivedX>& result){
    value_type sqrtX = x.sqrt();
    value_type Z = sqrtX.ldlt().solve(y).transpose();	
    result = sqrtX * sqrtX.transpose().ldlt().solve(Z).exp() * sqrtX;	
}

template <int N>
inline void Manifold <SPD, N>::log(cref_type x, cref_type y, ref_type result){
    value_type sqrtX = x.sqrt();
    value_type Z = sqrtX.solve(y).transpose();	
    result = sqrtX * sqrtX.transpose().ldlt().solve(Z).log() * sqrtX;	
}

// Tangent Plane restriction
template <int N>
inline void Manifold <SPD, N>::tangent_plane_base(cref_type x, tm_base_ref_type result){
    int d = value_type::RowsAtCompileTime;
    int k = 0;
    
    value_type S, T;
    S = x.sqrt();

    for(int i=0; i<d; i++){
	T.setZero();
	T.col(i) = S.col(i);
	T = T * S;

	result.col(k) = Eigen::Map<Eigen::VectorXd>(T.data(), T.size());
	++k;
    }

    for(int i=0; i<d-1; i++)
	for(int j=i+1; j<d; j++){
	    T.setZero();
	    T.col(i) = S.col(j);
	    T.col(j) = S.col(i);
	    T = T * S;

	    result.col(k) = Eigen::Map<Eigen::VectorXd>(T.data(), T.size());
	    ++k;
	}
}


template <int N>
inline void Manifold <SPD, N>::projector(ref_type x){
    // does not exist since SPD is an open set
    // TODO: Eventually implement projection to semi positive definite matrices
}


template <int N>
inline void Manifold<SPD, N>::interpolation_preprocessing(ref_type x){
    value_type t = x.log();
    x = t;
}

template <int N>
inline void Manifold<SPD, N>::interpolation_postprocessing(ref_type x){
    value_type t = ( 0.5 * (x + x.transpose()) ).exp();
    x = t;
}


} // end namespace tvmtl








#endif
