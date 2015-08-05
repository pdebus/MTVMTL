#ifndef TVMTL_MANIFOLD_SPD_HPP
#define TVMTL_MANIFOLD_SPD_HPP

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

// Specialization SPD
template <int N>
struct Manifold< SPD, N> {
    
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
	typedef Eigen::Matrix< scalar_type, N, N>				value_type;
	typedef value_type&							ref_type;
	typedef const value_type&						cref_type;
	typedef std::vector<value_type, Eigen::aligned_allocator<value_type> >	value_list; 


	// Tangent space typedefs
	typedef Eigen::Matrix <scalar_type, N * N, N * (N + 1) / 2>   tm_base_type;
	typedef tm_base_type&					    tm_base_ref_type;

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

	inline static void convex_combination(cref_type x, cref_type y, double t, ref_type result);

	// Implementation of the Karcher mean
	// Slow list version
	inline static void karcher_mean_gradient(ref_type x, const value_list& v);
	// Variadic templated version
	template <typename V, class... Args>
	inline static void karcher_mean_gradient(V& x, const Args&... args);
	template <typename V>
	inline static void variadic_karcher_mean_gradient(V& x, const V& y);
	template <typename V, class... Args>
	inline static void variadic_karcher_mean_gradient(V& x, const V& y1, const Args&... args);
	
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
    #ifdef TV_SPD_DIST_DEBUG
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
    T2T4tId = Eigen::kroneckerProduct(T2 * T4.transpose(), value_type::Identity());
    IdT2T4 = Eigen::kroneckerProduct(value_type::Identity(), T2 * T4);
    T2T2 = Eigen::kroneckerProduct(T2, T2);
    T2yId = Eigen::kroneckerProduct(T2 * y, value_type::Identity());
    IdT2y = Eigen::kroneckerProduct(value_type::Identity(), T2 * y);

    result =  2 * (T2T4tId + IdT2T4 + T2T2 * dlog * (T2yId + IdT2y) ) * T2T2 * dsqrt;
}
// Second Derivative of Squared SPD distance w.r.t first and second argument
template <int N>
inline void Manifold < SPD, N>::deriv2xy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    value_type isqrtX, T1;
    isqrtX = x.sqrt().eval().inverse();
    T1 = isqrtX * y * isqrtX;

    deriv2_type kp_isqrtX, dlog;
    kp_isqrtX = Eigen::kroneckerProduct(isqrtX, isqrtX);
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
    #ifdef TV_SPD_EXP_DEBUG
	std::cout << "\nEXP function with x=\n" << x << "\nand y=\n" << y << std::endl;
    #endif
    value_type sqrtX = x.sqrt();
    value_type Z = sqrtX.ldlt().solve(y).transpose();	
    result = sqrtX * sqrtX.transpose().ldlt().solve(Z).exp() * sqrtX;	
}

template <int N>
inline void Manifold <SPD, N>::log(cref_type x, cref_type y, ref_type result){
    #ifdef TV_SPD_LOG_DEBUG
	std::cout << "\nLOG function with x=\n" << x << "\nand y=\n" << y << std::endl;
    #endif
    value_type sqrtX = x.sqrt();
    value_type Z = sqrtX.ldlt().solve(y).transpose();	
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

// Convex geodesic combinations
template <int N>
inline void Manifold <SPD, N>::convex_combination(cref_type x, cref_type y, double t, ref_type result){
    value_type l;
    log(x, y, l);
    exp(x, l * t, result);
}

// Karcher mean implementations
template <int N>
inline void Manifold <SPD, N>::karcher_mean_gradient(ref_type x, const value_list& v){
    value_type L, temp;
    L = value_type::Zero();
    for(int i = 0; i < v.size(); ++i){
	log(x, v[i], temp);
	L += temp;
    }
    
    exp(x, 0.5 / v.size() * (L + L.transpose()), temp);
    x = temp;
}

template <int N>
template <typename V, class... Args>
inline void Manifold<SPD, N>::karcher_mean_gradient(V& x, const Args&... args){
    int numArgs = sizeof...(args);

    V temp, sum;
    sum = x;

    variadic_karcher_mean_gradient(sum, args...);
    exp(x, 0.5 / numArgs * (sum + sum.transpose()), temp);
    x = temp;
}

template <int N>
template <typename V>
inline void Manifold<SPD, N>::variadic_karcher_mean_gradient(V& x, const V& y){
    V temp;
    log(x, y, temp);
    x = temp;
}

template <int N>
template <typename V, class... Args>
inline void Manifold<SPD, N>::variadic_karcher_mean_gradient(V& x, const V& y1, const Args& ... args){
    V temp1, temp2;
    temp2 = x;
    
    log(x, y1, temp1);

    variadic_karcher_mean_gradient(temp2, args...);
    temp1 += temp2;
    x = temp1;
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
