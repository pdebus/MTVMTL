#ifndef TVMTL_MANIFOLD_SON_HPP
#define TVMTL_MANIFOLD_SON_HPP

#include <cmath>
#include <complex>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>

#include "enumerators.hpp"
#include "matrix_utils.hpp"

namespace tvmtl {

// Specialization SO(N)
template <int N>
struct Manifold< SO, N> {
    
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
	typedef std::vector<double> weight_list; 

	// Value Typedef
	typedef Eigen::Matrix< scalar_type, N, N>				value_type;
	typedef value_type&							ref_type;
	typedef const value_type&						cref_type;
	typedef std::vector<value_type, Eigen::aligned_allocator<value_type> >	value_list; 

	
	// Tangent space typedefs
	typedef Eigen::Matrix <scalar_type, N * N, N * (N - 1) / 2> tm_base_type;
	typedef tm_base_type& tm_base_ref_type;

	// Derivative Typedefs
	typedef value_type			     deriv1_type;
	typedef deriv1_type&			     deriv1_ref_type;
	
	typedef Eigen::Matrix<scalar_type, N*N, N*N>				deriv2_type;
	typedef deriv2_type&							deriv2_ref_type;
	typedef	Eigen::Matrix<scalar_type, N * (N - 1) / 2, N * (N - 1) / 2>	restricted_deriv2_type;

	// Helper Types
	typedef Eigen::PermutationMatrix<N*N, N*N, int> perm_type;

	inline static perm_type ConstructPermutationMatrix();

	// Manifold distance functions (for IRLS)
	inline static dist_type dist_squared(cref_type x, cref_type y);
	inline static void deriv1x_dist_squared(cref_type x, cref_type y, deriv1_ref_type result);
	inline static void deriv1y_dist_squared(cref_type x, cref_type y, deriv1_ref_type result);

	inline static void deriv2xx_dist_squared(cref_type x, cref_type y, deriv2_ref_type result);
	inline static void deriv2xy_dist_squared(cref_type x, cref_type y, deriv2_ref_type result);
	inline static void deriv2yy_dist_squared(cref_type x, cref_type y, deriv2_ref_type result);
	static const perm_type permutation_matrix;


	// Manifold exponentials und logarithms ( for Proximal point)
	template <typename DerivedX, typename DerivedY>
	inline static void exp(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y, Eigen::MatrixBase<DerivedX>& result);
	inline static void log(cref_type x, cref_type y, ref_type result);
	
	inline static void convex_combination(cref_type x, cref_type y, double t, ref_type result);

	// Implementations of the Karcher mean
	// Slow list version
	inline static void karcher_mean(ref_type x, const value_list& v, double tol=1e-10, int maxit=15);
	inline static void weighted_karcher_mean(ref_type x, const weight_list& w, const value_list& v, double tol=1e-10, int maxit=15);
	// Variadic templated version
	template <typename V, class... Args>
	inline static void karcher_mean(V& x, const Args&... args);
	template <typename V>
	inline static void variadic_karcher_mean_gradient(V& x, const V& y);
	template <typename V, class... Args>
	inline static void variadic_karcher_mean_gradient(V& x, const V& y1, const Args&... args);

	// Basis transformation for restriction to tangent space
	inline static void tangent_plane_base(cref_type x, tm_base_ref_type result);

	// Projection
	inline static void projector(ref_type x);

	// Interpolation pre- and postprocessing
	inline static void interpolation_preprocessing(ref_type x) {};
	inline static void interpolation_postprocessing(ref_type x) {};


};


/*-----IMPLEMENTATION SO----------*/

// Static constants, Outside definition to avoid linker error

template <int N>
const MANIFOLD_TYPE Manifold < SO, N>::MyType = SO; 

template <int N>
const int Manifold < SO, N>::manifold_dim = N * (N - 1) / 2; 

template <int N>
const int Manifold < SO, N>::value_dim = N * N; 

template <int N>
const bool Manifold < SO, N>::non_isometric_embedding = false; 

// PermutationMatrix
template <int N>
typename Manifold < SO, N>::perm_type Manifold<SO, N>::ConstructPermutationMatrix(){
    perm_type P;
    P.setIdentity();
    for(int i=0; i<N; i++)
	for(int j=0; j<i; j++)
	    P.applyTranspositionOnTheRight(j*N + i, i*N + j);
    return P;
}

template <int N>
const typename Manifold < SO, N>::perm_type Manifold<SO, N>::permutation_matrix = ConstructPermutationMatrix(); 




// Squared SO distance function
template <int N>
inline typename Manifold < SO, N>::dist_type Manifold < SO, N>::dist_squared( cref_type x, cref_type y ){
    #ifdef TV_SON_DEBUG
	std::cout << "\nDist2 function with x=\n" << x << "\nand y=\n" << y << std::endl;
    #endif 
    return (x.transpose() * y).log().squaredNorm();
}


// Derivative of Squared SO distance w.r.t. first argument
template <int N>
inline void Manifold < SO, N>::deriv1x_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    result = -2.0 * x * (x.transpose() * y).log();
}
// Derivative of Squared SO distance w.r.t. second argument
template <int N>
inline void Manifold < SO, N>::deriv1y_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    result =  -2.0 * y * (y.transpose() * x).log();
}




// Second Derivative of Squared SO distance w.r.t first argument
template <int N>
inline void Manifold < SO, N>::deriv2xx_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    value_type XtY = x.transpose()*y;

    deriv2_type logXtY_kron_I ,I_kron_x, Yt_kron_I, dlog; 
    logXtY_kron_I = Eigen::kroneckerProduct(XtY.log().eval(),value_type::Identity());
    I_kron_x = Eigen::kroneckerProduct(value_type::Identity(),x);
    Yt_kron_I = Eigen::kroneckerProduct(y.transpose(),value_type::Identity());
    KroneckerDLog(XtY, dlog);

    result = -2.0 * (logXtY_kron_I.transpose() + I_kron_x * dlog * Yt_kron_I * permutation_matrix );
}
// Second Derivative of Squared SO distance w.r.t first and second argument
template <int N>
inline void Manifold < SO, N>::deriv2xy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    value_type XtY = x.transpose()*y;

    deriv2_type I_kron_x, dlog; 
    I_kron_x = Eigen::kroneckerProduct(value_type::Identity(),x);
    KroneckerDLog(XtY, dlog);

    result = -2.0 * I_kron_x * dlog * I_kron_x.transpose();
}
// Second Derivative of Squared SO distance w.r.t second argument
template <int N>
inline void Manifold < SO, N>::deriv2yy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    deriv2xx_dist_squared(y, x, result);
}



// Exponential and Logarithm Map
template <int N>
template <typename DerivedX, typename DerivedY>
inline void Manifold <SO, N>::exp(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y, Eigen::MatrixBase<DerivedX>& result){
    result = x * (x.transpose() * y).exp();
}

template <int N>
inline void Manifold <SO, N>::log(cref_type x, cref_type y, ref_type result){
    result = x * (x.transpose() * y).log();
}

// Tangent Plane restriction
template <int N>
inline void Manifold <SO, N>::tangent_plane_base(cref_type x, tm_base_ref_type result){
    int d = value_type::RowsAtCompileTime;
    int k = 0;
    
    value_type T;

    for(int i=0; i<d-1; i++)
	for(int j=i+1; j<d; j++){
	    T.setZero();
	    scalar_type sqrt = 1.0/std::sqrt(2);
	    T.col(i) = -sqrt * x.col(j);
	    T.col(j) =  sqrt * x.col(i);

	    result.col(k) = Eigen::Map<Eigen::VectorXd>(T.data(), T.size());
	    k++;
	}
}

// Convex geodesic combinations
template <int N>
inline void Manifold <SO, N>::convex_combination(cref_type x, cref_type y, double t, ref_type result){
    value_type l;
    if (t == 0.5){
	result = x + y;
	projector(result);
    }
    else{
    log(x, y, l);
    exp(x, l * t, result);
    }
}

// Karcher mean implementations
template <int N>
inline void Manifold<SO, N>::karcher_mean(ref_type x, const value_list& v, double tol, int maxit){
    value_type L, temp;
   
    int k = 0;
    double error = 0.0;
    do{
	scalar_type m1 = x.sum();
	L = value_type::Zero();
	for(int i = 0; i < v.size(); ++i){
	    log(x, v[i], temp);
	    L += temp;
	}
	exp(x, 1.0 / v.size() * L , temp);
	x = temp;
	error = std::abs(x.sum() - m1);
	++k;
    } while(error > tol && k < maxit);

}

template <int N>
inline void Manifold<SO, N>::weighted_karcher_mean(ref_type x, const weight_list& w, const value_list& v, double tol, int maxit){
    value_type L, temp;
   
    int k = 0;
    double error = 0.0;
    do{
	scalar_type m1 = x.sum();
	L = value_type::Zero();
	for(int i = 0; i < v.size(); ++i){
	    log(x, v[i], temp);
	    L += w[i] * temp;
	}
	exp(x, 1.0 / v.size() * L , temp);
	x = temp;
	projector(x);
	error = std::abs(x.sum() - m1);
	++k;
    } while(error > tol && k < maxit);

}

template <int N>
template <typename V, class... Args>
inline void Manifold<SO, N>::karcher_mean(V& x, const Args&... args){
    V temp, sum;
    
    int numArgs = sizeof...(args);
    int k = 0;
    double error = 0.0;    
    double tol = 1e-10;
    int maxit = 15;
    do{
	scalar_type m1 = x.sum();
	sum = x;
	variadic_karcher_mean_gradient(sum, args...);
	exp(x, 1.0 / numArgs * sum, temp);
	x = temp;
	projector(x);
	error = std::abs(x.sum() - m1);
	++k;
    } while(error > tol && k < maxit);
}

template <int N>
template <typename V>
inline void Manifold<SO, N>::variadic_karcher_mean_gradient(V& x, const V& y){
    V temp;
    log(x, y, temp);
    x = temp;
}

template <int N>
template <typename V, class... Args>
inline void Manifold<SO, N>::variadic_karcher_mean_gradient(V& x, const V& y1, const Args& ... args){
    V temp1, temp2;
    temp2 = x;
    
    log(x, y1, temp1);

    variadic_karcher_mean_gradient(temp2, args...);
    temp1 += temp2;
    x = temp1;
}

template <int N>
inline void Manifold <SO, N>::projector(ref_type x){
    
    #ifdef TV_SON_DEBUG
	std::cout << "\n\nProjector with initial x=\n" << x << std::endl;
    #endif     
    
    Eigen::JacobiSVD<value_type> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
    x = svd.matrixU() * svd.matrixV().transpose();
    //if(x.determinant() < 0)
    //	x.row(1)*=-1.0;

    #ifdef TV_SON_DEBUG
	std::cout << "\nProjector with final x=\n" << x << std::endl;
    #endif
}



} // end namespace tvmtl








#endif
