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
    for(int i=0; i<3; i++)
	for(int j=0; j<i; j++)
	    P.applyTranspositionOnTheRight(j*3+i, i*3+j);
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
    result = x * (x.transpose() * y).exp();
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
