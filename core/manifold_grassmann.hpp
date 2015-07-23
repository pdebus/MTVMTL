#ifndef TVMTL_MANIFOLD_GRASSMANN_HPP
#define TVMTL_MANIFOLD_GRASSMANN_HPP

#include <cmath>
#include <complex>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/QR>
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
	typedef Eigen::Matrix <scalar_type, N * P, P * (N - P) > tm_base_type;
	typedef tm_base_type& tm_base_ref_type;

	// Derivative Typedefs
	typedef value_type			     deriv1_type;
	typedef deriv1_type&			     deriv1_ref_type;
	
	typedef Eigen::Matrix<scalar_type, N*P, N*P>				deriv2_type;
	typedef deriv2_type&							deriv2_ref_type;
	typedef	Eigen::Matrix<scalar_type, P * (N - P), P * (N - P) >		restricted_deriv2_type;

	// Helper Types
	typedef Eigen::PermutationMatrix<N * P, N * P, int> perm_type;

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


/*-----IMPLEMENTATION GRASSMANN----------*/

// Static constants, Outside definition to avoid linker error

template <int N, int P>
const MANIFOLD_TYPE Manifold <GRASSMANN, N, P>::MyType = GRASSMANN; 

template <int N, int P>
const int Manifold <GRASSMANN, N, P>::manifold_dim = (N - P) * P; 

template <int N, int P>
const int Manifold <GRASSMANN, N, P>::value_dim = N * P; 

template <int N, int P>
const bool Manifold <GRASSMANN, N, P>::non_isometric_embedding = false; 

// PermutationMatrix
template <int N, int P>
typename Manifold < GRASSMANN, N, P>::perm_type Manifold<GRASSMANN, N, P>::ConstructPermutationMatrix(){
    perm_type Perm;
    Perm.setIdentity();
    for(int i=0; i< N * P; i++)
	for(int j=0; j<i; j++)
	    Perm.applyTranspositionOnTheRight(j * N * P + i, i * N * P + j);
    return Perm;
}

template <int N, int P>
const typename Manifold < GRASSMANN, N, P>::perm_type Manifold<GRASSMANN, N, P>::permutation_matrix = ConstructPermutationMatrix(); 


// Squared GRASSMANN distance function
template <int N, int P>
inline typename Manifold <GRASSMANN, N, P>::dist_type Manifold <GRASSMANN, N, P>::dist_squared( cref_type x, cref_type y ){
    
    // Geodesic distance
    /*
    XtY = x.transpose()*Y;
    Eigen::JacobiSVD<value_type> svd(XtY, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd.singularValues().squaredNorm();
    */

    // Projection F-distance;
    return 0.5 * (x * x.transpose() - y * y.transpose()).squaredNorm();
}


// Derivative of Squared GRASSMANN distance w.r.t. first argument
template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::deriv1x_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    result = 2.0 * (x * x.transpose() - y * y.transpose()) * x;
}
// Derivative of Squared GRASSMANN distance w.r.t. second argument
template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::deriv1y_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    result = 2.0 * (y * y.transpose() - x * x.transpose()) * y;
}


// Second Derivative of Squared GRASSMANN distance w.r.t first argument
template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::deriv2xx_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    deriv2_type XtXId, IdXXt, XtXP, IdYYt;
    XtXId = Eigen::kroneckerProduct(x.transpose() * x, Eigen::Matrix<scalar_type, N, N>::Identity());
    IdXXt = Eigen::kroneckerProduct(Eigen::Matrix<scalar_type, P, P>::Identity(), x * x.transpose());
    XtXP = Eigen::kroneckerProduct(x.transpose(), x) * permutation_matrix;
    IdYYt = Eigen::kroneckerProduct(Eigen::Matrix<scalar_type, P, P>::Identity(), y * y.transpose());
    result = 2.0 * (XtXId + IdXXt + XtXId + IdYYt);
}
// Second Derivative of Squared GRASSMANN distance w.r.t first and second argument
template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::deriv2xy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    result = -2.0 * (Eigen::kroneckerProduct(x.transpose() * y, Eigen::Matrix<scalar_type, N, N>::Identity()) + Eigen::kroneckerProduct(x.transpose(), y) * permutation_matrix);
}
// Second Derivative of Squared GRASSMANN distance w.r.t second argument
template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::deriv2yy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    deriv2xx_dist_squared(y, x, result);
}



// Exponential and Logarithm Map
template <int N, int P>
template <typename DerivedX, typename DerivedY>
inline void Manifold <GRASSMANN, N, P>::exp(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y, Eigen::MatrixBase<DerivedX>& result){
    Eigen::JacobiSVD<value_type> svd(y, Eigen::ComputeThinU | Eigen::ComputeThinV);
    result = x * svd.matrixV() * svd.singularValues().array().cos().asDiagonal() * svd.matrixV().transpose() + svd.matrixU() * svd.singularValues().array().sin().asDiagonal() * svd.matrixV().transpose();
}

template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::log(cref_type x, cref_type y, ref_type result){
    value_type XtY = x.transpose() * y;
    Eigen::JacobiSVD<value_type> svd(XtY, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd D = svd.singularValues().array().acos() / svd.singularValues().array().asin();

    result = (-x * svd.matrixV() * svd.singularValues().asDiagonal() + y * svd.matrixV()) * D.asDiagonal() * svd.matrixU().transpose(); 
}

// Tangent Plane restriction
template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::tangent_plane_base(cref_type x, tm_base_ref_type result){
    value_type Hproj = (Eigen::Matrix<scalar_type, N, N>::Identity() - x * x.transpose()) * x;
    Eigen::JacobiSVD<value_type> svd(Hproj);
    Eigen::Matrix<scalar_type, N, N - P> xorth = svd.matrixU().rightCols(N-P);
    
    int k = 0;
    for(int r = 0; r < N - P; r++)
	for(int c = 0; c < P; c++ ){
	    Eigen::Matrix<scalar_type, N, P> T = Eigen::Matrix<scalar_type, N, P>::Zero();
	    T.col(c) = xorth.col(r);
	    result.col(k) = Eigen::Map<Eigen::VectorXd>(T.data(), T.size());
	    ++k;
	}

}


template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::projector(ref_type x){
    std::cout << "Projector with P=" << P << " and Argument\n" << x << std::endl;
	if(P > 1){
	    std::cout << "\nPerforming QR...\n";
	    Eigen::HouseholderQR<value_type> qr(x);
	    x = qr.householderQ();
	}
	else{
	    std::cout << "\nNormalizing...\n";
	    double norm = x.norm();
	    if(norm != 0)
		x = x / norm;
	}
}


} // end namespace tvmtl








#endif
