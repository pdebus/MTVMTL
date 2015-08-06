#ifndef TVMTL_MANIFOLD_GRASSMANN_HPP
#define TVMTL_MANIFOLD_GRASSMANN_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <functional>

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
	typedef std::vector<double> weight_list; 

	// Value Typedef
	typedef Eigen::Matrix< scalar_type, N, P>				value_type;
	typedef value_type&							ref_type;
	typedef const value_type&						cref_type;
	typedef std::vector<value_type, Eigen::aligned_allocator<value_type> >	value_list; 
	
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
	
	// Projections
	inline static void horizontal_space_projector(cref_type x, ref_type a);
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
    
    Eigen::Matrix<int, N * P, 1> indices;
    indices.setZero();

    int inc = 0;
    for(int i=1; i< indices.size(); ++i){
	if(i % 2 == 0)
	    inc = -P;
	else
	    inc = N;
	indices(i) = indices(i-1) + inc;
    }
    
    perm_type Perm(indices);
    return Perm.transpose();
}

template <int N, int P>
const typename Manifold < GRASSMANN, N, P>::perm_type Manifold<GRASSMANN, N, P>::permutation_matrix = ConstructPermutationMatrix(); 


// Squared GRASSMANN distance function
template <int N, int P>
inline typename Manifold <GRASSMANN, N, P>::dist_type Manifold <GRASSMANN, N, P>::dist_squared( cref_type x, cref_type y ){
    
    // Geodesic distance
//    Eigen::JacobiSVD<Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> > svd(x.transpose() * y, Eigen::ComputeThinU | Eigen::ComputeThinV);
//    return svd.singularValues().array().acos().matrix().squaredNorm();

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
    deriv2_type XtXId, IdXXtmYYt, XtXP;
    XtXId = Eigen::kroneckerProduct(x.transpose() * x, Eigen::Matrix<scalar_type, N, N>::Identity());
    IdXXtmYYt = Eigen::kroneckerProduct(Eigen::Matrix<scalar_type, P, P>::Identity(), x * x.transpose() - y * y.transpose());
    XtXP = Eigen::kroneckerProduct(x.transpose(), x) * permutation_matrix;
    result = 2.0 * (XtXId + XtXP + IdXXtmYYt);
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
    Eigen::JacobiSVD<Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> > svd(y, Eigen::ComputeThinU | Eigen::ComputeThinV);
    value_type temp_result = x * svd.matrixV() * svd.singularValues().array().cos().matrix().asDiagonal() * svd.matrixV().transpose() + svd.matrixU() * svd.singularValues().array().sin().matrix().asDiagonal() * svd.matrixV().transpose();
    // Reorthonormalization
    Eigen::HouseholderQR<value_type> qr(temp_result);
    result = qr.householderQ() * value_type::Identity();
}

template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::log(cref_type x, cref_type y, ref_type result){
    Eigen::Matrix<scalar_type, P, P> YtX = y.transpose() * x;
    Eigen::Matrix<scalar_type, P, N> At = y.transpose() - YtX * x.transpose();

    value_type B = YtX.householderQr().solve(At).transpose();

    Eigen::JacobiSVD<Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> > svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    #ifdef TVMTL_MANIFOLD_DEBUG_GRASSMANN
        std::cout << "\n\nB\n" << B << std::endl;
	std::cout << "U\n" << svd.matrixU() << std::endl;
	std::cout << "S\n" << svd.singularValues() << std::endl;
	std::cout << "Vt\n" << svd.matrixV().transpose() << std::endl;
    #endif

    result = svd.matrixU() * svd.singularValues().unaryExpr(std::function<scalar_type(scalar_type)>((scalar_type(*)(scalar_type))&std::atan)).asDiagonal() * svd.matrixV().transpose();
}

// Tangent Plane restriction
template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::tangent_plane_base(cref_type x, tm_base_ref_type result){
    //value_type Hproj = (Eigen::Matrix<scalar_type, N, N>::Identity() - x * x.transpose()) * x; // x - xx^tx = x - x = 0 TODO: Check of that makes sense
    //Eigen::JacobiSVD<value_type> svd(Hproj);
    
    // Compute X_orth by SVD 
    /*
    Eigen::JacobiSVD<value_type> svd(x);
    Eigen::Matrix<scalar_type, N, N - P> xorth = svd.matrixU().rightCols(N-P);
    */

    //Compute X_orth by QR
    Eigen::HouseholderQR<value_type> qr(x);
    Eigen::Matrix<scalar_type, N, N> Q = qr.householderQ();
    Eigen::Matrix<scalar_type, N, N - P> xorth = Q.rightCols(N-P);

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
inline void Manifold <GRASSMANN, N, P>::horizontal_space_projector(cref_type x, ref_type a){
	    a = a - x * x.transpose() * a;
}

// Convex geodesic combinations
template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::convex_combination(cref_type x, cref_type y, double t, ref_type result){
    value_type l;
    log(x, y, l);
    exp(x, l * t, result);
}

// Karcher mean implementations
template <int N, int P>
inline void Manifold<GRASSMANN, N, P>::karcher_mean(ref_type x, const value_list& v, double tol, int maxit){
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
	exp(x, 1.0 / v.size() * L, temp);
	x = temp;
	error = std::abs(x.sum() - m1);
	++k;
    } while(error > tol && k < maxit);

}

template <int N, int P>
inline void Manifold<GRASSMANN, N, P>::weighted_karcher_mean(ref_type x, const weight_list& w, const value_list& v, double tol, int maxit){
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
	exp(x, 1.0 / v.size() * L, temp);
	x = temp;
	error = std::abs(x.sum() - m1);
	++k;
    } while(error > tol && k < maxit);

}

template <int N, int P>
template <typename V, class... Args>
inline void Manifold<GRASSMANN, N, P>::karcher_mean(V& x, const Args&... args){
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
	error = std::abs(x.sum() - m1);
	++k;
    } while(error > tol && k < maxit);
}

template <int N, int P>
template <typename V>
inline void Manifold<GRASSMANN, N, P>::variadic_karcher_mean_gradient(V& x, const V& y){
    V temp;
    log(x, y, temp);
    x = temp;
}

template <int N, int P>
template <typename V, class... Args>
inline void Manifold<GRASSMANN, N, P>::variadic_karcher_mean_gradient(V& x, const V& y1, const Args& ... args){
    V temp1, temp2;
    temp2 = x;
    
    log(x, y1, temp1);

    variadic_karcher_mean_gradient(temp2, args...);
    temp1 += temp2;
    x = temp1;
}


template <int N, int P>
inline void Manifold <GRASSMANN, N, P>::projector(ref_type x){
	    Eigen::HouseholderQR<value_type> qr(x);
	    value_type thinQ = qr.householderQ() * value_type::Identity();
	    x = thinQ;
}


} // end namespace tvmtl








#endif
