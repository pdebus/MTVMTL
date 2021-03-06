#ifndef TVMTL_MANIFOLD_SPHERE_HPP
#define TVMTL_MANIFOLD_SPHERE_HPP

#include <cmath>
#include <complex>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "enumerators.hpp"

namespace tvmtl {

// Specialization SPHERE
template < int N>
struct Manifold< SPHERE, N> {
    
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
	typedef Eigen::Matrix< scalar_type, N, 1>				value_type;
	typedef value_type&							ref_type;
	typedef const value_type&						cref_type;
	typedef std::vector<value_type, Eigen::aligned_allocator<value_type> >	value_list; 
	
	// Tangent space typedefs
	typedef Eigen::Matrix < scalar_type, N, N-1> tm_base_type;
	typedef tm_base_type& tm_base_ref_type;

	// Derivative Typedefs
	typedef value_type			     deriv1_type;
	typedef deriv1_type&			     deriv1_ref_type;
	
	typedef Eigen::Matrix<scalar_type, N, N>     deriv2_type;
	typedef deriv2_type&			     deriv2_ref_type;
	typedef	Eigen::Matrix<scalar_type, N-1, N-1> restricted_deriv2_type;


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


/*-----IMPLEMENTATION SPHERE----------*/

// Static constants, Outside definition to avoid linker error

template <int N>
const MANIFOLD_TYPE Manifold < SPHERE, N>::MyType = SPHERE; 

template <int N>
const int Manifold < SPHERE, N>::manifold_dim = N-1; 

template <int N>
const int Manifold < SPHERE, N>::value_dim = N; 

template <int N>
const bool Manifold < SPHERE, N>::non_isometric_embedding = false; 


// Squared Sphere distance function
template <int N>
inline typename Manifold < SPHERE, N>::dist_type Manifold < SPHERE, N>::dist_squared( cref_type x, cref_type y ){
    scalar_type xdoty = x.dot(y);
    if (xdoty < - 1.0) xdoty = -1.0;
    if (xdoty >  1.0) xdoty = 1.0;
    dist_type d = std::acos(xdoty);
    #ifdef TVMTL_MANIFOLD_DEBUG
	    if(!std::isfinite(d)){
	    std::cout << "\nDX Non-Series: " << std::endl;
	    std::cout << "x "<< x  << std::endl;
	    std::cout << "y "<< y << std::endl;
	    std::cout << "x^Ty "<< x.dot(y) << std::endl;
	    std::cout << "xdoty "<< xdoty << std::endl;
	    std::cout << "result" << d << std::endl;
	    }
	#endif
    return d*d;
}


// Derivative of Squared Sphere distance w.r.t. first argument
// TODO: Extende implementation of series to 1.0-eps
template <int N>
inline void Manifold < SPHERE, N>::deriv1x_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    scalar_type xdoty = x.dot(y);
    bool useSeries = false;

    if (xdoty < - 1.0) xdoty = -1.0;
    if (xdoty >= 1.0) { xdoty = 1.0; useSeries = true; }
    
    if(!useSeries){
	scalar_type acos = std::acos(xdoty);
	result =  -2.0 * acos / std::sqrt(1.0 - xdoty * xdoty) * y;
	#ifdef TVMTL_MANIFOLD_DEBUG
	    if(!std::isfinite(result(0))){
	    std::cout << "\nDX Non-Series: " << std::endl;
	    std::cout << "x "<< x  << std::endl;
	    std::cout << "y "<< y << std::endl;
	    std::cout << "x^Ty "<< x.dot(y) << std::endl;
	    std::cout << "xdoty "<< xdoty << std::endl;
	    std::cout << "acos(x) " << acos << std::endl;
	    std::cout << "sqrt(1-x^2)" << std::sqrt(1.0 - xdoty * xdoty) << std::endl;
	    std::cout << "result" << result << std::endl;
	    }
	#endif
    }
    else{
	result = -2.0 * y;
	#ifdef TVMTL_MANIFOLD_DEBUG
	    if(!std::isfinite(result(0)))
		std::cout << "DX Series:" << result << std::endl;
	#endif
    }
}
// Derivative of Squared Sphere distance w.r.t. second argument
template <int N>
inline void Manifold < SPHERE, N>::deriv1y_dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    scalar_type xdoty = x.dot(y);
    bool useSeries = false;

    if (xdoty < - 1.0) xdoty = -1.0;
    if (xdoty >=  1.0) { xdoty = 1.0; useSeries = true; }
    
    if(!useSeries){
	scalar_type acos = std::acos(xdoty);
	result =  -2.0 * acos / std::sqrt(1.0 - xdoty * xdoty) * x;
    }
    else
	result = -2.0 * x;
}




// Second Derivative of Squared Sphere distance w.r.t first argument
template <int N>
inline void Manifold < SPHERE, N>::deriv2xx_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    scalar_type xdoty = x.dot(y);
    bool useSeries = false;

    if (xdoty < - 1.0) xdoty = -1.0;
    if (xdoty >=  1.0) { xdoty = 1.0; useSeries = true; }
    
    if(!useSeries){
	scalar_type onemx2 = 1.0 - xdoty * xdoty;
	scalar_type acos = std::acos(xdoty);
	scalar_type da =  -2.0 * acos / std::sqrt(onemx2);
	result = (2.0 + da * xdoty) / onemx2 * y * y.transpose() - da * xdoty * deriv2_type::Identity();
    }
    else
	result = 2.0/3.0 * y * y.transpose() + 2.0 * xdoty * deriv2_type::Identity();

}
// Second Derivative of Squared Sphere distance w.r.t first and second argument
template <int N>
inline void Manifold < SPHERE, N>::deriv2xy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    scalar_type xdoty = x.dot(y);
    bool useSeries = false;

    if (xdoty < - 1.0) xdoty = -1.0;
    if (xdoty >=  1.0) { xdoty = 1.0; useSeries = true; }
    
    if(!useSeries){
	scalar_type onemx2 = 1.0 - xdoty * xdoty;
	scalar_type acos = std::acos(xdoty);
	scalar_type da =  -2.0 * acos / std::sqrt(onemx2);
	result = (2.0 + da * xdoty) / onemx2 * y * x.transpose() + da * deriv2_type::Identity(); 
    }
    else
	result = 2.0/3.0 * y * x.transpose() - 2.0 * deriv2_type::Identity();
}
// Second Derivative of Squared Sphere distance w.r.t second argument
template <int N>
inline void Manifold < SPHERE, N>::deriv2yy_dist_squared( cref_type x, cref_type y, deriv2_ref_type result){
    scalar_type xdoty = x.dot(y);
    bool useSeries = false;

    if (xdoty < - 1.0) xdoty = -1.0;
    if (xdoty >=  1.0) { xdoty = 1.0; useSeries = true; }
    
    if(!useSeries){
	scalar_type onemx2 = 1.0 - xdoty * xdoty;
	scalar_type acos = std::acos(xdoty);
	scalar_type da =  -2.0 * acos / std::sqrt(onemx2);
	result = (2.0 + da * xdoty) / onemx2 * x * x.transpose() - da * xdoty * deriv2_type::Identity();

    }
    else
	result = 2.0/3.0 * x * x.transpose() + 2.0 * xdoty * deriv2_type::Identity();

}



// Exponential and Logarithm Map
template <int N>
template <typename DerivedX, typename DerivedY>
inline void Manifold <SPHERE, N>::exp(const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y, Eigen::MatrixBase<DerivedX>& result){
    //result=(x+y).normalized();
    scalar_type n = y.norm();
    if(n!=0)
	result = std::cos(n) * x + std::sin(n) * y / n;
    else
	result = x;
}

template <int N>
inline void Manifold <SPHERE, N>::log(cref_type x, cref_type y, ref_type result){
    //result = (y-x).normalized();
    scalar_type xdoty = x.dot(y);
    bool useSeries = false;

    if (xdoty < - 1.0) xdoty = -1.0;
    if (xdoty >=  1.0) { xdoty = 1.0; useSeries = true; }
    
    if(!useSeries){
	scalar_type fac = std::acos(xdoty) / std::sqrt(1.0 - xdoty * xdoty);
	result = fac * (y - xdoty * x);
    }
    else
	result =  y - xdoty * x;
}

// Tangent Plane restriction
// TODO: Implement general QR Composition here
template <int N>
inline void Manifold <SPHERE, N>::tangent_plane_base(cref_type x, tm_base_ref_type result){
    result = tm_base_type::Identity();
}

template <> // Special Version for S^2 utilizing the cross product
inline void Manifold <SPHERE, 3>::tangent_plane_base(cref_type x, tm_base_ref_type result){
    //int c = static_cast<int>(std::abs(x.coeff(0)) > 0.5);
    int c = static_cast<int>(std::abs(x.coeff(2)) > 0.5 || std::abs(x.coeff(1)) > 0.5);
    result.col(0) = value_type(0, x.coeff(2), -x.coeff(1)) * c + value_type(x.coeff(2), 0, -x.coeff(0)) * (1-c);
    result.col(0).normalize();
    result.col(1) = x.cross(result.col(0)).normalized();

    if(x.norm()==0)
	result = tm_base_type::Zero();

    #ifdef TVMTL_MANIFOLD_DEBUG
	    if(!std::isfinite(result(0,0))){
	    std::cout << "\n\nx "<< x  << std::endl;
	    std::cout << "col0 V1 "<<  value_type(0, x.coeff(2), -x.coeff(1)) << std::endl;
	    std::cout << "col0 V2 "<<  value_type(x.coeff(2), 0, -x.coeff(0)) << std::endl;
	    std::cout << "col0 norm " << result.col(0) << std::endl;
	    std::cout << "cross prod" << x.cross(result.col(0)) << std::endl;
	    std::cout << "cross prod norm" << result.col(1) << std::endl;
	    }
    #endif
}

// Convex geodesic combinations
template <int N>
inline void Manifold <SPHERE, N>::convex_combination(cref_type x, cref_type y, double t, ref_type result){
   if(t == 0.5){
	result = x + y;
	projector(result);
   }
   else{
	value_type l;
	log(x, y, l);
	exp(x, l * t, result);
   }
}

// Karcher mean implementations
template <int N>
inline void Manifold<SPHERE, N>::karcher_mean(ref_type x, const value_list& v, double tol, int maxit){
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

template <int N>
inline void Manifold<SPHERE, N>::weighted_karcher_mean(ref_type x, const weight_list& w, const value_list& v, double tol, int maxit){
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

template <int N>
template <typename V, class... Args>
inline void Manifold<SPHERE, N>::karcher_mean(V& x, const Args&... args){
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

template <int N>
template <typename V>
inline void Manifold<SPHERE, N>::variadic_karcher_mean_gradient(V& x, const V& y){
    V temp;
    log(x, y, temp);
    x = temp;
}

template <int N>
template <typename V, class... Args>
inline void Manifold<SPHERE, N>::variadic_karcher_mean_gradient(V& x, const V& y1, const Args& ... args){
    V temp1, temp2;
    temp2 = x;
    
    log(x, y1, temp1);

    variadic_karcher_mean_gradient(temp2, args...);
    temp1 += temp2;
    x = temp1;
}

template <int N>
inline void Manifold <SPHERE, N>::projector(ref_type x){
    
    #ifdef TVMTL_MANIFOLD_DEBUG
	if(!std::isfinite(x(0))) std::cout << "Projector recieved nan" << std::endl;
    #endif

    scalar_type norm = x.norm();
    if(norm!=0.0) x.normalize();
    else x.setConstant(1.0 / 256.0).normalize();

    #ifdef TVMTL_MANIFOLD_DEBUG
	if(!std::isfinite(x(0))) std::cout << "Projector returns nan" << std::endl;
    #endif
}



} // end namespace tvmtl








#endif
