#ifndef TVMTL_MANIFOLD_HPP
#define TVMTL_MANIFOLD_HPP

namespace tvmtl {

// Primary Template
template < enum MANIFOLD_TYPE MF, int N, enum LA_HANDLER = EIGEN  >
struct Manifold { };


// Specialization EUCLIDIAN
template < int N, LA_HANDLER >
struct Manifold< EUCLIDIAN, N, LA_HANDLER > {
    
    public:
	static const MANIFOLD_TYPE MyType = SPD;
	typedef linalg<LA_HANDLER> la;

	// Scalar type of manifold
	typedef double scalar_type;

	// Value Typedef
	typedef typename la::vect<scalar_type, N>::type		value_type;
	typedef typename la::vect<scalar_type, N>::ref_type	ref_type;
	typedef typename la::vect<scalar_type, N>::cref_type	cref_type;
   
	// Derivative Typedefs
	typedef typename la::vect<scalar_type, 2*N>::type		deriv1_type;
	typedef typename la::vect<scalar_type, 2*N>::ref_type		deriv1_ref_type;
	typedef typename la::mat<scalar_type, 2*N, 2*N>::type		deriv2_type;
	typedef typename la::mat<scalar_type, 2*N, 2*N>::ref_type	deriv2_ref_type;
	

	typedef double dist_type;

	// Manifold distance functions (for IRLS)
	inline static dist_type dist_squared(cref_type x, cref_type y) const;
	inline static void deriv_dist_squared(cref_type x, cref_type y, deriv1_ref_type result) const;

	// Manifold exponentials und logarithms ( for Proximal point)
	static value_type exp(cref_type x) const;
	static value_type log(cref_type x) const;

    private:
	std::ostream& write_( std::ostream& out ) const; 

};

// Specialization SPD
template < int N, LA_HANDLER >
struct Manifold< SPD, N, LA_HANDLER > {
    public:
	static const MANIFOLD_TYPE MyType = SPD;

	typedef double scalar_type;
	typedef typename LA_HANDLER::SPD<scalar_type, N> value_type;
	typedef typename LA_HANDLER::MAT<scalar_type, N> mat_type;
	
	typedef double dist_type;

	// Manifold distance functions (for IRLS)
	// FIXME Additional function template parameters for EIGEN, maybe outsource in traits class
	dist_type dist_squared(const value_type& x, const value_type& y) const;
	dist_type deriv_dist_squared(const value_type& x, const value_type& y) const;

	// Manifold exponentials und logarithms ( for Proximal point)
	value_type exp(const value_type& x) const;
	value_type log(const value_type& x) const;

	// Projector
	value_type projector(const mat_type& z) const;
	
	// Return orthonormal bases of tangent spaces
	mat_type tangent_space_basis(const value_type& x, const value_type& y) const;

    private:
	std::ostream& write_( std::ostream& out ) const; 
};

// Specialization SPHERE
template < int N, LA_HANDLER >
struct Manifold< SPHERE, N, LA_HANDLER > {

};

// Specialization SO(N)
template < int N, LA_HANDLER >
struct Manifold< SO, N, LA_HANDLER > {

};


/*-----IMPLEMENTATION EUCLIDIAN----------*/ 

// Squared Euclidian distance function
template <int N, LA_HANDLER >
static inline dist_type Manifold < EUCLIDIAN, N, LA_HANDLER >::dist_squared( cref_type x, cref_type y ){
    return la::squaredNorm(x-y);
}

// Derivative of Squared Euclidian distance
template <int N, LA_HANDLER >
static inline value_type deriv_dist_type Manifold < EUCLIDIAN, N, LA_HANDLER >::dist_squared( cref_type x, cref_type y, deriv1_ref_type result){
    result.head<N> = 2 * (x-y);
    result.tail<N> = -2 * (x-y);
}


} // end namespace tvmtl







#endif
