#ifndef TVMTL_LAHANDLER_HPP
#define TVMTL_LAHANDLER_HPP

#include <Eigen/Core>
#include <Eigen/QR>

namespace tvmtl {

// Primary Template
template <enum LA_HANDLER LA>
struct linalg{
};


// Eigen Interface
template <>
struct linalg <EIGEN>{

    //VECTOR TYPES
    template <scalar, N>
    struct vect{
	typedef Eigen::Matrix<scalar, N, 1> type;
	typedef type& ref_type;
	typedef const type& cref_type;

	inline static scalar squared_norm(cref_type x){
	    return x.squaredNorm();
	}
    }
    
    //MATRIX TYPES


};


} // end namespace tvmtl







#endif
