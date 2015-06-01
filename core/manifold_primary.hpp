#ifndef TVMTL_MANIFOLD_PRIMARY_HPP
#define TVMTL_MANIFOLD_PRIMARY_HPP

//own includes
#include "enumerators.hpp"

namespace tvmtl {

// Primary Template
template < enum MANIFOLD_TYPE MF, int N >
struct Manifold { };


// Specialization SO(N)
template < int N>
struct Manifold< SO, N> {

};

// Specialization SPD
template < int N >
struct Manifold< SPD, N > {
    public:
	static const MANIFOLD_TYPE MyType = SPD;
};

} // end namespace tvmtl








#endif
