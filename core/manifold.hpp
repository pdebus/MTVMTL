#ifndef TVMTL_MANIFOLD_HPP
#define TVMTL_MANIFOLD_HPP

//own includes

#include "enumerators.hpp"

namespace tvmtl {

// Primary Template
template < enum MANIFOLD_TYPE MF, int N, int P=0 >
struct Manifold { };

} // end namespace tvmtl

#include "manifold_euc.hpp"
#include "manifold_sphere.hpp"
#include "manifold_son.hpp"
#include "manifold_spd.hpp"
#include "manifold_grassmann.hpp"

#endif
