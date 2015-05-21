#ifndef TVMTL_ALGOTRAITS_HPP
#define TVMTL_ALGOTRAITS_HPP

// own includes 
#include "enumerators.hpp"

// Eigen includes
#include <Eigen/Sparse>

namespace tvmtl {

template<enum ALGORITHM AL>
struct algo_traits {

};


template<>
struct algo_traits<IRLS> {
	    
    static const int max_runtime = 100;
    static const int max_irls_steps = 5;
    static const int max_newtons_steps = 3;
    static constexpr double tolerance = 1e-4;

    static const bool use_preconditioner = false;
    
    template <typename H>
    using solver = Eigen::SimplicialLDLT<H, Eigen::Upper>;

};


}// end namespace tvmtl


#endif
