#ifndef TVMTL_ALGOTRAITS_HPP
#define TVMTL_ALGOTRAITS_HPP

// own includes 
#include "enumerators.hpp"

// Eigen includes
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>
#include <Eigen/SuperLUSupport>
//#include <Eigen/PaStiXSupport>

namespace tvmtl {

template<enum MANIFOLD_TYPE MF>
struct algo_traits {

};


template<>
struct algo_traits<EUCLIDIAN> {
	    
    static const int max_runtime = 1000;
    static const int max_irls_steps = 5;
    static const int max_newtons_steps = 1;
    static constexpr double tolerance = 1e-4;

    static const bool use_preconditioner = false;
    
    template <typename H>
    using solver = Eigen::CholmodSupernodalLLT<H, Eigen::Upper>;
    //using solver = Eigen::PastixLLT<H, Eigen::Lower>;
    //using solver = Eigen::SimplicialLLT<H, Eigen::Upper>;
};

template<>
struct algo_traits<SPHERE> {
	    
    static const int max_runtime = 1000;
    static const int max_irls_steps = 5;
    static const int max_newtons_steps = 1;
    static constexpr double tolerance = 1e-4;

    static const bool use_preconditioner = false;
    
    template <typename H>
    using solver = Eigen::SuperLU<H>;
    //using solver = Eigen::SparseLU<H>;
};

template<>
struct algo_traits<SO> {
	    
    static const int max_runtime = 1000;
    static const int max_irls_steps = 5;
    static const int max_newtons_steps = 1;
    static constexpr double tolerance = 1e-4;

    static const bool use_preconditioner = false;
    
    template <typename H>
    using solver = Eigen::SuperLU<H>;
    //using solver = Eigen::SparseLU<H>;
};

template<>
struct algo_traits<SPD> {
	    
    static const int max_runtime = 1000;
    static const int max_irls_steps = 5;
    static const int max_newtons_steps = 1;
    static constexpr double tolerance = 1e-4;

    static const bool use_preconditioner = false;
    
    template <typename H>
    using solver = Eigen::CholmodSupernodalLLT<H, Eigen::Upper>;
    //using solver = Eigen::SuperLU<H>;
};


}// end namespace tvmtl


#endif
