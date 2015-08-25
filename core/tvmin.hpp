#ifndef TVMTL_TVMINIMIZER_HPP
#define TVMTL_TVMINIMIZER_HPP

//System includes
#include <iostream>
#include <map>
#include <vector>
#include <chrono>

#ifdef TVMTL_TVMIN_DEBUG
    #include <string>
#endif

//Eigen includes
#include <Eigen/Sparse>
#include <Eigen/Core>

//CGAL includes For linear Interpolation
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>

//vpp includes
#include <vpp/vpp.hh>

//own inlcudes
#include "manifold.hpp"
#include "data.hpp"
#include "functional.hpp"


namespace tvmtl {

// Primary Template
template < enum ALGORITHM AL, class FUNCTIONAL, class MANIFOLD, class DATA, enum PARALLEL PAR=OMP, int DIM=2 > 
    class TV_Minimizer {
    
    };

}

#include "tvmin_irls.hpp"
#include "tvmin_irls3d.hpp"
#include "tvmin_prpt.hpp"
#include "tvmin_prpt3d.hpp"

#endif
