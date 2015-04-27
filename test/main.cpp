// System includes
#include <iostream>

// TVM includes
#include "../core/manifold.hpp"
#include "../core/data.hpp"
#include "../core/functional.hpp"
#include "../core/tvmin.hpp"


using namespace tvmtl;


int main(){

// SECTION DATA

    // Manifold and data typedefs
    typedef Manifold< SPD, 3, BLAZE > MF_t;
    typedef Data < MF_t, BLAZE > data_t;

 /*----------------------------------
 * Data input routines
 */

    // Instantiation and initialization of the data
    data_t myPicture(raw_input);

// SECTION FUNCTIONAL
    typedef Functional< MF_t, FIRSTORDER, ISO > functional_t;

    // Instantiation and initialization of the functional
    double lambda = 1.0;
    functional_t myFunc(lambda);

// SECTION TVMIN 
    
    typedef TV_Minimizer < IRLS, functional_t, data_t, BLAZE, OMP > tvmin_t;
    tvmin_t myTVM(myFunc, myPicture);

    // Compute result
    myTVM.minimize();
    output = myTVM.output()

    return 0;
}
