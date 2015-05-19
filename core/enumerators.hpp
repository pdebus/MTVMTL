#ifndef ENUMERATORS_HPP
#define ENUMERATORS_HPP

//system includes
#include <iostream>

namespace tvmtl{

enum MANIFOLD_TYPE {
    EUCLIDIAN,
    SPHERE,
    SO,
    SPD
};

enum ALGORITHM {
    IRLS,
    PRPT
};

enum FUNCTIONAL_ORDER {
    FIRSTORDER,
    SECONDORDER
};

enum FUNCTIONAL_DISC {
    ISO,
    ANISO
};

enum PARALLEL {
    SEQ,
    OMP,
    CUDA
};

enum LA_HANDLER {
    EIGEN,
    ARMADILLO
};


inline std::ostream& operator<<( std::ostream& out, const MANIFOLD_TYPE& ret ){
    switch( ret ) {
	case EUCLIDIAN:	    out << "EUCLIDIAN";	    break;
        case SPHERE:        out << "SPHERE";        break;
	case SO:            out << "SO";            break;
	case SPD:           out << "SPD";           break;
        default:            out << "Undefined";     break;
	}
    return out;
} 




}// end namespace tvtml

#endif
