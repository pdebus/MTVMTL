
//System includes
#include <iostream>
#include <chrono>

//Eigen includes
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay_triangulation;
typedef CGAL::Interpolation_traits_2<K> Traits;
typedef K::FT Coord_type;
typedef K::Point_2 Point;


int main(){

    typedef Eigen::Spline<double, 1, 1> spline_type;    
    typedef typename spline_type::PointType point_type;
    typedef typename spline_type::ControlPointVectorType cpv_type;

    const Eigen::VectorXd xvals = (Eigen::VectorXd(9) << 0, 0, 0, 1, 1, 1, 2, 2, 2).finished();
    const Eigen::VectorXd yvals = (Eigen::VectorXd(9) << 0, 1, 2, 0, 1, 2, 0, 1, 2).finished();
    cpv_type nodes(2,9);
    nodes.row(0)=xvals;
    nodes.row(1)=yvals;

    const Eigen::VectorXd zvals = xvals.array().square()+yvals.array().square(); 

    const spline_type spline = Eigen::SplineFitting<spline_type>::Interpolate( zvals.transpose(), 1, xvals.transpose());

    Delaunay_triangulation T;
 
    std::map<Point, Coord_type, K::Less_xy_2> function_values;
 
    typedef CGAL::Data_access< std::map<Point, Coord_type, K::Less_xy_2 > > Value_access;
 
    Coord_type a(0.25), bx(1.3), by(-0.7);
 
    for (int y=0 ; y<3 ; y++)
	for (int x=0 ; x<3 ; x++){
	    K::Point_2 p(x,y);
	    T.insert(p);
	    function_values.insert(std::make_pair(p,a + bx* x+ by*y));
	}
 
    //coordinate computation
    K::Point_2 p(1.3,0.34);
 
    std::vector< std::pair< Point, Coord_type > > coords;
 
    Coord_type norm = CGAL::natural_neighbor_coordinates_2 (T, p,std::back_inserter(coords)).second;
    Coord_type res = CGAL::linear_interpolation(coords.begin(), coords.end(), norm, Value_access(function_values));
 
    std::cout << " Tested interpolation on " << p << " interpolation: " << res << " exact: " << a + bx* p.x()+ by* p.y()<< std::endl;
    std::cout << "done" << std::endl;
 
 
     return 0; 
}
