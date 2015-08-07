#include <iostream>
#include <limits>

#include <vpp/vpp.hh>

#define TV_DATA_DEBUG
#include "../core/data.hpp"

int main(int argc, const char *argv[])
{
	using namespace tvmtl;

	typedef Manifold< EUCLIDIAN, 1 > mf_t;
	typedef Data< mf_t, 3> data_t;	

	data_t myData=data_t();
	myData.create_noisy_gray(7, 5, 3);

    int ns = myData.img_.nslices();  // z
    int nr = myData.img_.nrows();    // y
    int nc = myData.img_.ncols();    // x
    
    for(int s = 0; s < ns; ++s){
	std::cout << "\nSlice " << s << ":\n";
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		std::cout << myData.img_(s, r, c) << " ";
	    std::cout << std::endl;
	}
    }

    std::cout << "\n\nAddress of image:\t"  << &myData.img_(0, 0, 0) << std::endl;
    std::cout << "Size of element:\t"	    << sizeof(myData.img_(0,0,0)) << std::endl;
    std::cout << "Address of next slice:\t" << &myData.img_(1, 0, 0) << "\tDifference:\t"  << &myData.img_(1,0,0) - &myData.img_(0,0,0) << std::endl;
    std::cout << "Address of next row:\t"   << &myData.img_(0, 1, 0) <<  "\tDifference:\t" << &myData.img_(0,1,0) - &myData.img_(0,0,0) << std::endl;
    std::cout << "Address of next col:\t"   << &myData.img_(0, 0, 1) <<  "\tDifference:\t" << &myData.img_(0,0,1) - &myData.img_(0,0,0) << std::endl;

    auto output = [] (const mf_t::value_type& i) { std::cout << i << " "; };
    auto fill = [] (mf_t::value_type& i) { i.setConstant(0.5); };
    auto copy = [] (const mf_t::value_type& src, mf_t::value_type& dst) { dst = src;  };
    auto add = [] (mf_t::value_type& res, const mf_t::value_type& op1, const mf_t::value_type& op2) {res = op1 + op2;};

    pixel_wise3d(fill, myData.img_);
    pixel_wise3d(output, myData.img_);
    pixel_wise3d(copy, myData.noise_img_, myData.img_);
    pixel_wise3d(output, myData.img_);
    pixel_wise3d(add, myData.img_, myData.noise_img_, myData.img_);
    pixel_wise3d(output, myData.img_);


    return 0;
}
