#include <iostream>
#include <limits>

#include <vpp/vpp.hh>

#define TV_DATA_DEBUG
#include "../core/data.hpp"
#include "../core/visualization.hpp"

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
    std::cout << std::endl;

    pixel_wise3d(copy, myData.noise_img_, myData.img_);
    pixel_wise3d(output, myData.img_);
    std::cout << std::endl;

    pixel_wise3d(add, myData.img_, myData.noise_img_, myData.img_);
    pixel_wise3d(output, myData.img_);
    std::cout << std::endl;

    mf_t::value_type v; v.setConstant(0.5);

    fill3d(myData.img_, v);
    pixel_wise3d(output, myData.img_);
    std::cout << std::endl;
    
    clone3d(myData.noise_img_, myData.img_);
    pixel_wise3d(output, myData.img_);
    std::cout << std::endl;
    
    mf_t::value_type sum = sum3d(myData.img_);

    std::cout << "SUM of all elements: " <<  sum << std::endl;

    typedef Manifold< EUCLIDIAN, 3 > mf_t2;
    typedef Data< mf_t2, 3> data_t2;	

    data_t2 myData2=data_t2();
    myData2.create_noisy_rgb(3, 2, 2);

    ns = myData2.img_.nslices();  // z
    nr = myData2.img_.nrows();    // y
    nc = myData2.img_.ncols();    // x
    
    for(int s = 0; s < ns; ++s){
	std::cout << "\nSlice " << s << ":\n";
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		std::cout << myData2.img_(s, r, c) << " ";
	    std::cout << std::endl;
	}
    }

    std::cout << "\n\n\n Blockwise test" << std::endl;
    data_t myData3 = data_t();
    myData3.create_noisy_gray(6,4,2);

    ns = myData3.img_.nslices();  // z
    nr = myData3.img_.nrows();    // y
    nc = myData3.img_.ncols();    // x
    
    for(int s = 0; s < ns; ++s){
	std::cout << "\nSlice " << s << ":\n";
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		std::cout << myData3.img_(s, r, c) << " ";
	    std::cout << std::endl;
	}
    }

    int k=0; 
    vpp::vint3 dims(2,2,2);

    for(int s = 0; s < ns; s += dims(0)){
	#pragma omp parallel for
	for(int r = 0; r < nr; r += dims(1)){
	    for(int c = 0; c < nc; c+= dims(2)){
		vpp::vint3 first(s,r,c);
		vpp::vint3 last(s + dims(0) - 1, r + dims(1) - 1, c + dims(2)-1 );
		vpp::box3d block(first, last);
	
		for (auto p : block){
		    myData3.img_(p).setConstant(k);
		}
		++k;
	    }
	}
    }

   for(int s = 0; s < ns; ++s){
	std::cout << "\nSlice " << s << ":\n";
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		std::cout << myData3.img_(s, r, c) << " ";
	    std::cout << std::endl;
	}
    } 

// SLICE TEST
    vpp::box3d without_last_x(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 1, nc - 2)); // subdomain without last xslice
    vpp::box3d without_last_y(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 2, nc - 1)); // subdomain without last yslice
    vpp::box3d without_last_z(vpp::vint3(0,0,0), vpp::vint3(ns - 2, nr - 1, nc - 1)); // subdomain without last zslice
    vpp::box3d without_first_x(vpp::vint3(0,0,1), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first xlice
    vpp::box3d without_first_y(vpp::vint3(0,1,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first yslice
    vpp::box3d without_first_z(vpp::vint3(1,0,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first zslice

   
   std::cout << "\n\n\nSlice test:" << std::endl;
    k=0;
    for(int s = 0; s < ns; ++s){
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		myData3.img_(s,r,c).setConstant(k++);
	}
    }
    
    for(int s = 0; s < ns; ++s){
	std::cout << "\nSlice " << s << ":\n";
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		std::cout << myData3.img_(s, r, c) << " ";
	    std::cout << std::endl;
	}
    }

    auto I2 = myData3.img_ | without_last_z;
    auto I3 = myData3.img_ | without_first_z;

    for(int s = 0; s < ns-1; ++s){
	std::cout << "\nSlice " << s << ":\n";
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		std::cout << I2(s, r, c) << " ";
	    std::cout << std::endl;
	}
    }
    
    for(int s = 0; s < ns-1; ++s){
	std::cout << "\nSlice " << s << ":\n";
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		std::cout << I3(s, r, c) << " ";
	    std::cout << std::endl;
	}
    }
    
    data_t2 myData4=data_t2();
    myData4.rgb_slice_reader("slices/noisy_crayons0.jpg", 32);

    data_t2 myData5=data_t2();
    myData5.create_noisy_rgb(24, 64, 32);

/*
    ns = 10; //myData4.img_.nslices();  // z
    nr = 10; //myData4.img_.nrows();    // y
    nc = 10; //myData4.img_.ncols();    // x
    
    for(int s = 0; s < ns; ++s){
	std::cout << "\nSlice " << s << ":\n";
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		std::cout << myData4.img_(s, r, c) << " ";
	    std::cout << std::endl;
	}
    } 
*/
	typedef Visualization<EUCLIDIAN, 3, data_t2, 3> visual_t;

	visual_t myVisual(myData4);
	myVisual.saveImage("3D_Cube_Crayons.png");

	std::cout << "Starting OpenGL-Renderer..." << std::endl;
	myVisual.GLInit("Image Cube Renderer ");
	std::cout << "Rendering finished." << std::endl;


    return 0;
}
