#include <iostream>
#include <limits>

#include "../test/imageNd.hh"
#include <vpp/vpp.hh>



template <typename V, unsigned N>
  int coords_to_offset(vpp::imageNd<V,N>& img, const auto& p, const auto& strides) 
    {
        int row_idx = p[N-2];
        int ds = 1;
        for (int i = N - 3; i >= 0; i--)
        {
	      ds *= strides[i + 1];
	      row_idx += ds * p[i];
	    }
        return row_idx * img.pitch() + p[N - 1] * sizeof(V);
      }




int main(int argc, const char *argv[])
{
    vpp::image3d<int> img(5,4,3);

    int ns = img.nslices();  // z
    int nr = img.nrows();    // y
    int nc = img.ncols();    // x
    

// SLICE TEST
    vpp::box3d without_last_x(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 1, nc - 2)); // subdomain without last xslice
    vpp::box3d without_last_y(vpp::vint3(0,0,0), vpp::vint3(ns - 1, nr - 2, nc - 1)); // subdomain without last yslice
    vpp::box3d without_last_z(vpp::vint3(0,0,0), vpp::vint3(ns - 2, nr - 1, nc - 1)); // subdomain without last zslice
    vpp::box3d without_first_x(vpp::vint3(0,0,1), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first xlice
    vpp::box3d without_first_y(vpp::vint3(0,1,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first yslice
    vpp::box3d without_first_z(vpp::vint3(1,0,0), vpp::vint3(ns - 1, nr - 1, nc - 1)); // subdomain without first zslice

   
   std::cout << "\n\n\nSlice test:" << std::endl;
    int k=0;
    for(int s = 0; s < ns; ++s){
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		img(s,r,c) = k++;
	}
    }
    
    for(int s = 0; s < ns; ++s){
	std::cout << "\nSlice " << s << ":\n";
	for(int r = 0; r < nr; ++r){
	    for(int c = 0; c < nc; ++c)
		std::cout << img(s, r, c) << " ";
	    std::cout << std::endl;
	}
    }
 
    std::cout << "\nImage Iterator: \n";
    for( auto p : img ) std::cout << p << "  ";


    std::cout << "\nMemory pointer: \n";
    int* v = &img(0,0,0);
    int* end = &img(ns-1,nr-1,nc-1);
    for(; v!=end+1; ++v) std::cout << v << "  ";
    for(; v!=end+1; ++v) std::cout << *v << "  ";

    auto Ilz = img | without_last_z;
    auto Ifz = img | without_first_z;
    auto Ily = img | without_last_y;
    auto Ify = img | without_first_y;
    auto Ilx = img | without_last_x;
    auto Ifx = img | without_first_x;

    auto print = [] (vpp::image3d<int>& img, const std::string slname){
    std::cout << "\nWithout " << slname << std::endl;
        int ns = img.nslices();  // z
	int nr = img.nrows();    // y
	int nc = img.ncols();    // x
	std::cout << "Pitch: " << img.pitch() << std::endl;
	//std::cout << "Alignment: " << img.alignment();

	for(int s = 0; s < ns; ++s){
	    std::cout << "\nSlice " << s << ":\n";
	    for(int r = 0; r < nr; ++r){
		for(int c = 0; c < nc; ++c)
		    std::cout << img(s, r, c) << " ";
		std::cout << std::endl;
	    }
	}
    
    std::cout << "\nImage Iterator " << slname << std::endl;
    for( auto p : img ) std::cout << p << "  ";

    std::cout << "\n\nMemory pointer: " << slname << std::endl;
    int* v = &img(0,0,0);
    int* end = &img(ns-1,nr-1,nc-1);
    std::cout << "Memory pointer begin: " << v << std::endl;
    std::cout << "Memory pointer end:" << end << std::endl;
    for(; v!=end+1; ++v) std::cout << *v << "  ";

    std::cout << "\n\nCoord to offset: " << slname << std::endl;
	for(int s = 0; s < ns; ++s){
	    for(int r = 0; r < nr; ++r){
		for(int c = 0; c < nc; ++c)
		    std::cout << img.coords_to_offset(vpp::vint3(s, r, c)) / sizeof(int)<< " ";
	    }
	}


    std::cout << "\n\nCorrected Offset: " << slname << std::endl;
	for(int s = 0; s < ns; ++s){
	    for(int r = 0; r < nr; ++r){
		for(int c = 0; c < nc; ++c)
		    std::cout << coords_to_offset(img, vpp::vint3(s, r, c), vpp::vint3(5, 4, 3)) / sizeof(int)<< " ";
	    }
	}
    
    v = &img(0,0,0);
    std::cout << "\n\nCorrected Values: " << slname << std::endl;
	for(int s = 0; s < ns; ++s){
	    for(int r = 0; r < nr; ++r){
		for(int c = 0; c < nc; ++c)
		    std::cout << *(v + coords_to_offset(img, vpp::vint3(s, r, c), vpp::vint3(5,4,3)) / sizeof(int)) << " ";
	    }
	}


    std::cout << "\nSize of V: " <<  sizeof(int) << std::endl;

    std::cout << "\n\nRow Indices: " << slname << std::endl;
	for(int s = 0; s < ns; ++s){
	    for(int r = 0; r < nr; ++r){
		for(int c = 0; c < nc; ++c)
		    std::cout << (img.coords_to_offset(vpp::vint3(s, r, c)) - c * sizeof(int)) / img.pitch() << " ";
	    }
	} 
/*
    std::cout << "\n\nAll: " << slname << std::endl;
	for(int s = 0; s < ns; ++s){
	    for(int r = 0; r < nr; ++r){
		for(int c = 0; c < nc; ++c){
		    std::cout << "\n(" << s << ", " << r << ", " << c << ")" << std::endl;
		    std::cout << "Value:" << img(s, r, c) << std::endl; 
		    std::cout << "c*size: " << c * sizeof(typename mf_t::value_type) << std::endl; 
		    std::cout << "row_idx: " << (img.coords_to_offset(vpp::vint3(s, r, c)) - c * sizeof(typename mf_t::value_type)) / img.pitch() << std::endl;
		    std::cout << "offset: " << img.coords_to_offset(vpp::vint3(s, r, c)) / sizeof(typename mf_t::value_type) << std::endl; 
		}
	    }
	    std::cout << std::endl;
	}*/ 
    };
   
    print(Ilz, "last Z");
    print(Ifz, "first Z");
    print(Ily, "last Y");
    print(Ify, "first Y");
    print(Ilx, "last X");
    print(Ifx, "first X");
  
    return 0;
}
