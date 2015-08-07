#include <iostream>
#include <limits>

#include <vpp/vpp.hh>

#define TV_DATA_DEBUG
#include "../core/data.hpp"
int main(int argc, const char *argv[])
{
    /*
	using namespace tvmtl;

	if (argc != 2){
	    std::cerr << "Usage : " << argv[0] << " image" << std::endl;
	    return 1;
	}

	typedef Manifold< EUCLIDIAN, 1 > mf_t;
	typedef Data< mf_t, 3> data_t;	

	data_t myData=data_t();
	myData.create_noisy_gray(30, 20, 15);
*/
  using vpp::image3d;
  using vpp::vint3;
  using vpp::make_box3d;

  image3d<int> img1(make_box3d(100, 200, 300));
  image3d<int> img2({100, 200, 300});

  assert(img1.domain() == img2.domain());

  assert(img1.nslices() == 100);
  assert(img1.nrows() == 200);
  assert(img1.ncols() == 300);
    
  for(int s = 0; s < img1.nslices(); ++s)
    for(int r = 0; r < img1.nrows(); ++r)
	for(int c = 0; c < img1.ncols(); ++c)
	    img1(s, r, c) = 5;

	return 0;
}
