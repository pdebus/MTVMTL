#ifndef TVTML_DATA3D_UTILS_HPP
#define TVTML_DATA3D_UTILS_HPP

#include <vpp/vpp.hh>

namespace tvmtl{

template<class T>
auto* get_row_pointer(int s, int r, T&& image3d){
    return &image3d(s, r, 0);
}

template <class FUNC, class T>
void perform_inner_loop(FUNC func, int nc, T* head){
   for(int c = 0; c < nc; ++c)
	func(head[c]);
}

template <class FUNC, class T, class... Args>
void perform_inner_loop(FUNC func, int nc, T* head, Args*... args){
   for(int c = 0; c < nc; ++c)
	func( head[c], args[c]...);
}


template <class FUNC, class T>
void pixel_wise3d(FUNC func, T&& head){

    int ns = head.nslices();  // z
    int nr = head.nrows();    // y
    int nc = head.ncols();    // x
    
    for(int s = 0; s < ns; ++s){
	#pragma omp parallel for
	for(int r = 0; r < nr; ++r){
	    auto* head_row_pointer = &head(s, r, 0);
	    for(int c = 0; c < nc; ++c)
		func(head_row_pointer[c]);
	}
    } 

}

template <class FUNC, class T, class... Args>
void pixel_wise3d(FUNC func, T&& head, Args&&... args){

    int ns = head.nslices();  // z
    int nr = head.nrows();    // y
    int nc = head.ncols();    // x
   
    for(int s = 0; s < ns; ++s){
	#pragma omp parallel for
	for(int r = 0; r < nr; ++r){
	    auto* head_row_pointer = &head(s,r,0);
	    perform_inner_loop(func, nc, head_row_pointer, get_row_pointer(s, r, args)...);
	}
    }
}

template <class FUNC, class T>
void pixel_wise3d_nothreads(FUNC func, T&& head){

    int ns = head.nslices();  // z
    int nr = head.nrows();    // y
    int nc = head.ncols();    // x
    
    for(int s = 0; s < ns; ++s){
	for(int r = 0; r < nr; ++r){
	    auto* head_row_pointer = &head(s, r, 0);
	    for(int c = 0; c < nc; ++c)
		func(head_row_pointer[c]);
	}
    } 

}

template <class FUNC, class T, class... Args>
void pixel_wise3d_nothreads(FUNC func, T&& head, Args&&... args){

    int ns = head.nslices();  // z
    int nr = head.nrows();    // y
    int nc = head.ncols();    // x
   
    for(int s = 0; s < ns; ++s){
	for(int r = 0; r < nr; ++r){
	    auto* head_row_pointer = &head(s,r,0);
	    perform_inner_loop(func, nc, head_row_pointer, get_row_pointer(s, r, args)...);
	}
    } 
}


// FIXME: image(last) and subimage(1,1,1) are not the same. It seems the subimage can only be created from sequential memory addresses....Find workaround
/*
template <class FUNC, class DIMS, class T>
void block_wise3d(FUNC func, DIMS dims, T&& head){

    int ns = head.nslices();  // z
    int nr = head.nrows();    // y
    int nc = head.ncols();    // x

    for(int s = 0; s < ns; s += dims(0)){
	//#pragma omp parallel for
	for(int r = 0; r < nr; r += dims(1)){
	    for(int c = 0; c < nc; c+= dims(2)){
		vpp::vint3 first(s,r,c);
		vpp::vint3 last(s + dims(0)-1, r + dims(1)-1, c + dims(2)-1 );
		vpp::box3d block(first, last);
		std::cout << "\n first:\n " << first << std::endl;
		std::cout << "last:\n " << last << std::endl;
		auto temp = head.subimage(block);
		std::cout << "\nimage(first): " << head(first) << std::endl;
		std::cout << "subimage(0,0,0): " << temp(0,0,0) << std::endl;
		std::cout << "image(last): " << head(last) << std::endl;
		std::cout << "subimage(1,1,1): " << temp(1,1,1) << std::endl;
		func(head.subimage(block));
	    }
	}
    } 

}

template <class FUNC, class DIMS, class T, class... Args>
void block_wise3d(FUNC func, DIMS dims, T&& head, Args&&... args){

    int ns = head.nslices();  // z
    int nr = head.nrows();    // y
    int nc = head.ncols();    // x
   
    for(int s = 0; s < ns; s += dims(0)){
	//#pragma omp parallel for
	for(int r = 0; r < nr; r += dims(1)){
	    for(int c = 0; c < nc; c+= dims(2)){
		vpp::vint3 first(s,r,c);
		vpp::vint3 last(s + dims(0), r + dims(1), c + dims(2) );
		vpp::box3d block(first, last);
		func(head | block, (args | block)...);
	    }
	}
    } 
}
*/

template <class IMG, class VAL>
void fill3d(IMG&& img, VAL&& val){
    auto fill = [=] (VAL& i) { i = val;};
    pixel_wise3d(fill, img);
}

template <class SRC, class DST>
void clone3d(SRC&& src, DST&& dst){
    //TODO static assert for domains
    auto clone = [] (const auto& src_pixel, auto& dst_pixel) { dst_pixel = src_pixel;  };
    pixel_wise3d(clone, src, dst);
}

template <class IMG>
auto sum3d(IMG&& img){
    auto sum = img(0,0,0);
    auto add = [&] (const auto& i) { sum += i; };
    pixel_wise3d_nothreads(add, img);
    return sum;
}


}
#endif
