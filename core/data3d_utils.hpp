#ifndef TVTML_DATA3D_UTILS_HPP
#define TVTML_DATA3D_UTILS_HPP

#include <vpp/vpp.hh>

namespace tvmtl{

template<class T>
auto* get_row_pointer(int s, int r, T& image3d){
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
void pixel_wise3d(FUNC func, T& head){

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
void pixel_wise3d(FUNC func, T& head, Args&... args){

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


}
#endif
