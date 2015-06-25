#ifndef TVMTL_FUNCPTRUTILS_HPP
#define TVMTL_FUNCPTRUTILS_HPP

//System includes
#include <functional>


namespace tvmtl{

// Small Helper Class to provide an member function interface to the OpenGL C-API
template <typename T>
struct Callback;

template <typename Ret, typename... Params>
struct Callback<Ret(Params...)> {
   template <typename... Args> 
   static Ret callback(Args... args) {                    
         func(args...);  
      }
   static std::function<Ret(Params...)> func; 
};

template <typename Ret, typename... Params>
std::function<Ret(Params...)> Callback<Ret(Params...)>::func;

// Helper Function for the most common case of void(void) functions
typedef void (*callback_t)();
template<typename T>
callback_t getCFunctionPointer(void (T::*member_function)(), T* object_pointer){
    Callback<void()>::func = std::bind(member_function, object_pointer);
    callback_t func = static_cast<callback_t>(Callback<void()>::callback);
    return func;
}

} // end namespace tvmtl
#endif  
