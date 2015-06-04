    #include <Eigen/Dense>
    #include <Eigen/Sparse>
    #include <unsupported/Eigen/AutoDiff>

    #include <iostream>
    #include <cmath>

    template <typename T>
    T fun(Eigen::Matrix<T,Eigen::Dynamic,1> const &x){
       T y;
       y = x(0)*x(0)*x(0)*x(1) + x(0)*x(0)*x(1)*x(1)*x(1)*x(1); // f(x) = x[0]^3 * x[1]  + x[0]^2 * x[1]^4
       return y;
    }
/*
    template <typename T>
	T dist(Eigen::Matrix<T,Eigen::Dynamic,1> const &x, Eigen::Matrix<T,Eigen::Dynamic,1> const &y){
	T y;
	y = std::acos(x.dot(y));
	return y;
    }
*/

    int main(){
       //normal use of fun
       {
          typedef double scalar_t;
          typedef Eigen::Matrix<scalar_t,Eigen::Dynamic,1> input_t;
          input_t x(2);
          x.setConstant(1);
          scalar_t y = fun(x);
          std::cout << y << std::endl;
       }
	std::cout << std::endl;
       //autodiff use of dist
       {
       
       
	typedef typename Eigen::Matrix<double,Eigen::Dynamic,1> vec;
	vec vec1, vec2;
	vec1 = vec::Random(3).normalized(); 
	vec2 = vec::Random(3).normalized();

       
       
       }


	std::cout << std::endl;
       //autodiff use of fun
       {
          typedef Eigen::Matrix<double,Eigen::Dynamic,1> derivative_t;
          typedef Eigen::AutoDiffScalar<derivative_t> scalar_t;
          typedef Eigen::Matrix<scalar_t,Eigen::Dynamic,1> input_t;
          input_t x(2);
          x.setConstant(1);
          
          //set unit vectors for the derivative directions (partial derivatives of the input vector)
          x(0).derivatives().resize(2);
          x(0).derivatives()(0)=1;
          x(1).derivatives().resize(2);
          x(1).derivatives()(1)=1;

          scalar_t y = fun(x);
          std::cout << y.value() << std::endl;
          std::cout << y.derivatives() << std::endl;
       }
	std::cout << std::endl;
       //autodiff second derivative of fun
       {
          typedef Eigen::Matrix<double,Eigen::Dynamic,1> inner_derivative_t;
          typedef Eigen::AutoDiffScalar<inner_derivative_t> inner_scalar_t;
          typedef Eigen::Matrix<inner_scalar_t,Eigen::Dynamic,1> derivative_t;
          typedef Eigen::AutoDiffScalar<derivative_t> scalar_t;
          typedef Eigen::Matrix<scalar_t,Eigen::Dynamic,1> input_t;
          input_t x(2);
          x(0).value()=1;
          x(1).value()=1;
          
          //set unit vectors for the derivative directions (partial derivatives of the input vector)
          x(0).derivatives().resize(2);
          x(0).derivatives()(0)=1;
          x(1).derivatives().resize(2);
          x(1).derivatives()(1)=1;

          //repeat partial derivatives for the inner AutoDiffScalar
          x(0).value().derivatives() = inner_derivative_t::Unit(2,0);
          x(1).value().derivatives() = inner_derivative_t::Unit(2,1);

          //set the hessian matrix to zero
          for(int idx=0;idx<2;idx++){
             x(0).derivatives()(idx).derivatives()  = inner_derivative_t::Zero(2);
             x(1).derivatives()(idx).derivatives()  = inner_derivative_t::Zero(2);
          }

          scalar_t y = fun(x);
          std::cout << y.value().value() << std::endl;
          std::cout << y.value().derivatives() << std::endl;
          std::cout << y.derivatives()(0).value() << std::endl;
          std::cout << y.derivatives()(0).derivatives() << std::endl;
          std::cout << y.derivatives()(1).value() << std::endl;
          std::cout << y.derivatives()(1).derivatives() << std::endl;
       }
    }
