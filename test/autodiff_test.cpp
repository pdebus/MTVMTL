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

    template <typename T>
	T dist(Eigen::Matrix<T,Eigen::Dynamic,1> const &x, Eigen::Matrix<T,Eigen::Dynamic,1> const &y){
	T r;
	r =x.dot(y);
	return r;
    }


    int main(){
       //normal use of fun
       {
        
	  typedef double scalar_t;
          typedef Eigen::Matrix<scalar_t,Eigen::Dynamic,1> input_t;
          input_t x(2);
          x.setConstant(1);
          scalar_t y = fun(x);
	  std::cout << "Normal use of function" << std::endl;
	  std::cout << y << std::endl;
       }
	std::cout << std::endl;
       //autodiff use of dist
       {
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> vec;
	typedef Eigen::AutoDiffScalar<vec> AD;
	typedef Eigen::Matrix<AD,Eigen::Dynamic,1> ADvec;

	vec vec1, vec2;
	vec1 = vec::Random(3).normalized(); 
	vec2 = vec::Random(3).normalized();

	int s1 = vec1.size();
	int s2 = vec2.size();

	ADvec ax(s1);
	ADvec ay(s2);
	ax = vec1.cast<AD>();
	ay = vec2.cast<AD>();

	ax.setZero(s1);
	ay.setZero(s2);

	for(int i=0; i<s1; i++){
	    ax(i).derivatives().resize(s1);
	    ax(i).derivatives()(i)=1;
	    ay(i).derivatives().resize(s2);
	    ay(i).derivatives()(i)=1;
	}

	AD res = dist(ax,ay);
	std::cout << "\n Autodiff Dist function value: \n" << res.value() << std::endl;
        std::cout << "\n x-Derivatives\n " << res.derivatives() << std::endl;
//        std::cout << "\n y-Derivatives\n " << res.derivatives()(1) << std::endl;
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
	  std::cout << "Autodiff use of function" << std::endl;
          std::cout << "\nFunction:\n " << y.value() << std::endl;
          std::cout << "\nDerivatives\n " << y.derivatives() << std::endl;
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
          //std::cout << y.value().value() << std::endl;
          //std::cout << y.value().derivatives() << std::endl;
          //std::cout << y.derivatives()(0).value() << std::endl;
          std::cout << y.derivatives()(0).derivatives() << std::endl;
         // std::cout << y.derivatives()(1).value() << std::endl;
          std::cout << y.derivatives()(1).derivatives() << std::endl;
       }
    }
