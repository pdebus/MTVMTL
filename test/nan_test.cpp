#include <iostream>
#include <cmath>


int main(int argc, const char *argv[])
{
 
	std::cout << "\n\nNAN TEST" << std::endl;
	std::cout << std::boolalpha << "isfinite(NaN) = " << std::isfinite(NAN) << '\n'
		             << "isfinite(Inf) = " << std::isfinite(INFINITY) << '\n'
			     << "isfinite(0.0) = " << std::isfinite(0.0) << '\n'
			     << "isfinite(0.0/0.0) = " << std::isfinite(0.0/0.0) << '\n'
			     << "isfinite(1.0/0.0) = " << std::isfinite(1.0/0.0) << '\n'
			     << "isfinite(exp(800)) = " << std::isfinite(std::exp(800)) << '\n';
	return 0;
}
