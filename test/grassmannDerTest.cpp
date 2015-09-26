#include <Eigen/Eigen>
#include <unsupported/Eigen/KroneckerProduct>

#include <mtvmtl/core/manifold.hpp> 
#include <cmath>
#include <cstdlib>


using namespace Eigen;
using namespace tvmtl;


template <typename DerivedY, typename DerivedA>
double F(const MatrixBase<DerivedY>& Y, const MatrixBase<DerivedA>& A){
 return 0.5 * (Y.transpose() * A * Y).trace();
}

template <typename DerivedY, typename DerivedA>
MatrixBase<DerivedY> FY(const MatrixBase<DerivedY>& Y, const MatrixBase<DerivedA>& A){
 return A * Y;
}


int main(){

srand(42);

const int N=4;
const int P=3;

typedef Matrix<double, P, P> ppmat;
typedef Matrix<double, N, P> npmat;
typedef Matrix<double, N, N> nnmat;
typedef Matrix<double, N*P, N*P> np2mat;
typedef Manifold<GRASSMANN, N, P> mf;

nnmat R = nnmat::Random();
nnmat A = 0.5 * (R + R.transpose());

std::cout << "A:\n" << A << std::endl;

npmat Y;
Y = npmat::Random();
HouseholderQR<npmat> qr(Y);
npmat q = qr.householderQ() * npmat::Identity();
Y = q;

std::cout << "Y:\n" << Y << std::endl;

nnmat HYproj = nnmat::Identity() - Y * Y.transpose();

double h = 1e-2;
npmat dY;
dY = npmat::Random(); 
qr.compute(dY);
q = qr.householderQ() * npmat::Identity();
dY = h * HYproj * q;

std::cout << "DY:\n" << dY << std::endl;

VectorXd vecdY = Map<VectorXd>(dY.data(), dY.size());
	
npmat FY = A * Y;
std::cout << "\nFY:\n" << FY << std::endl;

np2mat FYY = kroneckerProduct(ppmat::Identity(), A);
std::cout << "FYY:\n" << FYY << std::endl;

npmat YpdY = Y + dY;
//mf::exp(Y,dY,YpdY);
mf::projector(YpdY);

double exact = F(Y + dY, A);
double exact2 = F(YpdY, A);
double firstorder = F(Y,A) + FY.cwiseProduct(dY).sum();
double secondorder_euc = firstorder + 0.5 * FYY.cwiseProduct(vecdY * vecdY.transpose()).sum() ;

npmat FYYdY_proj = HYproj * A * dY;
npmat FYY_corr_term = dY * Y.transpose() * FY;
double secondorder_grass_nv = firstorder + 0.5 * (FYYdY_proj.cwiseProduct(dY).sum() - FYY_corr_term.cwiseProduct(dY).sum());

np2mat FYY_proj = kroneckerProduct(ppmat::Identity(), HYproj) * FYY ;
double secondorder_grass_sv = firstorder + 0.5 * (FYY_proj.cwiseProduct(vecdY * vecdY.transpose()).sum() - FYY_corr_term.cwiseProduct(dY).sum());

np2mat FYY_grass = FYY_proj - kroneckerProduct(FY.transpose() * Y, nnmat::Identity());
double secondorder_grass_fv = firstorder + 0.5 * (FYY_grass.cwiseProduct(vecdY * vecdY.transpose()).sum());
std::cout << "FYY_Grass:\n" << FYY_grass << std::endl;

std::cout << "\n\nFirst order approximation: " << std::abs(exact-firstorder) << std::endl;
std::cout << "Second order approximation (Euclidian): " << std::abs(exact-secondorder_euc) << std::endl;
std::cout << "Second order approximation (Grassmann non-vectorized): " << std::abs(exact2-secondorder_grass_sv) << std::endl;
std::cout << "Second order approximation (Grassmann semi-vectorized): " << std::abs(exact2-secondorder_grass_sv) << std::endl;
std::cout << "Second order approximation (Grassmann vectorized): " << std::abs(exact2-secondorder_grass_fv) << std::endl;



return 0;
}
