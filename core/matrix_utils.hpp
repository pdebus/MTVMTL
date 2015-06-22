#ifndef TVMTL_MATRIX_UTILS_HPP
#define TVMTL_MATRIX_UTILS_HPP

#include <cmath>
#include <complex>
#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>


namespace tvmtl {

template <typename MatrixType>
    void SolveTriangularSylvester(const MatrixType& A, const MatrixType& B, const MatrixType& C, MatrixType& result)
    {
/*	Eigen::eigen_assert(A.rows() == A.cols());
	Eigen::eigen_assert(A.isUpperTriangular());
	Eigen::eigen_assert(B.rows() == B.cols());
	Eigen::eigen_assert(B.isUpperTriangular());
	Eigen::eigen_assert(C.rows() == A.rows());
	Eigen::eigen_assert(C.cols() == B.rows());
  */  
      typedef typename MatrixType::Index Index;
      typedef typename MatrixType::Scalar Scalar;
    
      Index m = A.rows();
      Index n = B.rows();
      MatrixType X(m, n);
    
      for (Index i = m - 1; i >= 0; --i) {
          for (Index j = 0; j < n; ++j) {
	        // Compute AX = \sum_{k=i+1}^m A_{ik} X_{kj}
		Scalar AX;
		if (i == m - 1) {
		    AX = 0; 
		} 
		else {
		    Eigen::Matrix<Scalar,1,1> AXmatrix = A.row(i).tail(m-1-i) * X.col(j).tail(m-1-i);
		    AX = AXmatrix(0,0);
		}

		// Compute XB = \sum_{k=1}^{j-1} X_{ik} B_{kj}
		Scalar XB;
		if (j == 0) {
		    XB = 0; 
		} 
		else {
		    Eigen::Matrix<Scalar,1,1> XBmatrix = X.row(i).head(j) * B.col(j).head(j);
		    XB = XBmatrix(0,0);
		}

		X(i,j) = (C(i,j) - AX - XB) / (A(i,i) + B(j,j));
	}   
    }
result = X;
}


template <typename DerivedX, typename DerivedY, typename DerivedZ>
void MatrixLogarithmFrechetDerivative(const Eigen::MatrixBase<DerivedX>& X, const Eigen::MatrixBase<DerivedY>& E, Eigen::MatrixBase<DerivedZ>& result){

    // Matrix Parameters
    typedef Eigen::internal::traits<DerivedX> Traits;
    typedef typename Traits::Scalar Scalar;
    static const int Rows = Traits::RowsAtCompileTime, Cols = Traits::ColsAtCompileTime;
    static const int Options = DerivedX::Options;
    static const int MaxRows = Traits::MaxRowsAtCompileTime, MaxCols = Traits::MaxColsAtCompileTime;
 
    // Switch to complex arithmetic
    typedef std::complex<Scalar> ComplexScalar;
    typedef Eigen::Matrix<ComplexScalar, Rows, Cols, Options, MaxRows, MaxCols> ComplexMatrix;

    ComplexMatrix CX = X.template cast<ComplexScalar>();
    ComplexMatrix CE = E.template cast<ComplexScalar>();
    ComplexMatrix CResult;

    // Number of Square roots
    const int s = 4;
    // Order of the Pade approximant
    const int m = 7;

    // Complex Schur Decomposition
    const Eigen::ComplexSchur<ComplexMatrix> SchurOfX(CX);
    ComplexMatrix T = SchurOfX.matrixT();
    ComplexMatrix U = SchurOfX.matrixU();    

    ComplexMatrix sqrtT;
    CE = U.adjoint()*CE*U;

    for(int k=0; k<s; k++){
	Eigen::MatrixSquareRootTriangular<ComplexMatrix>(T).compute(sqrtT);
	T=sqrtT;
	SolveTriangularSylvester(T, T, CE, CE);
    }

    const double nodes[]   = { 0.0254460438286207377369051579760744L, 0.1292344072003027800680676133596058L,
            0.2970774243113014165466967939615193L, 0.5000000000000000000000000000000000L,
            0.7029225756886985834533032060384807L, 0.8707655927996972199319323866403942L,
            0.9745539561713792622630948420239256L };
    const double weights[] = { 0.0647424830844348466353057163395410L, 0.1398526957446383339507338857118898L,
              0.1909150252525594724751848877444876L, 0.2089795918367346938775510204081633L,
              0.1909150252525594724751848877444876L, 0.1398526957446383339507338857118898L,
              0.0647424830844348466353057163395410L };

    ComplexMatrix TminusI = T - ComplexMatrix::Identity(T.rows(), T.rows());
    CResult.setZero();
    
    for (int i = 0; i < m; ++i){
  	ComplexMatrix IplusBetaTm = ComplexMatrix::Identity(T.rows(), T.rows()) + nodes[i] * TminusI;
  	ComplexMatrix X = IplusBetaTm.template triangularView< Eigen::Upper >().solve(CE).transpose();	
  	CResult += weights[i] * (IplusBetaTm.template triangularView< Eigen::Upper >().transpose().solve(X)).transpose();	
    }

    CResult = std::pow(2.0,s) * U * CResult * U.adjoint();

    result = CResult.real();
}


template <typename DerivedX, typename DerivedY>
void KroneckerDLog(const Eigen::MatrixBase<DerivedX>& X, Eigen::MatrixBase<DerivedY>& Result){
    
    typedef Eigen::internal::traits<DerivedX> Traits;
    typedef typename Traits::Scalar Scalar;
    static const int Rows = Traits::RowsAtCompileTime;

    DerivedX E, PartialDiff; 
    DerivedY R;

    for (int i = 0; i < Rows; i++) {
    	for (int j = 0; j < Rows; j++) {
	    E = DerivedX::Zero();
	    E(i,j) = 1.0;
	    MatrixLogarithmFrechetDerivative(X, E, PartialDiff);
	    PartialDiff.transposeInPlace();
	    R.row(i*Rows+j) = Eigen::Map<Eigen::VectorXd>(PartialDiff.data(), PartialDiff.size());
    	}
    }
    Result = R;
}




} // end namespace tvmtl


  
#endif
