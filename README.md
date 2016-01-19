MTVMTL
=============
The manifold total variation minimization template library (MTVMTL) is an easy-to-use, fast C++14 template library for TV minimization of
manifold-valued two- or three-dimensional images.

# Capabilities

## Manifolds
- Real Euclidean space R n
- Sphere S n = x ∈ R n+1 : kxk = 1
- Special orthogonal group SO(n) = Q ∈ R n×n : QQ T = 1 , det(Q) = 1
- Symmetric positive definite matrices SP D(n) = S ∈ R n×n : S = S T , x T Sx > 0 ∀x ∈ R n \ {0}
- Grassmann manifold Gr(n, p) = St(n, p)/O(p)

## Data
- 2D and 3D images
- Input/Output via OpenCV integration supporting all common 2D image formats
- CSV input for matrix valued data
- Input methods for raw volume image data as well as the NIfTI [24] format for DT-MRI images
- Various methods to identify damaged areas for inpainting

## Functionals
- isotropic (only possible for IRLS) or anisotropic TV functionals
- first order TV term
- weighting and inpainting possible 

## Minimizer
- Iteratively reweighted least squares using Riemannian Newton method [1]
- Proximal point [2]

## Visualizations
- OpenGL rotated cubes visualization for SO(3) images
- OpenGL ellipsoid visualization for SPD(3) images
- OpenGL volume renderer for 3D volume images

# Prerequisites
The following list shows the needed packages for the usage of MTVMTL:
- CMake (≥ 2.8.0)
- gcc (≥ 4.9.1), any C++14 compatible compiler should also be possible but is untested.
- Eigen (≥ 3.2.5)
- Video++ (a modified version will be provided with the MTVMTL, otherwise consider check [https://github.com/matt-42/vpp](https://github.com/matt-42/vpp) )
- Boost (≥ 1.56) (also needed for CGAL)

Recommended are also the following packages. They are needed if any of the described extended
functionality needs to be used.
- OpenCV (≥ 2.4.9), for image input and output, edge detection for inpainting
- CGAL (≥ 4.3), for first guess interpolation during inpainting
- OpenGL (≥ 3.0), GLEW(≥ 1.10) and freeGLUT(≥ 2.8.1), for visualizations of SPD, SO and
any 3D data
- SuiteSparse (≥ 4.2.1), faster parallel sparse solver for the linear system in the IRLS algorithm
- SuperLU (≥ 4.3), faster parallel sparse solver for the linear system in the IRLS algorithm

[1] P. Grohs and M. Sprecher. Total variation regularization by iteratively reweighted least
squares on hadamard spaces and the sphere. Technical Report 2014-39, Seminar for Applied
Mathematics, ETH Zürich, Switzerland, 2014
[2] A. Weinmann, L. Demaret, and M. Storath. Total variation regularization for manifold-valued
data. SIAM Journal on Imaging Sciences, 7(4):2226–2257, 2014


# Usage
## Example: 3D DT-MRI data denoising and visualization

Required Headers
```c++
#include <mtvmtl/core/algo_traits.hpp>
#include <mtvmtl/core/data.hpp>
#include <mtvmtl/core/functional.hpp>
#include <mtvmtl/core/tvmin.hpp>
#include <mtvmtl/core/visualization.hpp>
```

Type Definitions
```c++
using namespace tvmtl;
typedef Manifold< SPD, 3 > mf_t;
typedef Data< mf_t, 3> data_t;
typedef Functional<FIRSTORDER, ANISO, mf_t, data_t, 3> func_t;
typedef TV_Minimizer< PRPT, func_t, mf_t, data_t, OMP, 3 > tvmin_t;
typedef Visualization<SPD, 3, data_t, 3> visual_t;
```

Display original (noisy) Data
```c++
data_t myData = data_t();   // Create data object
myData.readMatrixDataFromCSV(argv[1], nz, ny, nx); // Read from CSV file

visual_t myVisual(myData);  // Create visualization object
myVisual.saveImage(nfname); // Specify file name to save a screenshot

std::cout << "Starting OpenGL-Renderer..." << std::endl;
myVisual.GLInit("SPD(3) Ellipsoid Visualization"); // Start the Rendering
```

Denoise and display results
```c++
double lam=0.7;
func_t myFunc(lam, myData); // Functional object
myFunc.seteps2(0);  // eps^2 should be 0 for PRPT

tvmin_t myTVMin(myFunc, myData); // Minimizer object

std::cout << "Start TV minimization.." << std::endl;
myTVMin.minimize();

std::string dfname = "denoised(prpt)_" + fname.str();
myVisual.saveImage(dfname); // Specify name for denoised image

std::cout << "Starting OpenGL-Renderer..." << std::endl;
myVisual.GLInit("SPD(3) Ellipsoid Visualization"); // Render
```
