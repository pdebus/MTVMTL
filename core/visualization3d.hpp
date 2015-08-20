#ifndef TVMTL_VISUALIZATION3D_HPP
#define TVMTL_VISUALIZATION3D_HPP

//System includes
#include <iostream>
#include <cmath>
#include <string>

//Eigen includes
#include <Eigen/Geometry>

//OpenCV includes
#include <opencv2/opencv.hpp>

//OpenGL includes
#include <GL/freeglut.h>

// video++ includes
#include <vpp/vpp.hh>

// own includes
#include "enumerators.hpp"
#include "func_ptr_utils.hpp"
#include "manifold.hpp"
#include "data.hpp"


namespace tvmtl{

// Specialization SO(3) 
template <class DATA>
class Visualization< SO, 3, DATA, 3>{
    public:
        typedef Manifold<SO,3> mf_t;
	typedef Visualization<SO,3, DATA, 3> myType;

	// Constructor
	Visualization(DATA& dat): data_(dat), filename_(""), paint_inpainted_pixel_(false), cam_(Camera()) {}

	// Static members GL Interface
	
	// Class members GL Intergace
	void GLInit(const char* windowname);
	void reshape(int x, int y);
	void draw(void);
	void saveImage(std::string filename);
	void keyboard(unsigned char key, int x, int y);
	void specialKeys(int key, int x, int y);
	//
	//Acces methods
	void paint_inpainted_pixel(bool setFlag) {paint_inpainted_pixel_ = setFlag; }

    private:
	void storeImage(std::string filename);

	DATA& data_;
	bool paint_inpainted_pixel_;
	std::string filename_;
	int width_, height_;
	
	Camera cam_;

};

// Specialization EUCLIDIAN 
template <class DATA, int N>
class Visualization< EUCLIDIAN, N, DATA, 3>{
    public:
        typedef Manifold<EUCLIDIAN, N> mf_t;
	typedef Visualization<EUCLIDIAN, N, DATA, 3> myType;

	// Constructor
	Visualization(DATA& dat): data_(dat), filename_(""), paint_inpainted_pixel_(false), cam_(Camera()) {}

	// Static members GL Interface
	
	// Class members GL Intergace
	void GLInit(const char* windowname);
	void initTextures();
	void reshape(int x, int y);
	void animation();
	void draw(void);
	void saveImage(std::string filename);
	void keyboard(unsigned char key, int x, int y);
	void specialKeys(int key, int x, int y);
	//
	//Acces methods
	void paint_inpainted_pixel(bool setFlag) {paint_inpainted_pixel_ = setFlag; }

    private:
	void storeImage(std::string filename);

	DATA& data_;
	bool paint_inpainted_pixel_;
	std::string filename_;
	int width_, height_;
	
	Camera cam_;

	GLuint texture3d_id;
};

// Specialization SPD(3) 
template <class DATA>
class Visualization< SPD, 3, DATA, 3>{
    public:
        typedef Manifold<SPD,3> mf_t;
	typedef Visualization<SPD,3, DATA, 3> myType;

	// Constructor
	Visualization(DATA& dat): data_(dat), filename_(""), paint_inpainted_pixel_(false), cam_(Camera()) {}

	// Static members GL Interface
	static void initLighting();

	// Class members GL Interface
	void GLInit(const char* windowname);
	void reshape(int x, int y);
	void draw(void);
	void saveImage(std::string filename);
	void keyboard(unsigned char key, int x, int y);
	void specialKeys(int key, int x, int y);

	//Acces methods
	void paint_inpainted_pixel(bool setFlag) {paint_inpainted_pixel_ = setFlag; }

    private:
	void storeImage(std::string filename);

	DATA& data_;
	bool paint_inpainted_pixel_;
	std::string filename_;
	int width_, height_;

	Camera cam_;
};


// IMPLEMENTATION SO(3)
template <class DATA>
void Visualization<SO, 3, DATA, 3>::draw(void)
{

    glMatrixMode(GL_MODELVIEW);
    // clear the drawing buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS); 

    int nz = data_.img_.nslices();
    int nx = data_.img_.ncols();
    int ny = data_.img_.nrows();

   int max = std::max(nx,std::max(ny,nz));

    float scaling = 2.0 / (3.0 * max);
    float spacing = 4.0 * scaling;
    float z_distance= -4.5;
    scaling *= 0.5;

    #ifdef TV_VISUAL_DEBUG
	std::cout << "ny: " << ny << " nx: " << nx << std::endl;
	std::cout << "Maximum: " << max << " Scaling " << scaling << std::endl;
    #endif
 
    glLoadIdentity();
    gluLookAt(cam_.xPos_, cam_.yPos_, cam_.zPos_, cam_.xDir_, cam_.yDir_, cam_.zDir_, 0.0f, 1.0f, 0.0f);

    glTranslatef(-1.0, -1.25, z_distance); // Left->Right, Bottom->Top
   
    for(int s = 0; s < nz; ++s){
	for(int r = 0; r < ny; ++r){
	// Start of row pointers
	const typename mf_t::value_type* v = &data_.img_(s, r, 0);
	const typename DATA::inp_type* i = &data_.inp_(s, r, 0);
	for(int c = 0; c < nx; ++c){
	    if(paint_inpainted_pixel_ || !i[c]){
		glPushMatrix();
	        glTranslatef(c * spacing, r * spacing, s * spacing); //Left->Right, Bottom->Top 

		Eigen::Affine3f t = Eigen::Affine3f::Identity();
		t.linear() = v[c].cast<float>();

		#ifdef TV_VISUAL_DEBUG
		    std::cout << "Transformation matrix:\n" << t.matrix() << std::endl;
		#endif

		glMultMatrixf(t.data());

		glScalef(scaling, scaling, scaling);
	    
		glBegin(GL_QUADS);        // Draw The Cube Using quads
		glColor3f(0.0f,1.0f,0.0f);    // Color Blue
		glVertex3f( 1.0f, 1.0f,-1.0f);    // Top Right Of The Quad (Top)
		glVertex3f(-1.0f, 1.0f,-1.0f);    // Top Left Of The Quad (Top)
		glVertex3f(-1.0f, 1.0f, 1.0f);    // Bottom Left Of The Quad (Top)
		glVertex3f( 1.0f, 1.0f, 1.0f);    // Bottom Right Of The Quad (Top)
		glColor3f(1.0f,0.5f,0.0f);    // Color Orange
		glVertex3f( 1.0f,-1.0f, 1.0f);    // Top Right Of The Quad (Bottom)
		glVertex3f(-1.0f,-1.0f, 1.0f);    // Top Left Of The Quad (Bottom)
		glVertex3f(-1.0f,-1.0f,-1.0f);    // Bottom Left Of The Quad (Bottom)
		glVertex3f( 1.0f,-1.0f,-1.0f);    // Bottom Right Of The Quad (Bottom)
		glColor3f(1.0f,0.0f,0.0f);    // Color Red    
		glVertex3f( 1.0f, 1.0f, 1.0f);    // Top Right Of The Quad (Front)
		glVertex3f(-1.0f, 1.0f, 1.0f);    // Top Left Of The Quad (Front)
		glVertex3f(-1.0f,-1.0f, 1.0f);    // Bottom Left Of The Quad (Front)
		glVertex3f( 1.0f,-1.0f, 1.0f);    // Bottom Right Of The Quad (Front)
		glColor3f(1.0f,1.0f,0.0f);    // Color Yellow
		glVertex3f( 1.0f,-1.0f,-1.0f);    // Top Right Of The Quad (Back)
		glVertex3f(-1.0f,-1.0f,-1.0f);    // Top Left Of The Quad (Back)
		glVertex3f(-1.0f, 1.0f,-1.0f);    // Bottom Left Of The Quad (Back)
		glVertex3f( 1.0f, 1.0f,-1.0f);    // Bottom Right Of The Quad (Back)
		glColor3f(0.0f,0.0f,1.0f);    // Color Blue
		glVertex3f(-1.0f, 1.0f, 1.0f);    // Top Right Of The Quad (Left)
		glVertex3f(-1.0f, 1.0f,-1.0f);    // Top Left Of The Quad (Left)
		glVertex3f(-1.0f,-1.0f,-1.0f);    // Bottom Left Of The Quad (Left)
		glVertex3f(-1.0f,-1.0f, 1.0f);    // Bottom Right Of The Quad (Left)
		glColor3f(1.0f,0.0f,1.0f);    // Color Violet
		glVertex3f( 1.0f, 1.0f,-1.0f);    // Top Right Of The Quad (Right)
		glVertex3f( 1.0f, 1.0f, 1.0f);    // Top Left Of The Quad (Right)
		glVertex3f( 1.0f,-1.0f, 1.0f);    // Bottom Left Of The Quad (Right)
		glVertex3f( 1.0f,-1.0f,-1.0f);    // Bottom Right Of The Quad (Right)
		glEnd();            // End Drawing The Cube
		glPopMatrix();
	    }
	}
    }
}
    glutSwapBuffers();
}




template <class DATA>
void Visualization<SO, 3, DATA, 3>::reshape(int x, int y)
{
    if (y == 0 || x == 0) return;  //Nothing is visible then, so return

    width_ = x;
    height_ = y;

    glMatrixMode(GL_PROJECTION);  
    glLoadIdentity();
    gluPerspective(45.0,(GLdouble)x/(GLdouble)y,0.1,100.0);
    glViewport(0.0, 0.0 , x, y);  //Use the whole window for rendering
}

template <class DATA>
/* Callback handler for normal-key event */
void Visualization<SO, 3, DATA, 3>::keyboard(unsigned char key, int x, int y) {
   switch (key) {
         case 27:     // ESC key
	 case 113:	// q ley  
	    glutLeaveMainLoop();
            break;
	case 115:
	    if(filename_ != "") storeImage(filename_);
	    break;
    }
}


template <class DATA>
/* Callback handler for special-key event */
void Visualization<SO, 3, DATA, 3>::specialKeys(int key, int x, int y) {
   #ifdef TV_VISUAL_CONTROLS_DEBUG
	std::cout << "BEFORE keypress: ";
	cam_.printParams();
    #endif
	switch (key) {
	case GLUT_KEY_RIGHT:
	    cam_.yAngle_ += 0.01f;
	    cam_.xDir_ = std::sin(cam_.yAngle_);
	    cam_.zDir_ = -std::cos(cam_.yAngle_);
	    break;
	case GLUT_KEY_LEFT:
	    cam_.yAngle_ -= 0.01f;
	    cam_.xDir_ = std::sin(cam_.yAngle_);
	    cam_.zDir_ = -std::cos(cam_.yAngle_);
	    break;
	case GLUT_KEY_UP:
	    cam_.xPos_ += cam_.xDir_ * cam_.step_;
	    cam_.yPos_ += cam_.yDir_ * cam_.step_;
	    cam_.zPos_ += cam_.zDir_ * cam_.step_;
	    break;
	case GLUT_KEY_DOWN:
	    cam_.xPos_ -= cam_.xDir_ * cam_.step_;
	    cam_.yPos_ -= cam_.yDir_ * cam_.step_;
	    cam_.zPos_ -= cam_.zDir_ * cam_.step_;
	    break;
	case GLUT_KEY_PAGE_UP:
	    break;
	case GLUT_KEY_PAGE_DOWN:
	    break;
      }

    #ifdef TV_VISUAL_CONTROLS_DEBUG
	std::cout << "AFTER keypress: ";
	cam_.printParams();
    #endif
  
}



template <class DATA>
void Visualization<SO, 3, DATA, 3>::saveImage(std::string filename){
    filename_ = filename;
}

template <class DATA>
void Visualization<SO, 3, DATA, 3>::storeImage(std::string filename){
    cv::Mat img(height_, width_, CV_8UC3);
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3)?1:4);
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());
    glReadPixels(0, 0, img.cols, img.rows, GL_BGR_EXT, GL_UNSIGNED_BYTE, img.data);
    cv::Mat flipped(img);
    cv::flip(img, flipped, 0);
    cv::imwrite(filename, img);
}

template <class DATA>
void Visualization<SO, 3, DATA, 3>::GLInit(const char* window_name){

    int argc = 1;
    char** argv=0;
    glutInit(&argc, argv);

    //we initialilze the glut. functions
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB| GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(window_name);
    
    glClearColor(1.0,1.0,1.0,0);
    
    glutDisplayFunc(getCFunctionPointer(&myType::draw,this));
    glutIdleFunc(getCFunctionPointer(&myType::draw,this));
    
	typedef void (*reshape_mptr)(int, int);
	Callback<void(int, int)>::func = std::bind(&myType::reshape, this, std::placeholders::_1, std::placeholders::_2);
	reshape_mptr reshape_ptr = static_cast<reshape_mptr>(Callback<void(int, int)>::callback);
    glutReshapeFunc(reshape_ptr);

	typedef void (*keyboard_mptr)(unsigned char, int, int);
	Callback<void(unsigned char, int, int)>::func = std::bind(&myType::keyboard, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	keyboard_mptr keyboard_ptr = static_cast<keyboard_mptr>(Callback<void(unsigned char, int, int)>::callback);
    glutKeyboardFunc(keyboard_ptr);

	typedef void (*specialkeyboard_mptr)(int, int, int);
	Callback<void(int, int, int)>::func = std::bind(&myType::specialKeys, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	specialkeyboard_mptr skeyboard_ptr = static_cast<specialkeyboard_mptr>(Callback<void(int, int, int)>::callback);
    glutSpecialFunc(skeyboard_ptr);


    glutMainLoop();
} 

// IMPLEMENTATION SPD

template <class DATA>
void Visualization<SPD, 3, DATA, 3>::initLighting()
{

    // Set lighting intensity and color
    GLfloat light_ambient[]  = { 0.0f, 0.0f, 0.0f, 1.0f };
    GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    // Light source position
    GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };  
    
    // Enable lighting
    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);

     // Set lighting intensity and color
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    
    // Set the light position
     glLightfv(GL_LIGHT0, GL_POSITION, light_position);

}

template <class DATA>
void Visualization<SPD, 3, DATA, 3>::draw(void)
{
    // Color Definitions
    GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
    GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
    GLfloat mat_specular[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat high_shininess =  100.0f; 

    glMatrixMode(GL_MODELVIEW);
    // clear the drawing buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS); 

    // Set material properties
    glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialf(GL_FRONT, GL_SHININESS, high_shininess);

    int nz=data_.img_.nslices();
    int nx=data_.img_.ncols();
    int ny=data_.img_.nrows();
    
    int max = std::max(nx,std::max(ny,nz)); 
   
    float scaling = 2.0 / (3.0 * max);
    float spacing = 4.0 * scaling;

    float z_distance= -4.5;
    scaling *=0.5;

    #ifdef TV_VISUAL_DEBUG
	std::cout << "ny: " << ny << " nx: " << nx << std::endl;
	std::cout << "Maximum: " << max << " Scaling " << scaling << std::endl;
    #endif
 
    glLoadIdentity();
   
    gluLookAt(cam_.xPos_, cam_.yPos_, cam_.zPos_, cam_.xDir_, cam_.yDir_, cam_.zDir_, 0.0f, 1.0f, 0.0f);


    //glTranslatef(-1.2, 1.2, z_distance); // Left->Right, Top->Bottom
    glTranslatef(-1.0, -1.25, z_distance); // Left->Right, Bottom->Top

    for(int s = 0; s < nz; ++s){
	for(int r = 0; r < ny; ++r){
	    // Start of row pointers
	    const typename mf_t::value_type* v = &data_.img_(s, r, 0);
	    const typename DATA::inp_type* i = &data_.inp_(s, r, 0);
	    for(int c = 0; c < nx; ++c){
		if(paint_inpainted_pixel_ || !i[c]){
		    glPushMatrix();

		    //glTranslatef(coords(1) * spacing, -coords(0) * spacing, 0.0); //Left->Right, Top->Bottom
		    glTranslatef(c * spacing, r * spacing, s * spacing); //Left->Right, Bottom->Top
		    
		    
		    Eigen::Affine3f t = Eigen::Affine3f::Identity();
		    Eigen::SelfAdjointEigenSolver<typename mf_t::value_type> es(v[c]);
		
		    double mean_diffusity = es.eigenvalues().sum() / 3.0;

		    // anisotropic scaling transfomation and rotation
		    t.linear() = (es.eigenvectors() * (es.eigenvalues() / mean_diffusity).asDiagonal()).cast<float>();
		    //t.linear() = es.eigenvalues().cast<float>().asDiagonal(); 
		    glMultMatrixf(t.data());
		   
		    #ifdef TV_VISUAL_DEBUG
			std::cout << "Transformation matrix:\n" << t.matrix() << std::endl;
		    #endif
		
		    // isotropic scaling transformation
		    glScalef(5.0 * scaling, 5.0 * scaling, 5.0 * scaling);
		    
		    int argmax;
		    es.eigenvalues().maxCoeff(&argmax);
		    Eigen::Vector3d principal_direction = es.eigenvectors().col(argmax);
		    //principal_direction /= principal_direction.maxCoeff();	    
		    principal_direction.normalize();

		    glColor3d(std::abs(principal_direction(0)), std::abs(principal_direction(1)),std::abs(principal_direction(2)));
		    // draw sphere
		    glutSolidSphere(0.5, 35, 35);
		    
		    
		    // scaling to correct global size
		    glPopMatrix();
		}
	    
	    }
	}
    }
    glutSwapBuffers();
}

template <class DATA>
void Visualization<SPD, 3, DATA, 3>::reshape(int x, int y)
{
    if (y == 0 || x == 0) return;  //Nothing is visible then, so return
    
    width_ = x;
    height_ = y;
    
    glMatrixMode(GL_PROJECTION);  
    glLoadIdentity();
    gluPerspective(45.0,(GLdouble)x/(GLdouble)y,0.1,100.0);
    glViewport(0.0, 0.0 , x, y);  //Use the whole window for rendering
}

template <class DATA>
/* Callback handler for normal-key event */
void Visualization<SPD, 3, DATA, 3>::keyboard(unsigned char key, int x, int y) {
   switch (key) {
         case 27:	// ESC key
	 case 113:	// q ley
	    glutLeaveMainLoop();
            break;
	case 115:
	    if(filename_ != "") storeImage(filename_);
	    break;
      }
}

template <class DATA>
/* Callback handler for special-key event */
void Visualization<SPD, 3, DATA, 3>::specialKeys(int key, int x, int y) {
   #ifdef TV_VISUAL_CONTROLS_DEBUG
	std::cout << "BEFORE keypress: ";
	cam_.printParams();
    #endif
	switch (key) {
	case GLUT_KEY_RIGHT:
	    cam_.yAngle_ += 0.01f;
	    cam_.xDir_ = std::sin(cam_.yAngle_);
	    cam_.zDir_ = -std::cos(cam_.yAngle_);
	    break;
	case GLUT_KEY_LEFT:
	    cam_.yAngle_ -= 0.01f;
	    cam_.xDir_ = std::sin(cam_.yAngle_);
	    cam_.zDir_ = -std::cos(cam_.yAngle_);
	    break;
	case GLUT_KEY_UP:
	    cam_.xPos_ += cam_.xDir_ * cam_.step_;
	    cam_.yPos_ += cam_.yDir_ * cam_.step_;
	    cam_.zPos_ += cam_.zDir_ * cam_.step_;
	    break;
	case GLUT_KEY_DOWN:
	    cam_.xPos_ -= cam_.xDir_ * cam_.step_;
	    cam_.yPos_ -= cam_.yDir_ * cam_.step_;
	    cam_.zPos_ -= cam_.zDir_ * cam_.step_;
	    break;
	case GLUT_KEY_PAGE_UP:
	    break;
	case GLUT_KEY_PAGE_DOWN:
	    break;
      }

    #ifdef TV_VISUAL_CONTROLS_DEBUG
	std::cout << "AFTER keypress: ";
	cam_.printParams();
    #endif
  
}

template <class DATA>
void Visualization<SPD, 3, DATA, 3>::saveImage(std::string filename){
    filename_ = filename;
}

template <class DATA>
void Visualization<SPD, 3, DATA, 3>::storeImage(std::string filename)
{
    std::cout << "Saving image...\n Width: " << width_ << "\n Height: " << height_ << "\n Filename: " << filename << std::endl;
    cv::Mat img(height_, width_, CV_8UC3);
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3)?1:4);
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());
    glReadPixels(0, 0, img.cols, img.rows, GL_BGR_EXT, GL_UNSIGNED_BYTE, img.data);
    cv::Mat flipped(img);
    cv::flip(img, flipped, 0);
    cv::imwrite(filename, img);
}

template <class DATA>
void Visualization<SPD, 3, DATA, 3>::GLInit(const char* window_name){

    int argc = 1;
    char** argv=0;
    glutInit(&argc, argv);

    //we initialilze the glut. functions
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    glutInitDisplayMode(GLUT_DOUBLE| GLUT_RGB| GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(window_name);
    initLighting(); 
    
    glClearColor(1.0,1.0,1.0,0);
    
    glutDisplayFunc(getCFunctionPointer(&myType::draw,this));
    glutIdleFunc(getCFunctionPointer(&myType::draw,this));

    typedef void (*reshape_mptr)(int, int);
    Callback<void(int, int)>::func = std::bind(&myType::reshape, this, std::placeholders::_1, std::placeholders::_2);
    reshape_mptr reshape_ptr = static_cast<reshape_mptr>(Callback<void(int, int)>::callback);
	glutReshapeFunc(reshape_ptr);
    
    typedef void (*keyboard_mptr)(unsigned char, int, int);
    Callback<void(unsigned char, int, int)>::func = std::bind(&myType::keyboard, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    keyboard_mptr keyboard_ptr = static_cast<keyboard_mptr>(Callback<void(unsigned char, int, int)>::callback);
	    glutKeyboardFunc(keyboard_ptr);

    typedef void (*specialkeyboard_mptr)(int, int, int);
    Callback<void(int, int, int)>::func = std::bind(&myType::specialKeys, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    specialkeyboard_mptr skeyboard_ptr = static_cast<specialkeyboard_mptr>(Callback<void(int, int, int)>::callback);
	    glutSpecialFunc(skeyboard_ptr);

    glutMainLoop();
}

//Implementation EUCLIDIAN
//

template <class DATA, int N>
void Visualization<EUCLIDIAN, N, DATA, 3>::initTextures(){

    int nz = data_.img_.nslices();
    int ny = data_.img_.nrows();
    int nx = data_.img_.ncols();
    
    int size = nz * ny * nx;
    char* RGBAbuffer = new char[size * 4];

    static_assert(N == 3 || N == 1, "Cube Visualization only possible for N=1 or N=3");

    int k = 0;
    if(N == 3){
	for(auto& p : data_.img_){
	    typename mf_t::value_type v = p * (double) std::numeric_limits<unsigned char>::max();
	    //std::cout << "Image Vector:\n" << p << std::endl;
	    //std::cout << "Scaling Factor: " << (int) std::numeric_limits<unsigned char>::max() << std::endl;
	    //std::cout << "Scaled Vector:\n " << v << std::endl;
	    double avg = v.sum() /3; 
	    RGBAbuffer[k * 4]	= static_cast<char>(v[0]);
	    RGBAbuffer[k * 4 + 1]	= static_cast<char>(v[1]);
	    RGBAbuffer[k * 4 + 2]	= static_cast<char>(v[2]);
	    RGBAbuffer[k * 4 + 3]	= static_cast<char>(avg); 
	    //for(int i = 0; i<4; ++i) std::cout << (int) RGBAbuffer[k*4 + i] << ", ";
	    //std::cout << std::endl;
	    ++k;
	}
    }
    else{
	for(auto& p : data_.img_){
	    double  v = p[0] * (double) std::numeric_limits<unsigned char>::max();
	    RGBAbuffer[k * 4]		= static_cast<char>(v);
	    RGBAbuffer[k * 4 + 1]	= static_cast<char>(v);
	    RGBAbuffer[k * 4 + 2]	= static_cast<char>(v);
	    RGBAbuffer[k * 4 + 3]	= static_cast<char>(v); 
	    ++k;
	}
    }

    glGenTextures(1, (GLuint*) &texture3d_id);
    glBindTexture( GL_TEXTURE_3D, texture3d_id );
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
 
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, nx, ny, nz, 0, GL_RGBA, GL_UNSIGNED_BYTE, RGBAbuffer );
    glBindTexture( GL_TEXTURE_3D, 0 );

    delete[] RGBAbuffer;
}


template <class DATA, int N>
void Visualization<EUCLIDIAN, N, DATA, 3>::draw(void)
{

    float dOrthoSize = 1.0;

    glMatrixMode(GL_MODELVIEW);
    // clear the drawing buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    // Enable depth test
    //glEnable(GL_DEPTH_TEST);
    //glDepthFunc(GL_LESS); 

    glEnable(GL_ALPHA_TEST );
    glAlphaFunc(GL_GREATER, 0.05f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    int nz = data_.img_.nslices();
    int nx = data_.img_.ncols();
    int ny = data_.img_.nrows();

    #ifdef TV_VISUAL_DEBUG
	std::cout << "ny: " << ny << " nx: " << nx << std::endl;
    #endif

    glMatrixMode(GL_TEXTURE); 
    glLoadIdentity();
    glTranslatef(0.5f, 0.5f, 0.5f);
    glScaled( (float)nx /(float) nx, -1.0f*(float) nx/(float)ny, (float)nx/(float)nz);
    glRotated(cam_.yAngle_, 0.0, 1.0, 0.0);
    glTranslatef(-0.5f, -0.5f, -0.5f);

    //gluLookAt(cam_.xPos_, cam_.yPos_, cam_.zPos_, cam_.xDir_, cam_.yDir_, cam_.zDir_, 0.0f, 1.0f, 0.0f);

    glEnable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, texture3d_id);

    for (float fIndx = -1.0f; fIndx <= 1.0f; fIndx+=0.01f ){
	glBegin(GL_QUADS);
	glTexCoord3f(0.0f, 0.0f, ((float)fIndx+1.0f)/2.0f);  
        glVertex3f(-dOrthoSize,-dOrthoSize,fIndx);
        glTexCoord3f(1.0f, 0.0f, ((float)fIndx+1.0f)/2.0f);  
        glVertex3f(dOrthoSize,-dOrthoSize,fIndx);
        glTexCoord3f(1.0f, 1.0f, ((float)fIndx+1.0f)/2.0f);  
        glVertex3f(dOrthoSize,dOrthoSize,fIndx);
        glTexCoord3f(0.0f, 1.0f, ((float)fIndx+1.0f)/2.0f); 
        glVertex3f(-dOrthoSize,dOrthoSize,fIndx);
	glEnd();
    }
 
    glutSwapBuffers();
}




template <class DATA, int N>
void Visualization<EUCLIDIAN, N, DATA, 3>::reshape(int x, int y)
{
    width_ = x;    height_ = y;
    GLfloat dOrthoSize = 1.0f;
    GLdouble AspectRatio = ( GLdouble )(x) / ( GLdouble )(y);

    glViewport(0.0, 0.0 , x, y);  //Use the whole window for rendering
    glMatrixMode(GL_PROJECTION);  
    glLoadIdentity();

    if( x <= y )
	glOrtho( -dOrthoSize, dOrthoSize, -( dOrthoSize / AspectRatio ) ,dOrthoSize / AspectRatio, 2.0f*-dOrthoSize, 2.0f*dOrthoSize );
    else
	glOrtho( -dOrthoSize * AspectRatio, dOrthoSize * AspectRatio, -dOrthoSize, dOrthoSize, 2.0f*-dOrthoSize, 2.0f*dOrthoSize );
    
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
}

template <class DATA, int N>
/* Callback handler for normal-key event */
void Visualization<EUCLIDIAN, N, DATA, 3>::keyboard(unsigned char key, int x, int y) {
   switch (key) {
         case 27:     // ESC key
	 case 113:	// q ley  
	    glutLeaveMainLoop();
            break;
	case 115:
	    if(filename_ != "") storeImage(filename_);
	    break;
    }
}


template <class DATA, int N>
/* Callback handler for special-key event */
void Visualization<EUCLIDIAN, N, DATA, 3>::specialKeys(int key, int x, int y) {
   #ifdef TV_VISUAL_CONTROLS_DEBUG
	std::cout << "BEFORE keypress: ";
	cam_.printParams();
    #endif
	switch (key) {
	case GLUT_KEY_RIGHT:
	    cam_.yAngle_ += 0.01f;
	    cam_.xDir_ = std::sin(cam_.yAngle_);
	    cam_.zDir_ = -std::cos(cam_.yAngle_);
	    break;
	case GLUT_KEY_LEFT:
	    cam_.yAngle_ -= 0.01f;
	    cam_.xDir_ = std::sin(cam_.yAngle_);
	    cam_.zDir_ = -std::cos(cam_.yAngle_);
	    break;
	case GLUT_KEY_UP:
	    cam_.xPos_ += cam_.xDir_ * cam_.step_;
	    cam_.yPos_ += cam_.yDir_ * cam_.step_;
	    cam_.zPos_ += cam_.zDir_ * cam_.step_;
	    break;
	case GLUT_KEY_DOWN:
	    cam_.xPos_ -= cam_.xDir_ * cam_.step_;
	    cam_.yPos_ -= cam_.yDir_ * cam_.step_;
	    cam_.zPos_ -= cam_.zDir_ * cam_.step_;
	    break;
	case GLUT_KEY_PAGE_UP:
	    break;
	case GLUT_KEY_PAGE_DOWN:
	    break;
      }

    #ifdef TV_VISUAL_CONTROLS_DEBUG
	std::cout << "AFTER keypress: ";
	cam_.printParams();
    #endif
  
}


template <class DATA, int N>
void Visualization<EUCLIDIAN, N, DATA, 3>::animation(){
    cam_.yAngle_+=0.50;
    draw();
}


template <class DATA, int N>
void Visualization<EUCLIDIAN, N, DATA, 3>::saveImage(std::string filename){
    filename_ = filename;
}

template <class DATA, int N>
void Visualization<EUCLIDIAN, N, DATA, 3>::storeImage(std::string filename){
    cv::Mat img(height_, width_, CV_8UC3);
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3)?1:4);
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());
    glReadPixels(0, 0, img.cols, img.rows, GL_BGR_EXT, GL_UNSIGNED_BYTE, img.data);
    cv::Mat flipped(img);
    cv::flip(img, flipped, 0);
    cv::imwrite(filename, img);
}

template <class DATA, int N>
void Visualization<EUCLIDIAN, N, DATA, 3>::GLInit(const char* window_name){

    int argc = 1;
    char** argv=0;
    glutInit(&argc, argv);

    //we initialilze the glut. functions
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(window_name);
    
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glewInit();
    initTextures();

    glutDisplayFunc(getCFunctionPointer(&myType::draw,this));
    glutIdleFunc(getCFunctionPointer(&myType::animation,this));
    
	typedef void (*reshape_mptr)(int, int);
	Callback<void(int, int)>::func = std::bind(&myType::reshape, this, std::placeholders::_1, std::placeholders::_2);
	reshape_mptr reshape_ptr = static_cast<reshape_mptr>(Callback<void(int, int)>::callback);
    glutReshapeFunc(reshape_ptr);

	typedef void (*keyboard_mptr)(unsigned char, int, int);
	Callback<void(unsigned char, int, int)>::func = std::bind(&myType::keyboard, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	keyboard_mptr keyboard_ptr = static_cast<keyboard_mptr>(Callback<void(unsigned char, int, int)>::callback);
    glutKeyboardFunc(keyboard_ptr);

	typedef void (*specialkeyboard_mptr)(int, int, int);
	Callback<void(int, int, int)>::func = std::bind(&myType::specialKeys, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	specialkeyboard_mptr skeyboard_ptr = static_cast<specialkeyboard_mptr>(Callback<void(int, int, int)>::callback);
    glutSpecialFunc(skeyboard_ptr);


    glutMainLoop();
}

} // end namespace tvmtl
#endif  
