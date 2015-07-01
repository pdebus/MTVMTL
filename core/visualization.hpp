#ifndef TVMTL_VISUALIZATION_HPP
#define TVMTL_VISUALIZATION_HPP

//System includes
#include <iostream>

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

// Primary Template
template < enum MANIFOLD_TYPE MF, int N, class DATA >
class Visualization {
};

// Specialization SO(3) 
template <class DATA>
class Visualization< SO, 3, DATA>{
    public:
        typedef Manifold<SO,3> mf_t;
	typedef Visualization<SO,3, DATA> myType;

	// Constructor
	Visualization(DATA& dat): data_(dat), paint_inpainted_pixel_(false) {}

	// Static members GL Interface
	
	// Class members GL Intergace
	void GLInit(const char* filename);
	void reshape(int x, int y);
	void draw(void);
	void saveImage(char* filename);
	void keyboard(unsigned char key, int x, int y);
	//
	//Acces methods
	void paint_inpainted_pixel(bool setFlag) {paint_inpainted_pixel_ = setFlag; }

    private:
	void storeImage(char* filename);

	DATA& data_;
	bool paint_inpainted_pixel_;
	char* filename_;
	int width_, height_;
};

// Specialization SPD(3) 
template <class DATA>
class Visualization< SPD, 3, DATA>{
    public:
        typedef Manifold<SPD,3> mf_t;
	typedef Visualization<SPD,3, DATA> myType;

	// Constructor
	Visualization(DATA& dat): data_(dat), filename_(0), paint_inpainted_pixel_(false) {}

	// Static members GL Interface
	static void initLighting();

	// Class members GL Interface
	void GLInit(const char* windowname);
	void reshape(int x, int y);
	void draw(void);
	void saveImage(char* filename);
	void keyboard(unsigned char key, int x, int y);

	//Acces methods
	void paint_inpainted_pixel(bool setFlag) {paint_inpainted_pixel_ = setFlag; }

    private:
	void storeImage(char* filename);

	DATA& data_;
	bool paint_inpainted_pixel_;
	char* filename_;
	int width_, height_;
};


// IMPLEMENTATION SO(3)
template <class DATA>
void Visualization<SO, 3, DATA>::draw(void)
{

    glMatrixMode(GL_MODELVIEW);
    // clear the drawing buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS); 

    int nx=data_.img_.ncols();
    int ny=data_.img_.nrows();
    
    int max =  nx ^ ((nx ^ ny) & -(nx < ny));
   
    float scaling = 2.0 / (3.0 * max);
    float spacing = 4.0 * scaling;
    float z_distance= -3.0;
    
    #ifdef TV_VISUAL_DEBUG
	std::cout << "ny: " << ny << " nx: " << nx << std::endl;
	std::cout << "Maximum: " << max << " Scaling " << scaling << std::endl;
    #endif
 
    glLoadIdentity();
    //glTranslatef(-1.1, 1.1, z_distance); // Left->Right, Top->Bottom
    glTranslatef(-1.1, -1.1, z_distance); // Left->Right, Bottom->Top

    auto cube_drawer = [&](const typename mf_t::value_type& v, const vpp::vint2& coords, const typename DATA::inp_type& i){
	if(paint_inpainted_pixel_ || !i){
	    glPushMatrix();
	    //glTranslatef(coords(1) * spacing, -coords(0) * spacing, 0.0); //Left->Right, Top->Bottom
	    glTranslatef(coords(1) * spacing, coords(0) * spacing, 0.0); //Left->Right, Bottom->Top
	    
	    #ifdef TV_VISUAL_DEBUG
		std::cout << "\ncoord0: " << coords(0) << " coord1 " << coords(1) << std::endl;
		std::cout << "translateX: " << coords(1)*spacing << " translateY: " << -coords(0)*spacing << std::endl;
	    #endif
    
	    Eigen::Affine3f t = Eigen::Affine3f::Identity();
	    t.linear() = v.cast<float>();

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
    };

    vpp::pixel_wise(data_.img_, data_.img_.domain(), data_.inp_)(vpp::_no_threads)| cube_drawer;

    glFlush();
}




template <class DATA>
void Visualization<SO, 3, DATA>::reshape(int x, int y)
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
void Visualization<SO, 3, DATA>::keyboard(unsigned char key, int x, int y) {
   switch (key) {
         case 27:     // ESC key
	 case 113:	// q ley  
	    if(filename_!=0) storeImage(filename_);
	    glutLeaveMainLoop();
            break;
      }
}

template <class DATA>
void Visualization<SO, 3, DATA>::saveImage(char* filename){
    filename_ = filename;
}

template <class DATA>
void Visualization<SO, 3, DATA>::storeImage(char* filename){
    cv::Mat img(height_, width_, CV_8UC3);
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3)?1:4);
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());
    glReadPixels(0, 0, img.cols, img.rows, GL_BGR_EXT, GL_UNSIGNED_BYTE, img.data);
    cv::Mat flipped(img);
    cv::flip(img, flipped, 0);
    cv::imwrite(filename, img);
}

template <class DATA>
void Visualization<SO, 3, DATA>::GLInit(const char* window_name){

    int argc = 1;
    char** argv=0;
    glutInit(&argc, argv);

    //we initialilze the glut. functions
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutInitDisplayMode(GLUT_SINGLE| GLUT_RGB| GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(window_name);
    
    glClearColor(1.0,1.0,1.0,0);
    
    glutDisplayFunc(getCFunctionPointer(&myType::draw,this));
    
	typedef void (*reshape_mptr)(int, int);
	Callback<void(int, int)>::func = std::bind(&myType::reshape, this, std::placeholders::_1, std::placeholders::_2);
	reshape_mptr reshape_ptr = static_cast<reshape_mptr>(Callback<void(int, int)>::callback);
    glutReshapeFunc(reshape_ptr);

	typedef void (*keyboard_mptr)(unsigned char, int, int);
	Callback<void(unsigned char, int, int)>::func = std::bind(&myType::keyboard, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	keyboard_mptr keyboard_ptr = static_cast<keyboard_mptr>(Callback<void(unsigned char, int, int)>::callback);
    glutKeyboardFunc(keyboard_ptr);

    glutMainLoop();
} 

// IMPLEMENTATION SPD

template <class DATA>
void Visualization<SPD, 3, DATA>::initLighting()
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
void Visualization<SPD, 3, DATA>::draw(void)
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

    int nx=data_.img_.ncols();
    int ny=data_.img_.nrows();
    
    int max =  nx ^ ((nx ^ ny) & -(nx < ny));
   
    float scaling = 2.0 / (3.0 * max);
    float spacing = 4.0 * scaling;
    float z_distance= -4.0;
    
    #ifdef TV_VISUAL_DEBUG
	std::cout << "ny: " << ny << " nx: " << nx << std::endl;
	std::cout << "Maximum: " << max << " Scaling " << scaling << std::endl;
    #endif
 
    glLoadIdentity();
    
    //glTranslatef(-1.2, 1.2, z_distance); // Left->Right, Top->Bottom
    glTranslatef(-1.2, -1.2, z_distance); // Left->Right, Bottom->Top

    auto ellipsoid_drawer = [&](const typename mf_t::value_type& v, const vpp::vint2& coords, const typename DATA::inp_type& i){
	if(paint_inpainted_pixel_ || !i){
	    glPushMatrix();

	    //glTranslatef(coords(1) * spacing, -coords(0) * spacing, 0.0); //Left->Right, Top->Bottom
	    glTranslatef(coords(1) * spacing, coords(0) * spacing, 0.0); //Left->Right, Bottom->Top
	    
	    
	    Eigen::Affine3f t = Eigen::Affine3f::Identity();
	    Eigen::SelfAdjointEigenSolver<typename mf_t::value_type> es(v);
	    
	    // anisotropic scaling transfomation and rotation
	    t.linear() = (es.eigenvectors() * es.eigenvalues().asDiagonal() ).cast<float>();
	    //t.linear() = es.eigenvalues().cast<float>().asDiagonal(); 
	    glMultMatrixf(t.data());
	   
	    #ifdef TV_VISUAL_DEBUG
		std::cout << "Transformation matrix:\n" << t.matrix() << std::endl;
	    #endif
	
	    // isotropic scaling transformation
	    glScalef(5.0 * scaling, 5.0 * scaling, 5.0 * scaling);
	    
	    // built-in (glut library) function , draw you a sphere.
	    glColor3d(1.0, 0.0, 0.0);
	    glutSolidSphere(0.5, 35, 35);
	    
	    
	    // scaling to correct global size
	    glPopMatrix();
	}	
    };

    vpp::pixel_wise(data_.img_, data_.img_.domain(), data_.inp_)(vpp::_no_threads)| ellipsoid_drawer;

    glFlush();
}

template <class DATA>
void Visualization<SPD, 3, DATA>::reshape(int x, int y)
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
void Visualization<SPD, 3, DATA>::keyboard(unsigned char key, int x, int y) {
   switch (key) {
         case 27:	// ESC key
	 case 113:	// q ley
	    if(filename_!=0) storeImage(filename_);
	    glutLeaveMainLoop();
            break;
      }
}

template <class DATA>
void Visualization<SPD, 3, DATA>::saveImage(char* filename){
    filename_ = filename;
}

template <class DATA>
void Visualization<SPD, 3, DATA>::storeImage(char* filename)
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
void Visualization<SPD, 3, DATA>::GLInit(const char* window_name){

    int argc = 1;
    char** argv=0;
    glutInit(&argc, argv);

    //we initialilze the glut. functions
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    glutInitDisplayMode(GLUT_SINGLE| GLUT_RGB| GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(window_name);
    initLighting(); 
    
    glClearColor(1.0,1.0,1.0,0);
    
    glutDisplayFunc(getCFunctionPointer(&myType::draw,this));

	typedef void (*reshape_mptr)(int, int);
	Callback<void(int, int)>::func = std::bind(&myType::reshape, this, std::placeholders::_1, std::placeholders::_2);
	reshape_mptr reshape_ptr = static_cast<reshape_mptr>(Callback<void(int, int)>::callback);
    glutReshapeFunc(reshape_ptr);
    
    typedef void (*keyboard_mptr)(unsigned char, int, int);
    Callback<void(unsigned char, int, int)>::func = std::bind(&myType::keyboard, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    keyboard_mptr keyboard_ptr = static_cast<keyboard_mptr>(Callback<void(unsigned char, int, int)>::callback);
	    glutKeyboardFunc(keyboard_ptr);

    glutMainLoop();
}

} // end namespace tvmtl
#endif  
