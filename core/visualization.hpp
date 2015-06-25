#ifndef TVMTL_VISUALIZATION_HPP
#define TVMTL_VISUALIZATION_HPP

//System includes
#include <iostream>

//Eigen includes
#include <Eigen/Geometry>

//OpenGL includes
#include <GL/glut.h>

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
	Visualization(DATA& dat): data_(dat) {}

	// Static members GL Interface
	static void reshape(int x, int y);

	// Class members GL Intergace
	void GLInit(void);
	void draw(void);

    private:
	DATA& data_;
};

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
    
    glLoadIdentity();
    //glTranslatef(-nx * 0.5 * spacing + 1.0, ny*0.5*spacing - 1.0, z_distance);
    glTranslatef(-1.1, 1.1, z_distance);

    auto cube_drawer = [&](const typename mf_t::value_type& v, const vpp::vint2& coords, const typename DATA::inp_type& i){
	if(i){
	    glPushMatrix();
	    glTranslatef(coords(1) * spacing, -coords(0) * spacing, 0.0);
	    
	    Eigen::Affine3f t = Eigen::Affine3f::Identity();
	    t.linear() = v.cast<float>();
	    glLoadMatrixf(t.data());

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

    vpp::pixel_wise(data_.img_, data_.img_.domain(), data_.inp_mat)(vpp::_no_threads)| cube_drawer;

    glFlush();
}


template <class DATA>
void Visualization<SO, 3, DATA>::reshape(int x, int y)
{
    if (y == 0 || x == 0) return;  //Nothing is visible then, so return
    glMatrixMode(GL_PROJECTION);  
    glLoadIdentity();
    gluPerspective(45.0,(GLdouble)x/(GLdouble)y,0.1,100.0);
    glViewport(0.0, 0.0 , x, y);  //Use the whole window for rendering
}

template <class DATA>
void Visualization<SO, 3, DATA>::GLInit(const char* window_name){

    argc = 1;
    glutInit(&argc, window_name);

    //we initialilze the glut. functions
    glutInitDisplayMode(GLUT_SINGLE| GLUT_RGB| GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(window_name);
    
    glClearColor(1.0,1.0,1.0,0);
    
    glutDisplayFunc(getCFunctionPointer(&myType::draw,this));
    glutReshapeFunc(reshape);

    glutMainLoop();
} 






} // end namespace tvmtl
#endif  
