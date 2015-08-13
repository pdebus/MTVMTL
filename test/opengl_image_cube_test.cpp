#include <GL/freeglut.h>

#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>

 GLfloat xRotated, yRotated, zRotated;
 int ny,nx; 

void init(void)
{
glClearColor(1.0,1.0,1.0,0);
 
}

void DrawCube(void)
{

    glMatrixMode(GL_MODELVIEW);
    // clear the drawing buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS); 

    int max =  nx ^ ((nx ^ ny) & -(nx < ny));

    float scaling = 2.0 / (3.0 * max);
    float spacing = 4.0 * scaling;
    float z_distance= -3.0;
  std::cout << "ny: " << ny << " nx: " << nx << std::endl;
  std::cout << "Maximum: " << max << " Scaling " << scaling << std::endl;
  
    glLoadIdentity();
    //glTranslatef(-nx * 0.5 * spacing + 1.0, ny*0.5*spacing - 1.0, z_distance);
    glTranslatef(-1.1, 1.1, z_distance);

    for (int i=0; i<ny; i++)
	for(int j=0; j<nx; j++){
	    
	    glPushMatrix();
	    glTranslatef(j*spacing, -i*spacing, 0.0);
	    
	    glRotatef(xRotated,1.0,0.0,0.0);
	    // rotation about Y axis
	    glRotatef(yRotated*0.2*i,0.0,1.0,0.0);
	    // rotation about Z axis
	    glRotatef(zRotated,0.0,0.0,1.0);
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
glFlush();
}


void animation(void)
{
 
     yRotated += 0.01;
     xRotated += 0.02;
    DrawCube();
}

/* Callback handler for normal-key event */
void keyboard(unsigned char key, int x, int y) {
   switch (key) {
         case 27:     // ESC key
	    glutLeaveMainLoop();
            break;
      }
}

void reshape(int x, int y)
{
    if (y == 0 || x == 0) return;  //Nothing is visible then, so return
    //Set a new projection matrix
    glMatrixMode(GL_PROJECTION);  
    glLoadIdentity();
    //Angle of view:40 degrees
    //Near clipping plane distance: 0.5
    //Far clipping plane distance: 20.0
    
    float nx = 5*1.5;
    float ny = 4*1.5;
    
    gluPerspective(45.0,(GLdouble)x/(GLdouble)y,0.1,100.0);
    glViewport(0.0, 0.0 , x, y);  //Use the whole window for rendering
    
}

int main(int argc, char** argv){
    int nz = 30;
    int ny = 30;
    int nx = 30;

    std::string fname("test.raw");

    if(argc > 1){
	fname = argv[1];
	nz=atoi(argv[2]);
	ny=atoi(argv[3]);
	nx=atoi(argv[4]);
    }


    int pixel_num = nz * ny * nx;
    
    std::fstream file;
    file.open(fname, std::ios::in|std::ios::binary|std::ios::ate);

    std::streampos size;
    char* buffer;
    bool read_failure = true;

    std::cout << "Reading file " << fname << " with dimensions(Slices, Rows, Cols) " << nz << " X " << ny << " X " << nx << std::endl;

    if(file.is_open()){
	size = file.tellg();
	buffer = new char[size];
	file.seekg(0, std::ios::beg);
	file.read(buffer, size);
	file.close();
	read_failure = false;
    }

    if(read_failure){
	std::cout << "File import not successfull!" << std::endl;
	return 1;
	}

    std::cout << "File successfully imported!" << std::endl;

/*
    glutInit(&argc, argv);
    //we initizlilze the glut. functions
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutInitDisplayMode(GLUT_SINGLE| GLUT_RGB| GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(argv[0]);
    init();
    glutDisplayFunc(DrawCube);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    //Set the function for the animation.
    //glutIdleFunc(animation);
    glutMainLoop();

    std::cout << "After MainLoop\n";*/

    

    delete[] buffer;
    return 0;
} 






  
