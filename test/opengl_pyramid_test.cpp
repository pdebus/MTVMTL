#include <GL/glut.h>

 GLfloat xRotated, yRotated, zRotated;
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

    int nx=15;
    int ny=10;
    
    int max =  nx ^ ((nx ^ ny) & -(nx < ny));

    float scaling = 2.0 / (3.0 * max);
    float spacing = 4.0 * scaling;
    float z_distance= -3.0;
    
    glLoadIdentity();
    //glTranslatef(-nx * 0.5 * spacing + 1.0, ny*0.5*spacing - 1.0, z_distance);
    glTranslatef(-1.1, 1.1, z_distance);

    for (int i=0; i<nx; i++)
	for(int j=0; j<ny; j++){
	    
	    glPushMatrix();
	    glTranslatef(i*spacing, -j*spacing, 0.0);
	    glRotatef(xRotated,1.0,0.0,0.0);
	    // rotation about Y axis
	    glRotatef(yRotated*0.2*i,0.0,1.0,0.0);
	    // rotation about Z axis
	    glRotatef(zRotated,0.0,0.0,1.0);
	    glScalef(scaling, scaling, scaling);

	    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	    glBegin(GL_TRIANGLES);        // Draw The Cube Using quads
	      glColor3f(0.0f,0.0f,0.0f);       
	      
	      // Front
	      //glColor3f(1.0f, 0.0f, 0.0f);     
	      glVertex3f( 0.0f, 1.0f, 0.0f);
	      glVertex3f(-1.0f, -1.0f, 1.0f);
	      glVertex3f(1.0f, -1.0f, 1.0f);
	 
	      // Right
	      //glColor3f(0.0f, 1.0f, 0.0f);     
	      glVertex3f(0.0f, 1.0f, 0.0f);
	      glVertex3f(1.0f, -1.0f, 1.0f);
	      glVertex3f(1.0f, -1.0f, -1.0f);
	 
	      // Back
	      //glColor3f(0.0f, 0.0f, 1.0f);     
	      glVertex3f(0.0f, 1.0f, 0.0f);
	      glVertex3f(1.0f, -1.0f, -1.0f);
	      glVertex3f(-1.0f, -1.0f, -1.0f);
	 
	      // Left
	      //glColor3f(0.0f,0.0f,0.0f);       
	      glVertex3f( 0.0f, 1.0f, 0.0f);
	      glVertex3f(-1.0f,-1.0f,-1.0f);
	      glVertex3f(-1.0f,-1.0f, 1.0f);
	    glEnd();            // End Drawing The Cube

	    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	    glBegin(GL_TRIANGLES);        // Draw The Cube Using quads
	      glColor3f(1.0f,1.0f,1.0f);       
	      
	      // Front
	      //glColor3f(1.0f, 0.0f, 0.0f);     
	      glVertex3f( 0.0f, 1.0f, 0.0f);
	      glVertex3f(-1.0f, -1.0f, 1.0f);
	      glVertex3f(1.0f, -1.0f, 1.0f);
	 
	      // Right
	      //glColor3f(0.0f, 1.0f, 0.0f);     
	      glVertex3f(0.0f, 1.0f, 0.0f);
	      glVertex3f(1.0f, -1.0f, 1.0f);
	      glVertex3f(1.0f, -1.0f, -1.0f);
	 
	      // Back
	      //glColor3f(0.0f, 0.0f, 1.0f);     
	      glVertex3f(0.0f, 1.0f, 0.0f);
	      glVertex3f(1.0f, -1.0f, -1.0f);
	      glVertex3f(-1.0f, -1.0f, -1.0f);
	 
	      // Left
	      //glColor3f(0.0f,0.0f,0.0f);       
	      glVertex3f( 0.0f, 1.0f, 0.0f);
	      glVertex3f(-1.0f,-1.0f,-1.0f);
	      glVertex3f(-1.0f,-1.0f, 1.0f);
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

    glutInit(&argc, argv);
    //we initizlilze the glut. functions
    glutInitDisplayMode(GLUT_SINGLE| GLUT_RGB| GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(argv[0]);
    init();
    glutDisplayFunc(DrawCube);
    glutReshapeFunc(reshape);

    //Set the function for the animation.
    //glutIdleFunc(animation);
    glutMainLoop();
return 0;
} 






  
