#include <GL/glut.h>
#include <functional>

GLfloat xRotated, yRotated, zRotated;
GLdouble radius=0.5;
GLfloat qaBlack[] = {0.0, 0.0, 0.0, 1.0}; //Black Color
GLfloat qaGreen[] = {0.0, 1.0, 0.0, 1.0}; //Green Color
GLfloat qaWhite[] = {1.0, 1.0, 1.0, 1.0}; //White Color
GLfloat qaRed[] = {1.0, 0.0, 0.0, 1.0}; //White Color

    // Set lighting intensity and color
GLfloat qaAmbientLight[]    = {0.2, 0.2, 0.2, 1.0};
GLfloat qaDiffuseLight[]    = {0.8, 0.8, 0.8, 1.0};
GLfloat qaSpecularLight[]    = {1.0, 1.0, 1.0, 1.0};
GLfloat emitLight[] = {0.9, 0.9, 0.9, 0.01};
GLfloat Noemit[] = {0.0, 0.0, 0.0, 1.0};
    // Light source position
GLfloat qaLightPosition[]    = {0.5, 0, -3.5, 0.5};

void display(void);
void reshape(int x, int y);
 
void idleFunc(void)
{
 
     zRotated += 0.06;
     
    display();
}
void initLighting()
{

    // Enable lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);

     // Set lighting intensity and color
       glLightfv(GL_LIGHT0, GL_AMBIENT, qaAmbientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, qaDiffuseLight);
     glLightfv(GL_LIGHT0, GL_SPECULAR, qaSpecularLight);
    
    // Set the light position
     glLightfv(GL_LIGHT0, GL_POSITION, qaLightPosition);

}

int main (int argc, char **argv)
{   //foo f;
    glutInit(&argc, argv); 
     glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
    glutInitWindowSize(350,350);
    glutCreateWindow("Solid Sphere");
    initLighting(); 
    xRotated = yRotated = zRotated = 0.0;
    
    glutIdleFunc(idleFunc);
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMainLoop();
    return 0;
}
 
void display(void)
{


    glMatrixMode(GL_MODELVIEW);
    // clear the drawing buffer.
    glClear(GL_COLOR_BUFFER_BIT);
    // clear the identity matrix.
    glLoadIdentity();
    
    // translate the draw by z = -4.0
    // Note this when you decrease z like -8.0 the drawing will looks far , or smaller.
    glTranslatef(0.0,0.0,-5.0);
  
    // scaling transfomation 
    glScalef(2.5,1.1,0.7);
    glRotatef(zRotated,0.0,0.0,1.0);
    // Set material properties
       glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, qaGreen);

    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, qaGreen);

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, qaWhite);

    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 20);

    // built-in (glut library) function , draw you a sphere.
    glutSolidSphere(radius,35,35);
    // Flush buffers to screen
     
    glFlush();
    glutSwapBuffers();      
    // sawp buffers called because we are using double buffering 
   // glutSwapBuffers();
}

void reshape(int x, int y)
{
    if (y == 0 || x == 0) return;   
    glMatrixMode(GL_PROJECTION);  
    glLoadIdentity(); 
    
    gluPerspective(39.0,(GLdouble)x/(GLdouble)y,0.6,21.0);
    glMatrixMode(GL_MODELVIEW);
    glViewport(0,0,x,y);  //Use the whole window for rendering
}
