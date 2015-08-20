#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>

 GLfloat xRotated, yRotated, zRotated;
 int nz, ny, nx; 
 std::string fname;
 GLuint* textureIDs;
 GLuint texture3d_id;

 GLfloat dOrthoSize = 1.0f;

void initTextures(){
    
    int pixel_num = nz * ny * nx;
    
    std::fstream file;
    file.open(fname, std::ios::in|std::ios::binary|std::ios::ate);

    std::streampos size;
    char* buffer;
    bool read_failure = true;

    std::cout << "Reading file " << fname << " with dimensions(Slices, Rows, Cols) " << nz << " X " << ny << " X " << nx << std::endl;
    std::cout << "Number of pixels = " << pixel_num << std::endl;

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
	return;
	}

    std::cout << "File successfully imported! File Size = " << size << " Bytes" <<std::endl;
/*
    textureIDs = new GLuint[nz];
    glGenTextures(nz, (GLuint*) textureIDs );

    char* framebuffer;// = new char[ny * nx];
    char* RGBAbuffer =  new char[ny * nx * 4];
    for(int i=0; i < nz; ++i){
	framebuffer = &buffer[i * ny * nx];
	for(int px = 0; px < ny*nx; px++){
	    RGBAbuffer[px * 4] = framebuffer[px];
	    RGBAbuffer[px * 4 + 1] = framebuffer[px];
	    RGBAbuffer[px * 4 + 2] = framebuffer[px];
	    RGBAbuffer[px * 4 + 3] = framebuffer[px];
	}

	glBindTexture(GL_TEXTURE_2D, textureIDs[i]);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nx, ny , 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid *) RGBAbuffer);
	glBindTexture(GL_TEXTURE_2D, 0);
    }
    
    delete[] textureIDs;
    */
    char* RGBAbuffer = new char[size * 4];
    for(int px = 0; px < size; px++){
	RGBAbuffer[px * 4] = buffer[px];
	RGBAbuffer[px * 4 + 1] = buffer[px];
	RGBAbuffer[px * 4 + 2] = buffer[px];
	RGBAbuffer[px * 4 + 3] = buffer[px];
 
    }
    glGenTextures(1, (GLuint*)&texture3d_id);
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

    delete[] buffer;
}

void init(void)
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glewInit();
    initTextures();
}

void DrawCube(void)
{

    glMatrixMode(GL_MODELVIEW);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    
/*  glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS); */

    glEnable(GL_ALPHA_TEST );
    glAlphaFunc(GL_GREATER, 0.05f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();

    glTranslatef(0.5f, 0.5f, 0.5f);
    glScaled( (float)nx /(float) nx, -1.0f*(float) nx/(float)ny, (float)nx/(float)nz);
    glRotated(yRotated, 0.0, 1.0, 0.0);
    glTranslatef(-0.5f, -0.5f, -0.5f);
/*
    glEnable(GL_TEXTURE_2D); 
	    
    for(int i=0; i < nz; ++i){
        glBindTexture(GL_TEXTURE_2D, textureIDs[i]);
	
	glBegin(GL_QUADS);
	    glTexCoord2f(	0.0f,	0.0f);
	    glVertex3f(	-1.0f,	-1.0f, 1.0 - (GLfloat)(i / nz));    // Top Right Of The Quad (Top)
	    glTexCoord2f(	1.0f,	0.0f);
	    glVertex3f(	1.0f,   -1.0f, 1.0 - (GLfloat)(i / nz));    // Top Left Of The Quad (Top)
	    glTexCoord2f(	1.0f,	1.0f);
	    glVertex3f(	1.0f,   1.0f,  1.0 - (GLfloat)(i / nz));    // Bottom Left Of The Quad (Top)
	    glTexCoord2f(	0.0f,	1.0f);
	    glVertex3f(	-1.0f,  1.0f,  1.0 - (GLfloat)(i / nz));    // Bottom Right Of The Quad (Top)
	glEnd();            
	
	glBindTexture(GL_TEXTURE_2D, 0);
    }
*/
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


void animation(void)
{
 
     yRotated += 0.50;
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

int main(int argc, char** argv){
    nz = 30;
    ny = 30;
    nx = 30;
    yRotated = 0.0;
    fname = "test.raw";

    if(argc > 1){
	fname = argv[1];
	nz=atoi(argv[2]);
	ny=atoi(argv[3]);
	nx=atoi(argv[4]);
    }

    glutInit(&argc, argv);
    //we initizlilze the glut. functions
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(argv[0]);
    init();
    glutDisplayFunc(DrawCube);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    //Set the function for the animation.
    glutIdleFunc(animation);
    glutMainLoop();

        return 0;
} 






  
