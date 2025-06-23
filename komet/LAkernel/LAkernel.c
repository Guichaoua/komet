/*********************************************************************
    LAkernel.c (version 0.2)                                          
                                                                    
    Compute the local alignment kernel

    Reference:
    H. Saigo, J.-P. Vert, T. Akutsu and N. Ueda, "Protein homology 
    detection using string alignment kernels", Bioinformatics, 
    vol.20, p.1682-1689, 2004.
                                                                    
                                                                    
    Copyright 2003 Jean-Philippe Vert                                 

    This file is part of LAkernel.

    LAkernel is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

 *********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include "parameters.h"       /* to be adjusted if necessary */



/* Usefule constants */
#define LOG0 -10000           /* log(0) */


/*********************/
/* Static variables  */
/*********************/


static int *isAA;                /* Indicates whether a char is an amino-acid */

static int *aaIndex;             /* The correspondance between amino-acid letter and index */

static double opening,extension; /* Gap penalties */





/*********************/
/* Private Functions */
/*********************/



void terror(text)
  char *text;
  /* Print an error message and quit */
{
  fprintf(stderr,"\n\nERROR: %s.\n\n",text);
  exit(-1);
}


double *DarrayNew(size)
     /* Allocate an array of 'double' of size 'size' */
     int size;
{
  double *d;  
  if ((d=(double *)malloc(size*sizeof(double))) == NULL)
    terror("run out or memory");
  return d;
}


void initialize(void)
     /* Initialize all static variables. This function should be called once before computing the first pair HMM score */
{
  register int i;

  /* Initialization of the array which gives the position of each amino-acid in the set of amino-acid */
  if ((aaIndex=(int *)calloc(NLET,sizeof(int))) == NULL)
    terror("run out or memory");
  for (i=0;i<NAA;i++) 
    aaIndex[aaList[i]-'A']=i;
  
  /* Initialization of the array which indicates whether a char is an amino-acid */
  if ((isAA=(int *)calloc(256,sizeof(int))) == NULL)
    terror("run out or memory");
  for (i=0;i<NAA;i++) 
    isAA[(int)aaList[i]]=1;

  /* Scale the blossum matrix */
  for (i=0 ; i<NAA*(NAA+1)/2 ; blosum[i++] *= SCALING);

  /* Scale of gap penalties */
  opening = OPENING * SCALING;
  extension = EXTENSION * SCALING;
}




double LAkernelcompute(aaX,aaY,nX,nY)
  /* Implementation of the convolution kernel which generalizes the Smith-Waterman algorithm */
     int *aaX , *aaY;   /* the two amino-acid sequences (as sequences of indexes in [0..NAA-1] indicating the position of the amino-acid in the variable 'aaList') */
     int nX , nY;       /* the lengths of both sequences */
{
   register int
    i,j,                /* loop indexes */
    cur, old,           /* to indicate the array to use (0 or 1) */
    curpos, frompos;    /* position in an array */


  double 
    *logX,           /* arrays to store the log-values of each state */
    *logY,
    *logM,
    *logX2,
    *logY2,
    aux , aux2;
  int
    cl;                /* length of a column for the dynamic programming */

  /*
  printf("now computing LAkernel between %d and %d:\n",nX,nY);
  for (i=0;i<nX;printf("%d ",aaX[i++]));
  printf("\n and \n");
  for (i=0;i<nY;printf("%d ",aaY[i++]));
  printf("\n");
  */

  /* Initialization of the arrays */
  /* Each array stores two successive columns of the (nX+1)x(nY+1) table used in dynamic programming */
  cl = nY+1;           /* each column stores the positions in the aaY sequence, plus a position at zero */

  logM = DarrayNew(2*cl);
  logX = DarrayNew(2*cl);
  logY = DarrayNew(2*cl);
  logX2 = DarrayNew(2*cl);
  logY2 = DarrayNew(2*cl);


  /************************************************/
  /* First iteration : initialization of column 0 */
  /************************************************/
  /* The log=proabilities of each state are initialized for the first column (x=0,y=0..nY) */

  for (j=0;j<cl;j++) {
    logM[j]=LOG0;
    logX[j]=LOG0;
    logY[j]=LOG0;
    logX2[j]=LOG0;
    logY2[j]=LOG0;
  }

  /* Update column order */
  cur = 1;      /* Indexes [0..cl-1] are used to process the next column */
  old = 0;      /* Indexes [cl..2*cl-1] were used for column 0 */


  /************************************************/
  /* Next iterations : processing columns 1 .. nX */
  /************************************************/

  /* Main loop to vary the position in aaX : i=1..nX */
  for (i=1;i<=nX;i++) {

    /* Special update for positions (i=1..nX,j=0) */
    curpos = cur*cl;                  /* index of the state (i,0) */
    logM[curpos] = LOG0; 
    logX[curpos] = LOG0; 
    logY[curpos] = LOG0; 
    logX2[curpos] = LOG0; 
    logY2[curpos] = LOG0; 

    /* Secondary loop to vary the position in aaY : j=1..nY */
    for (j=1;j<=nY;j++) {

      curpos = cur*cl + j;            /* index of the state (i,j) */

      /* Update for states which emit X only */
      /***************************************/

      frompos = old*cl + j;            /* index of the state (i-1,j) */
      
      /* State RX */
      logX[curpos] = LOGP( - opening + logM[frompos] , - extension + logX[frompos] );

      /* State RX2 */
      logX2[curpos] = LOGP ( logM[frompos] , logX2[frompos] );

      
      /* Update for states which emit Y only */
      /***************************************/

      frompos = cur*cl + j-1;          /* index of the state (i,j-1) */

      /* State RY */
      aux = LOGP( - opening + logM[frompos] , - extension + logY[frompos] );
      logY[curpos] = LOGP( aux , - opening + logX[frompos] );

      /* State RY2 */
      aux = LOGP( logM[frompos] , logY2[frompos] );
      logY2[curpos] = LOGP( aux , logX2[frompos] );
      

      /* Update for states which emit X and Y */
      /****************************************/

      frompos = old*cl + j-1;          /* index of the state (i-1,j-1) */

      aux = LOGP( logX[frompos] , logY[frompos] );
      aux2 = LOGP( 0 , logM[frompos] );

      logM[curpos] = LOGP( aux , aux2 ) + blosum[ BINDEX( aaX[i-1] , aaY[j-1] ) ];

      /*      printf("i=%d , j=%d\nM=%.5f\nX=%.5f\nY=%.5f\nX2=%.5f\nY2=%.5f\n",i,j,logM[curpos],logX[curpos],logY[curpos],logX2[curpos],logY2[curpos]);
       */

    }  /* end of j=1:nY loop */


    /* Update the culumn order */
    cur = 1-cur;
    old = 1-old;

  }  /* end of j=1:nX loop */


  /* Termination */
  /***************/

  curpos = old*cl + nY;                /* index of the state (nX,nY) */
  aux = LOGP( logX2[curpos] , logY2[curpos] );
  aux2 = LOGP( 0 , logM[curpos] );

  /* Memory release */
  free( logM );
  free( logX );
  free( logY );
  free( logX2 );
  free( logY2 );

  /* Return the logarithm of the kernel */
  return LOGP(aux,aux2);
}




/********************/
/* Public functions */
/********************/


double LAkernel(x,y)
     /* Return the log-probability of two sequences x and y under a pair HMM model */
     /* x and y are strings of aminoacid letters, e.g., "AABRS" */
     char *x , *y;
{
  int *aax,*aay;  /* to convert x and y into sequences of amino-acid indexes */
  int lx,ly;       /* lengths of x and y */
  int i,j;
  float result;

  /* If necessary, initialize static variables */
  if (isAA == NULL)
    initialize();

  lx = strlen(x);
  ly = strlen(y);

  if ((lx<1) || (ly<1))
    terror("empty chain");

  /* Create aax and aay */

  if ((aax=(int *)calloc(lx,sizeof(int))) == NULL)
    terror("run out or memory");
  if ((aay=(int *)calloc(ly,sizeof(int))) == NULL)
    terror("run out or memory");

  /* Extract the characters corresponding to aminoacids and keep their indexes */

  j=0;
  for (i=0 ; i<lx ; i++) 
    if (isAA[toupper(x[i])])
      aax[j++] = aaIndex[toupper(x[i])-'A'];
  lx = j;
  j=0;
  for (i=0 ; i<ly ; i++)
    if (isAA[toupper(y[i])])
      aay[j++] = aaIndex[toupper(y[i])-'A'];
  ly = j;

  /* Compute the pair HMM score */
  result = LAkernelcompute(aax,aay,lx,ly);

  /* Release memory */
  free(aax);
  free(aay);

  return result;
}

