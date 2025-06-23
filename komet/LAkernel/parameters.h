/*********************************************************************
    parameters.h (version 0.2)                                          
                                                                    
    Parameters to be used by the local alignment kernel.

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



#ifndef PARAMETERS_H
#define PARAMETERS_H

/****************/
/* The alphabet */
/****************/

#define NAA 20                                  /* Number of amino-acids */
#define NLET 26                                 /* Number of letters in the alphabet */
static char *aaList= "ARNDCQEGHILKMFPSTWYV";    /* The list of amino acids */

/************************/
/* Alignment parameters */
/************************/

#define OPENING 12                              /* Gap opening penalty */
#define EXTENSION 2                             /* Gap extension penalty */

/* mutation matrix */
static double blosum[] = {
  4,
 -1,  5,
 -2,  0,  6,
 -2, -2,  1,  6,
  0, -3, -3, -3,  9,
 -1,  1,  0,  0, -3,  5,
 -1,  0,  0,  2, -4,  2,  5,
  0, -2,  0, -1, -3, -2, -2,  6,
 -2,  0, 1, -1, -3,  0,  0, -2,  8,
 -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,
 -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4,
 -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5,
 -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,
 -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6,
 -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7,
  1, -1,  1, 0, -1,  0, 0,  0, -1, -2, -2, 0, -1, -2, -1,  4,
  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5,
 -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2,  11,
 -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7,
  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4};

/* Index corresponding to the (i,j) entry (i,j=0..19) in the blosum matrix */
#define BINDEX(i,j) (((i)>(j))?(j)+(((i)*(i+1))/2):(i)+(((j)*(j+1))/2))



/*********************
 * Kernel parameters *
 *********************/

#define SCALING 1           /* Factor to scale all SW parameters */

/* If you want to compute the sum over all local alignments (to get a valid kernel), uncomment the following line : */
/* If x=log(a) and y=log(b), compute log(a+b) : */
#define LOGP(x,y) (((x)>(y))?(x)+log1p(exp((y)-(x))):(y)+log1p(exp((x)-(y)))) 

/* OR if you want to compute the score of the best local alignment (to get the SW score by Viterbi), uncomment the following line : */
/*
#define LOGP(x,y) (((x)>(y))?(x):(y)) 
*/



#endif
