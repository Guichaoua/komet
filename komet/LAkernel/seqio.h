/*
 * SEQIO.H  -  A C Package for Performing Sequence File I/O  (Version 1.1)
 *
 *   Copyright (c) 1996 by James Knight at Univ. of California, Davis
 * 
 *   Permission to use, copy, modify, distribute and sell this software
 *   and its documentation is hereby granted, subject to the following
 *   restrictions and understandings:
 * 
 *     1) Any copy of this software or any copy of software derived
 *        from it must include this copyright notice in full.
 * 
 *     2) All materials or software developed as a consequence of the
 *        use of this software or software derived from it must duly
 *        acknowledge such use, in accordance with the usual standards
 *        of acknowledging credit in academic research.
 * 
 *     3) The software may be used freely by anyone for any purpose,
 *        commercial or non-commercial.  That includes, but is not
 *        limited to, its incorporation into software sold for a profit
 *        or the development of commercial software derived from it.
 *  
 *     4) This software is provided AS IS with no warranties of any
 *        kind.  The author shall have no liability with respect to the
 *        infringement of copyrights, trade secrets or any patents by
 *        this software or any part thereof.  In no event will the
 *        author be liable for any lost revenue or profits or other
 *        special, indirect and consequential damages. 
 */
#ifndef _SEQIO_H_
#define _SEQIO_H_

#ifdef __cplusplus
  extern "C" {
#endif

/*
 * SEQIO File & Database Input, Output and Entry Information functions.
 *
 * See the documentation files included with the SEQIO release for
 * descriptions of the structures and functions.
 */

typedef void SEQFILE;

typedef struct {
  char *dbname, *filename, *format;
  int entryno, seqno, numseqs;

  char *date, *idlist, *description;
  char *comment, *organism, *history;
  int isfragment, iscircular, alphabet;
  int fragstart, truelen, rawlen;
} SEQINFO;

#define UNKNOWN 0
#define DNA 1
#define RNA 2
#define PROTEIN 3
#define AMINO 3


SEQFILE *seqfopen(char *filename, char *mode, char *format);
SEQFILE *seqfopendb(char *dbname);
SEQFILE *seqfopen2(char *str);
void seqfclose(SEQFILE *sfp);

int seqfread(SEQFILE *sfp, int flag);

char *seqfgetseq(SEQFILE *sfp, int *length_out, int newbuffer);
char *seqfgetrawseq(SEQFILE *sfp, int *length_out, int newbuffer);
char *seqfgetentry(SEQFILE *sfp, int *length_out, int newbuffer);
SEQINFO *seqfgetinfo(SEQFILE *sfp, int newbuffer);

char *seqfsequence(SEQFILE *sfp, int *length_out, int newbuffer);
char *seqfrawseq(SEQFILE *sfp, int *length_out, int newbuffer);
char *seqfentry(SEQFILE *sfp, int *length_out, int newbuffer);
SEQINFO *seqfinfo(SEQFILE *sfp, int newbuffer);
SEQINFO *seqfallinfo(SEQFILE *sfp, int newbuffer);

char *seqfdbname(SEQFILE *sfp, int newbuffer);
char *seqffilename(SEQFILE *sfp, int newbuffer);
char *seqfformat(SEQFILE *sfp, int newbuffer);
int seqfentryno(SEQFILE *sfp);
int seqfseqno(SEQFILE *sfp);
int seqfnumseqs(SEQFILE *sfp);
char *seqfdate(SEQFILE *sfp, int newbuffer);
char *seqfidlist(SEQFILE *sfp, int newbuffer);
char *seqfdescription(SEQFILE *sfp, int newbuffer);
char *seqfcomment(SEQFILE *sfp, int newbuffer);
char *seqforganism(SEQFILE *sfp, int newbuffer);
int seqfiscircular(SEQFILE *sfp);
int seqfisfragment(SEQFILE *sfp);
int seqffragstart(SEQFILE *sfp);
int seqfalphabet(SEQFILE *sfp);
int seqftruelen(SEQFILE *sfp);
int seqfrawlen(SEQFILE *sfp);

char *seqfmainid(SEQFILE *sfp, int newbuffer);
char *seqfmainacc(SEQFILE *sfp, int newbuffer);
int seqfoneline(SEQINFO *info, char *buffer, int buflen, int idonly);

void seqfsetidpref(SEQFILE *sfp, char *idprefix);
void seqfsetdbname(SEQFILE *sfp, char *dbname);
void seqfsetalpha(SEQFILE *sfp, char *alphabet);

int seqfwrite(SEQFILE *sfp, char *seq, int seqlen, SEQINFO *info);
int seqfconvert(SEQFILE *input_sfp, SEQFILE *output_sfp);
int seqfputs(SEQFILE *sfp, char *s, int len);
int seqfannotate(SEQFILE *fp, char *entry, int entrylen, char *newcomment,
                 int flag);
int seqfgcgify(SEQFILE *sfpout, char *entry, int entrylen);
int seqfungcgify(SEQFILE *sfpout, char *entry, int entrylen);

int bioseq_read(char *filelist);
int bioseq_check(char *dbspec);
char *bioseq_info(char *dbspec, char *fieldname);
char *bioseq_matchinfo(char *fieldname, char *fieldvalue);
char *bioseq_parse(char *dbspec);

SEQINFO *seqfparseent(char *entry, int entrylen, char *format);
int asn_parse(char *begin, char *end, ...);
void seqfsetpretty(SEQFILE *sfp, int value);

int seqfisaformat(char *format);
int seqffmttype(char *format);
int seqfcanwrite(char *format);
int seqfcanannotate(char *format);
int seqfcanparseent(char *format);
int seqfcangcgify(char *format);

int seqfbytepos(SEQFILE *sfp);
int seqfisafile(char *file);

#define T_INVFORMAT 0
#define T_SEQONLY 1
#define T_DATABANK 2
#define T_GENERAL 3
#define T_LIMITED 4
#define T_ALIGNMENT 5
#define T_OUTPUT 6


/*
 * SEQIO Error reporting variables, constants and functions.
 */

extern int seqferrno;
extern char seqferrstr[];

#define E_EOF -1             /* End of File/Database reached */
#define E_NOERROR 0          /* No error occurred */
#define E_OPENFAILED 1       /* Opening a file failed */
#define E_READFAILED 2       /* Reading a file failed */
#define E_NOMEMORY 3         /* Ran out of memory */
#define E_PROGRAMERROR 4     /* Bug in the SEQIO program */
#define E_PREVERROR 5        /* Previous fatal error occurred */ 
#define E_PARAMERROR 6       /* Invalid parameter value */
#define E_INVFORMAT 7        /* Invalid file format given */
#define E_DETFAILED 8        /* Unable to determine format of a file */
#define E_PARSEERROR 9       /* File could not be parsed */
#define E_DBPARSEERROR 10    /* Database specification could not be parsed */
#define E_DBFILEERROR 11     /* One of the database files is missing */
#define E_NOSEQ 12           /* Entry does not contain a sequence */
#define E_DIFFLENGTH 13      /* Seq. chars. read differs from len. in entry */
#define E_INVINFO 14         /* Detected an invalid SEQINFO field value */
#define E_FILEERROR 15       /* Invalid filename format detected */


void seqfperror(char *s);
void seqfsetperror(void (*perr_fn)(char *));
int seqferrpolicy(int pe);

#define PE_NONE 1
#define PE_WARNONLY 2
#define PE_ERRONLY 3
#define PE_NOWARN 4
#define PE_NOEXIT 5
#define PE_ALL 6

#ifdef __cplusplus
  }
#endif

#endif
