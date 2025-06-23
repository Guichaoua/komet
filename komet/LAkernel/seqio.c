/*
 * SEQIO.C  -  A C Package for Performing Sequence File I/O  (Version 1.2)
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

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fcntl.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __unix
#include <unistd.h>
#include <dirent.h>
#ifdef SYSV
#include <sys/dirent.h>
#endif
#endif
#ifdef WIN32
#include <windows.h>
#endif
#include "seqio.h"


/*
 * Portability Issues.
 *
 * Integers must be 4 bytes long (they will take values larger than 65536).
 *
 * The character dividing directories in a filepath (in Unix, '/') must
 * be specified as the value of variable "dirch" below.
 *
 * The structure used when reading a raw file using open and read must
 * be specified as the value of typedef FILEPTR.  The structure used
 * when reading a directory file must be specified as the value of
 * typedef DIRPTR.
 *
 * Current set of external calls in main section of code:
 *      exit, fclose, fopen, fputc, fputc, fprintf, free, fwrite,
 *      getenv, isalpha, isalnum, isdigit, isspace,
 *      malloc, memcpy, memset, realloc, sizeof, sprintf,
 *      strcpy, strcmp, strlen, strncmp, tolower, va_arg, va_end,
 *      va_start, vsprintf
 *      mmap, munmap (these are ifdef'd inside `ISMAPABLE')
 *
 * Current set of (unusual?) data-structures/variables in main section:
 *      errno, va_list, __LINE__,
 *      caddr_t (this is ifdef'd inside `ISMAPABLE')
 *
 * Procedures found at the end of this file which cover all of the file I/O:
 *      open_raw_file, read_raw_file, seek_raw_file, close_raw_file,
 *      open_raw_stdin, open_stdout, puterror, read_small_file, open_directory,
 *      read_dirname, close_directory, isa_file, get_filesize, isa_dir,
 *      get_truename, is_absolute, get_today
 *
 * Current set of external calls in end section of code:
 *      close, ctime, open, lseek, read, stat, time
 *
 *      closedir, opendir, readdir  (these are ifdef'd inside `__unix')
 *
 *      GetCurrentDirectory, SetCurrentDirectory,
 *      FindFirstFile, FindNextFile, CloseHandle
 *                              (these are ifdef'd inside `WIN32')
 *
 * Current set of (unusual?) data-structures/variables in end section:
 *      stat structure, time_t, stdin, stdout, stderr
 *      DIR, dirent structure  (these are ifdef'd inside `__unix')
 *      WIN32_FIND_DATA, HANDLE   (these are ifdef'd inside `WIN32')
 *
 */

#ifdef WIN32

static char dirch = '\\';
typedef struct {
  int init_flag;
  WIN32_FIND_DATA dirinfo;
  HANDLE handle;
} DIRSTRUCT, *DIRPTR;

int open(), read(), close();

#else

static char dirch = '/';
typedef DIR *DIRPTR;

#endif


typedef int FILEPTR;

static int open_raw_file(char *filename, FILEPTR *ptr_out);
static int read_raw_file(FILEPTR ptr, char *buffer, int size);
static int seek_raw_file(FILEPTR, int pos);
static int close_raw_file(FILEPTR ptr);
static int open_raw_stdin(FILEPTR *ptr_out);
static int open_stdout(FILE **ptr_out);
static void puterror(char *s);
static char *read_small_file(char *filename);


static int open_directory(char *dirname, DIRPTR *dp_out);
static char *read_dirname(DIRPTR dp);
static void close_directory(DIRPTR dp);
static int isa_file(char *filename);
static int get_filesize(char *filename);
static char *get_truename(char *filename, char *fileend);
static int is_absolute(char *path);
static char *get_today();




/*
 *
 * Prototypes for external functions that are not declared in the include
 * files, and replacement functions for system calls that don't exist on
 * one or more machines.
 
extern char *sys_errlist[];
*/
#if defined(__sun) && !defined(FILENAME_MAX)
#include <sys/param.h>
#define FILENAME_MAX MAXPATHLEN
#endif


static int ctype_initflag = 0;
static char tubuf[384], *tuary;

#define toupper(chr) tuary[(int) (chr)]

static void init_ctype(void)
{
  int i;
  char j;

  tuary = tubuf + 128;

  for (i=-128; i < 255; i++)
    tuary[i] = i;
  for (i='a',j='A'; i <= 'z'; i++,j++)
    tuary[i] = j;

  ctype_initflag = 1;
}

static int mycasecmp(char *s, char *t)
{
  int diff;

  for ( ; !(diff = toupper(*s) - toupper(*t)) && *s; s++,t++) ;
  return diff;
}

static int myncasecmp(char *s, char *t, int n)
{
  int diff, i;

  diff = 0;
  for (i=0; i < n && !(diff = toupper(*s) - toupper(*t)) && *s; s++,t++,i++) ;
  return diff;
}

static char *mystrdup(char *s)
{
  char *temp;

  temp = (char *) malloc(strlen(s)+1);
  return (temp == NULL ? NULL : strcpy(temp, s));
}

static char *mystrdup2(char *s, char *t)
{
  char *temp;

  if ((temp = (char *) malloc(t - s + 1)) == NULL)
    return NULL;

  memcpy(temp, s, t - s);
  temp[t - s] = '\0';
  return temp;
}

#define mystreq(s1,ch,s2)  (toupper(*(s1)) == (ch) && mystreqfn((s1),(s2)))
static int mystreqfn(char *s1, char *s2)
{
  int diff;
  while (!(diff = toupper(*++s1) - *++s2) && *s2) ;
  return !*s2;
}

static int myatoi(char *s, int base, char basechar)
{
  int num, sign;

  while (isspace(*s)) s++;

  sign = 0;
  if (*s == '+' || *s == '-') {
    sign = (*s == '-');
    s++;
  }

  for (num=0; *s >= basechar && *s < basechar + base; s++) {
    num *= base;
    num += *s - basechar;
  }

  return (sign ? -num : num);
}

char *myitoa(char *s, int num, int base, char basechar)
{
  int pos, digit;
  char buffer[128];

  if (num < 0) {
    *s++ = '-';
    num *= -1;
  }

  pos = 0;
  do {
    digit = num % base;
    buffer[pos++] = (char) (digit + basechar);
    num /= base;
  } while (num != 0);

  for (pos--; pos >= 0; pos--)
    *s++ = buffer[pos];

  return s;
}



/*
 *
 * Includes and defines for the mmap operation.
 *
 */
#if defined(__sgi) || defined(__sun) || defined(__alpha)
#define ISMAPABLE 1
#endif

#ifdef ISMAPABLE

#include <sys/mman.h>

#ifdef __sgi
void *mmap();
int munmap();
#endif
#ifdef __sun
int munmap();
#endif

/*
 * Largest number under 1 million which is a multiple of
 * 512, 1024, 2048, 4096, 8192 and 16384, in order to 
 * guarantee that it matches the page size.
 */
#define MYPAGESIZE 16384
#define MAXMAPSIZE 999424

#endif





/*
 *
 * The internal SEQFILE data structure.
 *
 */
typedef enum { OP_READ, OP_DB, OP_WRITE } OPTYPE;
typedef enum { OP_ACTIVE, OP_EOF, OP_ERROR, OP_TEMPERR, OP_FREED } OPSTATUS;
typedef enum { INFO_NONE, INFO_ANY, INFO_ALL, INFO_ALLINFO } INFOSTATUS;

typedef struct {
  OPTYPE optype;
  OPSTATUS opstatus;

  char *db_files, *db_currentfile;
  char *db_spec, *db_name, *db_format;
  char *db_alpha, *db_idprefix;

  char *filename;
  FILEPTR input_fd;
  FILE *output_fp;
  int format, openflag, prettyflag;
  int autodetermined, initreadflag;
  int randaccessflag, *byteoffsets, currentoffset, num_offsets;

  char *fp_buffer;
  int fp_bufsize, fp_bytepos;
  char *fp_current, *fp_top;
  char *fp_entrystart, *fp_seqstart, *fp_entryend;
  char savech, *savech_loc;
  int isendtagged;

  int ismapped, mapsize, filepos, filesize, mapentflag, mapentsize;
  char *mapentry;

  char *seq;
  int seqlen, seqsize, isseqcurrent, rawseqflag;

  int entry_count, entry_seqlen, entry_seqno, entry_numseqs;
  int entry_truelen, entry_rawlen, iflag_truelen, iflag_rawlen;

  SEQINFO *info;
  int infosize, infobufsize;
  char *idbuffer;

  INFOSTATUS istatus;
  int iflag_date, iflag_idlist, iflag_description;
  int iflag_comment, iflag_organism, iflag_fragment;
  int iflag_circular, iflag_alphabet, iflag_fragstart;

  char *nbrf_header;

  int fout_mode, fout_markx, fout_len1, fout_alpha1, fout_len2, fout_alpha2;
  char *fout_id1, *fout_descr1, *fout_id2, *fout_descr2, fout_progname[64];

  int malign_count, malign_size, malign_seqno;
  char **malign_seqs, **malign_ids;
  int *malign_seqlens;

  int phylip_origfmt;

  int gcg_subformat;
  char *gcg_infoline;
} INTSEQFILE;

#define INIT_BUFSIZE 65537
#define INIT_SEQSIZE 16384
#define CONCAT_READ_POINT 2048

#define SEQINFO_ALL 0
#define SEQINFO_ALLINFO 1
#define SEQINFO_DATE 2
#define SEQINFO_IDLIST 3
#define SEQINFO_DESCRIPTION 4
#define SEQINFO_COMMENT 5
#define SEQINFO_ORGANISM 6
#define SEQINFO_HISTORY 7
#define SEQINFO_FRAGMENT 8
#define SEQINFO_CIRCULAR 9
#define SEQINFO_ALPHABET 10
#define SEQINFO_FILENAME 11
#define SEQINFO_DBNAME 12
#define SEQINFO_FORMAT 13
#define SEQINFO_ENTRYNO 14
#define SEQINFO_SEQNO 15
#define SEQINFO_NUMSEQS 16
#define SEQINFO_STARTPOS 17
#define SEQINFO_TRUELEN 18
#define SEQINFO_RAWLEN 19
#define SEQINFO_MAINID 20
#define SEQINFO_MAINACC 21

#define GETSEQ_SEQUENCE 0
#define GETSEQ_RAWSEQ 1
#define GETSEQ_LENGTHS 2



/*
 * The file table.  Gives the C functions which parses particular file
 * formats.  Also gives the command line and pattern file option strings, and
 * the determinant string used to determine the format of an unknown file.
 *
 * This determinant is matched against the first line in the sequence file
 * whose first character is not a space (all of the formats supported so far
 * have a non-space character occuring in the first position of the first line
 * of a file).
 *
 * In the determinant strings below, a question mark '?' denotes a wildcard
 * character (to distinguish the nbrf and fasta formats).  The code in 
 * file.c must be changed to support a new file format which actually 
 * uses period for its determinant string.
 *
 * (NOTE:  The FORMAT defines must have values corresponding to the indices
 *         into the file_table.
 */

static int databank_read(INTSEQFILE *, int), basic_read(INTSEQFILE *, int);
static int basic_getseq(INTSEQFILE *, int);
static int databank_fast_read(INTSEQFILE *, int);
static int databank_fast_getseq(INTSEQFILE *, int);

static int raw_read(INTSEQFILE *, int), raw_getseq(INTSEQFILE *, int);
static int raw_getinfo(INTSEQFILE *, char *, int, int);
static int raw_putseq(INTSEQFILE *, char *, int, SEQINFO *);

static int plain_putseq(INTSEQFILE *, char *, int, SEQINFO *);

static int genbank_getinfo(INTSEQFILE *, char *, int, int);
static int genbank_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int genbank_annotate(FILE *, char *, int, char *, int);

static int nbrf_getinfo(INTSEQFILE *, char *, int, int);
static int nbrf_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int nbrfold_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int nbrf_annotate(FILE *, char *, int, char *, int);

static int fasta_getinfo(INTSEQFILE *, char *, int, int);
static int fasta_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int fastaold_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int fasta_annotate(FILE *, char *, int, char *, int);

static int embl_getinfo(INTSEQFILE *, char *, int, int);
static int embl_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int sprot_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int embl_annotate(FILE *, char *, int, char *, int);
static int sprot_annotate(FILE *, char *, int, char *, int);

static int pir_getinfo(INTSEQFILE *, char *, int, int);
static int pir_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int pir_annotate(FILE *, char *, int, char *, int);

static int stanford_getinfo(INTSEQFILE *, char *, int, int);
static int stanford_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int stanfordold_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int stanford_annotate(FILE *, char *, int, char *, int);

static int asn_read(INTSEQFILE *, int), asn_getseq(INTSEQFILE *, int);
static int asn_getinfo(INTSEQFILE *, char *, int, int);
static int asn_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int asn_putseqend(INTSEQFILE *);
static int asn_annotate(FILE *, char *, int, char *, int);

static int fastaout_read(INTSEQFILE *, int);
static int fastaout_getseq(INTSEQFILE *, int);
static int fastaout_getinfo(INTSEQFILE *, char *, int, int);

static int blastout_read(INTSEQFILE *, int);
static int blastout_getseq(INTSEQFILE *, int);
static int blastout_getinfo(INTSEQFILE *, char *, int, int);

static int phylip_read(INTSEQFILE *, int);
static int phyint_getseq(INTSEQFILE *, int), physeq_getseq(INTSEQFILE *, int);
static int phyint_getinfo(INTSEQFILE *, char *, int, int);
static int physeq_getinfo(INTSEQFILE *, char *, int, int);
static int phylip_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int phyint_putseqend(INTSEQFILE *);
static int physeq_putseqend(INTSEQFILE *);

static int clustal_read(INTSEQFILE *, int), clustal_getseq(INTSEQFILE *, int);
static int clustal_getinfo(INTSEQFILE *, char *, int, int);
static int clustal_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int clustal_putseqend(INTSEQFILE *);

static int gcg_getseq(INTSEQFILE *, int);
static int gcg_getinfo(INTSEQFILE *, char *, int, int);
static int gcg_putseq(INTSEQFILE *, char *, int, SEQINFO *);

static int msf_read(INTSEQFILE *, int), msf_getseq(INTSEQFILE *, int);
static int msf_getinfo(INTSEQFILE *, char *, int, int);
static int msf_putseq(INTSEQFILE *, char *, int, SEQINFO *);
static int msf_putseqend(INTSEQFILE *);


#define FORMAT_UNKNOWN -1
#define FORMAT_RAW 0
#define FORMAT_PLAIN 1
#define FORMAT_GENBANK 2
#define FORMAT_GBFAST 4
#define FORMAT_NBRF 5
#define FORMAT_NBRFOLD 6
#define FORMAT_FASTA 8
#define FORMAT_FASTAOLD 10
#define FORMAT_EMBL 12
#define FORMAT_EMBLFAST 13
#define FORMAT_SPROT 14
#define FORMAT_SPFAST 17
#define FORMAT_PIR 18
#define FORMAT_PIRFAST 20
#define FORMAT_STANFORD 21
#define FORMAT_STANFORDOLD 24
#define FORMAT_ASN 28
#define FORMAT_FOUT 30
#define FORMAT_PHYLIP 34
#define FORMAT_PHYSEQ 35
#define FORMAT_PHYINT 38
#define FORMAT_CLUSTAL 41
#define FORMAT_GCG 43
#define FORMAT_MSF 44
#define FORMAT_BOUT 45

typedef struct {
  char *ident;
  int format, type;
  char *determinant;
  int (*read_fn)(INTSEQFILE *, int);
  int (*getseq_fn)(INTSEQFILE *, int);
  int (*getinfo_fn)(INTSEQFILE *, char *, int, int);
  int (*putseq_fn)(INTSEQFILE *, char *, int, SEQINFO *);
  int (*annotate_fn)(FILE *, char *, int, char *, int);
} FILE_TABLE;


#define file_table_size 49
static FILE_TABLE file_table[file_table_size] = {
  { "Raw",  FORMAT_RAW,  T_SEQONLY,
      NULL,
      raw_read,  raw_getseq,  raw_getinfo,  raw_putseq, NULL }, 
  { "Plain",  FORMAT_PLAIN,  T_SEQONLY,
      NULL,
      raw_read,  basic_getseq,  raw_getinfo,  plain_putseq, NULL },
  { "GenBank",  FORMAT_GENBANK,  T_DATABANK,
      "LOCUS |GB???.SEQ          Genetic Sequence Data Bank",
      databank_read,  basic_getseq,  genbank_getinfo,  genbank_putseq,
      genbank_annotate },
  { "gb",  FORMAT_GENBANK,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "gbfast",  FORMAT_GBFAST,  T_DATABANK,
      NULL,
      databank_fast_read,  databank_fast_getseq,  genbank_getinfo,
      genbank_putseq,  genbank_annotate },
  { "NBRF",  FORMAT_NBRF,  T_GENERAL,
      ">??;",
      basic_read,  basic_getseq,  nbrf_getinfo,  nbrf_putseq, nbrf_annotate },
  { "NBRF-old",  FORMAT_NBRFOLD,  T_LIMITED,
      NULL,
      basic_read,  basic_getseq,  nbrf_getinfo,  nbrfold_putseq,  NULL },
  { "NBRFold",  FORMAT_NBRFOLD,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "FASTA",  FORMAT_FASTA,  T_GENERAL,
      ">",
      basic_read,  basic_getseq,  fasta_getinfo,  fasta_putseq,
      fasta_annotate },
  { "Pearson",  FORMAT_FASTA,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "FASTA-old",  FORMAT_FASTAOLD,  T_LIMITED, 
      NULL,
      basic_read,  basic_getseq,  fasta_getinfo,  fastaold_putseq, NULL },
  { "FASTAold",  FORMAT_FASTAOLD,  0,  NULL,  NULL,  NULL,  NULL,  NULL,
      NULL },
  { "EMBL",  FORMAT_EMBL,  T_DATABANK,
      "ID   |CC |XX ",
      databank_read,  basic_getseq,  embl_getinfo,  embl_putseq,
      embl_annotate },
  { "emblfast",  FORMAT_EMBLFAST,  T_DATABANK,
      NULL,
      databank_fast_read,  databank_fast_getseq,  embl_getinfo,  embl_putseq,
      embl_annotate },
  { "Swiss-Prot",  FORMAT_SPROT,  T_DATABANK,
      NULL,
      databank_read,  basic_getseq,  embl_getinfo,  sprot_putseq,
      sprot_annotate },
  { "swissprot",  FORMAT_SPROT,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "sprot",  FORMAT_SPROT,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "spfast",  FORMAT_SPFAST,  T_DATABANK,
      NULL,
      databank_fast_read,  databank_fast_getseq,  embl_getinfo,  sprot_putseq,
      sprot_annotate },
  { "PIR",  FORMAT_PIR,  T_DATABANK,
      "\\\\\\|ENTRY|P R O T E I N  S E Q U E N C E  D A T A B A S E",
      databank_read,  basic_getseq,  pir_getinfo,  pir_putseq, pir_annotate },
  { "CODATA",  FORMAT_PIR,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "pirfast",  FORMAT_PIRFAST,  T_DATABANK,
      NULL,
      databank_fast_read,  databank_fast_getseq,  pir_getinfo,  pir_putseq,
      pir_annotate },
  { "IG/Stanford",  FORMAT_STANFORD,  T_GENERAL,
      ";",
      basic_read,  basic_getseq,  stanford_getinfo,  stanford_putseq,
      stanford_annotate },
  { "IG",  FORMAT_STANFORD,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "Stanford",  FORMAT_STANFORD,  0,  NULL,  NULL,  NULL,  NULL,  NULL,
      NULL },
  { "Stanford-old",  FORMAT_STANFORDOLD,  T_LIMITED,
      NULL,
      basic_read,  basic_getseq,  stanford_getinfo,  stanfordold_putseq,
      NULL },
  { "Stanfordold",  FORMAT_STANFORDOLD, 0,  NULL,  NULL,  NULL,  NULL,  NULL,
      NULL },
  { "IG-old",  FORMAT_STANFORDOLD,  0,  NULL,  NULL,  NULL,  NULL,  NULL,
      NULL }, 
  { "IGold",  FORMAT_STANFORDOLD,  0,  NULL,  NULL,  NULL,  NULL,  NULL,
      NULL }, 
  { "ASN.1",  FORMAT_ASN,  T_DATABANK,
      "Bioseq-set ::= {|Seq-set ::= {",
      asn_read,  asn_getseq,  asn_getinfo,  asn_putseq, asn_annotate },
  { "ASN",  FORMAT_ASN,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "FASTA-output",  FORMAT_FOUT,  T_OUTPUT,
      "FASTA|TFASTA|SSEARCH|LFASTA|LALIGN|ALIGN|FASTX",
      fastaout_read,  fastaout_getseq,  fastaout_getinfo,  NULL,  NULL },
  { "FASTA-out",  FORMAT_FOUT,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "FASTAout",  FORMAT_FOUT,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "Fout",  FORMAT_FOUT,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "PHYLIP",  FORMAT_PHYLIP,  T_ALIGNMENT,
      "0|1|2|3|4|5|6|7|8|9",
      phylip_read,  phyint_getseq,  phyint_getinfo,  phylip_putseq,  NULL },
  { "PHYLIP-seq",  FORMAT_PHYSEQ,  T_ALIGNMENT,
      NULL,
      phylip_read,  physeq_getseq,  physeq_getinfo,  phylip_putseq,  NULL },
  { "PHYLIPseq",  FORMAT_PHYSEQ,  0,  NULL,  NULL,  NULL,  NULL,  NULL,
      NULL },
  { "PHYLIPs",  FORMAT_PHYSEQ,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "PHYLIP-int",  FORMAT_PHYINT,  T_ALIGNMENT,
      NULL,
      phylip_read,  phyint_getseq,  phyint_getinfo,  phylip_putseq,  NULL },
  { "PHYLIPint",  FORMAT_PHYINT,  0,  NULL,  NULL,  NULL,  NULL,  NULL,
      NULL },
  { "PHYLIPi",  FORMAT_PHYINT,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "Clustalw",  FORMAT_CLUSTAL,  T_ALIGNMENT,
      "CLUSTAL",
      clustal_read,  clustal_getseq,  clustal_getinfo,  clustal_putseq,
      NULL },
  { "Clustal",  FORMAT_CLUSTAL,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "GCG", FORMAT_GCG, T_GENERAL,
      NULL,
      raw_read,  gcg_getseq,  gcg_getinfo,  gcg_putseq,  NULL },
  { "MSF", FORMAT_MSF, T_ALIGNMENT,
      "PileUp",
      msf_read, msf_getseq, msf_getinfo, msf_putseq, NULL },
  { "BLAST-output",  FORMAT_BOUT,  T_OUTPUT,
      "BLASTN|BLASTP|BLASTX",
      blastout_read,  blastout_getseq,  blastout_getinfo,  NULL,  NULL },
  { "BLAST-out",  FORMAT_BOUT,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "BLASTout",  FORMAT_BOUT,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL },
  { "Bout",  FORMAT_BOUT,  0,  NULL,  NULL,  NULL,  NULL,  NULL,  NULL }
};


typedef struct {
  char *ident;
  int format;
} GCG_TABLE;

#define gcg_table_size 23
static GCG_TABLE gcg_table[gcg_table_size] = {
  { "GCG-GenBank", FORMAT_GENBANK },
  { "GCG-gb", FORMAT_GENBANK },
  { "GCG-PIR",  FORMAT_PIR },
  { "GCG-CODATA",  FORMAT_PIR },
  { "GCG-EMBL",  FORMAT_EMBL },
  { "GCG-Swiss-Prot",  FORMAT_SPROT },
  { "GCG-swissprot",  FORMAT_SPROT },
  { "GCG-sprot",  FORMAT_SPROT },
  { "GCG-NBRF",  FORMAT_NBRF },
  { "GCG-NBRF-old",  FORMAT_NBRFOLD },
  { "GCG-NBRFold",  FORMAT_NBRFOLD },
  { "GCG-FASTA",  FORMAT_FASTA },
  { "GCG-Pearson",  FORMAT_FASTA },
  { "GCG-FASTA-old",  FORMAT_FASTAOLD },
  { "GCG-FASTAold",  FORMAT_FASTAOLD },
  { "GCG-IG/Stanford",  FORMAT_STANFORD },
  { "GCG-IG",  FORMAT_STANFORD },
  { "GCG-Stanford",  FORMAT_STANFORD },
  { "GCG-Stanford-old",  FORMAT_STANFORDOLD },
  { "GCG-Stanfordold",  FORMAT_STANFORDOLD },
  { "GCG-IG-old",  FORMAT_STANFORDOLD }, 
  { "GCG-IGold",  FORMAT_STANFORDOLD }, 
  { "GCG-MSF", FORMAT_MSF }
};



/*
 * The idprefix table.
 */

#define idpref_table_size 39
struct {
  char *idprefix, *dbname;
} idpref_table[idpref_table_size] = {
  { "acc", "Accession" },
  { "ag2d", "AARHUS/GHENT-2DPAGE" },
  { "agis", "AGIS" },
  { "bbs", "GIBBSQ" },
  { "bbm", "GIBBMT" },
  { "blks", "BLOCKS" },
  { "cpg", "CpGIsle" },
  { "ddb", "DICTYDB" },
  { "ddbj" "DDBJ" },
  { "ec", "ENZYME" },
  { "eco", "ECOGENE" },
  { "embl", "EMBL" },
  { "epd", "EPD" },
  { "est", "dbEST" },
  { "fly", "FlyBase" },
  { "gb", "GenBank" },
  { "gcr", "GCRDB" },
  { "gdb", "GDB" },
  { "gp", "GenPept" },
  { "gi", "GI" },
  { "giim", "GIIM" },
  { "hiv", "HIV" },
  { "imgt", "IMGT" },
  { "mdb", "MaizeDB" },
  { "muid", "MEDLINE" },
  { "nid", "NID" },
  { "omim", "OMIM" },
  { "pat", "Patent" },
  { "pdb", "PDB" },
  { "pir", "PIR" },
  { "prf", "PRF" },
  { "pros", "PROSITE" },
  { "reb", "REBASE" },
  { "rpb", "REPBASE" },
  { "sp", "SWISSPROT" },
  { "sts", "dbSTS" },
  { "tfd", "TRANSFAC" },
  { "wpep", "WORMPEP" },
  { "yepd", "YEPD" }
};

/*
 * The defines, constants and data structures used to handle error reporting.
 */
#define STATUS_OK 0
#define STATUS_WARNING 1
#define STATUS_EOF 2
#define STATUS_ERROR 3
#define STATUS_FATAL 4


int seqferrno = E_NOERROR;
char seqferrstr[1024];

static int pe_flag = PE_ALL;
static int err_batchmode = 0;
static void (*perror_fn)(char *) = puterror;

#define reset_errors()  seqferrno = E_NOERROR; seqferrstr[0] = '\0'
#define set_error(errnum)  seqferrno = errnum

static void print_fatal(char *format, ...);
static void print_error(char *format, ...);
static void print_warning(char *format, ...);

#define raise_error(errorval,retcmd,printcmd) \
          { \
            set_error(errorval); \
            printcmd; \
            retcmd; \
          }

#define error_test(expr,errorval,retcmd,printcmd) \
          { \
            if (expr) { \
              set_error(errorval); \
              printcmd; \
              retcmd; \
            } \
          }

#define param_error(expr,retcmd,function,string) \
          { \
            if (expr) { \
              set_error(E_PARAMERROR); \
              print_error("Parameter Error in %s:  %s\n", function, string); \
              retcmd; \
            } \
          }

#define status_error(retcmd,function) \
          { \
            set_error(E_PROGRAMERROR); \
            err_batchmode = 1; \
            print_error("SEQIO Program Error in %s:  " \
                        "Invalid status return value.\n", function); \
            print_fatal("\n   *** This is probably a bug in the SEQIO " \
                        "package, and not a user error." \
                        "\n   *** Please report the error to the authors " \
                        "of this software.\n\n"); \
            err_batchmode = 0; \
            retcmd; \
          }

#define program_error(expr,retcmd,printcmd) \
          { \
            if (expr) { \
              set_error(E_PROGRAMERROR); \
              err_batchmode = 1; \
              print_error("SEQIO Program Error, line %d:\n", __LINE__); \
              printcmd; \
              print_fatal("\n   *** This is probably a bug in the SEQIO " \
                          "package, and not a user error." \
                          "\n   *** Please report the error to the authors " \
                          "of this software.\n\n"); \
              err_batchmode = 0; \
              retcmd; \
            } \
          }

#define memory_error(expr,retcmd) \
          { \
            if (expr) { \
              set_error(E_NOMEMORY); \
              print_fatal("Memory Error:  Ran out of memory.\n"); \
              retcmd; \
            } \
          }

#define preverror_test(expr,retcmd) \
          { \
            if (expr) { \
              set_error(E_PREVERROR); \
              retcmd; \
            } \
          }

#define eof_test(expr,retcmd) \
          { \
            if (expr) { \
              set_error(E_EOF); \
              retcmd; \
            } \
          }

/*
 * Internal Prototypes and miscellaneous variables.
 */
static int intseqf_open(INTSEQFILE *, char *, char *);
static int intseqf_open_for_writing(INTSEQFILE *, char *, char *, char *);
static int determine_format(INTSEQFILE *);
static int resolve_offsets(INTSEQFILE *, char *, char *);
static void intseqf_close(INTSEQFILE *);
static int intseqf_read(INTSEQFILE *, int);
static char *intseqf_info(INTSEQFILE *, int, int);

static int fp_get_line(INTSEQFILE *, char **, char **);
static int fp_read_more(INTSEQFILE *, char **, char **, char **);
static int fp_read_all(INTSEQFILE *);


static char *months[13] = { "", "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
                          };
static char *full_months[13] = { "", "JANUARY", "FEBRUARY", "MARCH",
                                     "APRIL",   "MAY",      "JUNE", 
                                     "JULY",    "AUGUST",   "SEPTEMBER", 
                                     "OCTOBER", "NOVEMBER", "DECEMBER"
                               };
static char *gcg_full_months[13] = { "", "January", "February", "March",
                                         "April",   "May",      "June", 
                                         "July",    "August",   "September", 
                                         "October", "November", "December"
                                   };

static int isamonth(char *s)
{
  int i;

  switch (toupper(*s)) {
  case 'J': case 'F': case 'M': case 'A':
  case 'S': case 'O': case 'N': case 'D':
    for (i=1; i <= 12; i++)
      if (mystreq(s, full_months[i][0], full_months[i]))
        return i;

  default:
    return 0;
  }
}



/*
 *
 *
 * The File Input Procedures:
 *     seqfopen, dbopen, seqfclose, seqfread, seqfgetseq, seqfgetent
 *
 * The File/Entry Access Procedures:
 *     seqfformat, seqfsequence, seqfentry
 *
 *
 *
 */

/*
 * seqfopen
 *
 * Open the given file and return a structure which can be used to 
 * read or write sequence entries.  Similar to the fopen function,
 * except that only the simple read, write and append modes are 
 * permitted, and an extra file format parameter is needed.
 *
 * Parameters:  filename  -  name of the file to be opened.
 *              mode      -  opening mode, either "r", "w" or "a".
 *              format    -  file format to use (could be NULL if mode is "r").
 *
 * Returns:  A SEQFILE structure.
 */ 
SEQFILE *seqfopen(char *filename, char *mode, char *format)
{
  int status;
  INTSEQFILE *isfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(filename == NULL, return NULL, "seqfopen", "arg 1 is NULL");
  param_error(filename[0] == '\0', return NULL, "seqfopen",
              "arg 1 is an empty string");
  param_error(mode == NULL, return NULL, "seqfopen", "arg 2 is NULL");
  param_error(mode[0] != 'r' && mode[0] != 'w' && mode[0] != 'a', return NULL,
              "seqfopen", "arg 2 is not \"r\", \"w\" or \"a\"");
  param_error((mode[0] == 'w' || mode[0] == 'a') && format == NULL, 
              return NULL, "seqfopen",
              "arg 2 is \"w\" or \"a\", but no file format specified");
  param_error(format != NULL && format[0] == '\0', return NULL, "seqfopen",
              "arg 3 is an empty string");

  /*
   * Allocate the sequence-file structure, and initialize all fields.
   */
  isfp = (INTSEQFILE *) malloc(sizeof(INTSEQFILE));
  memory_error(isfp == NULL, return NULL);
  memset(isfp, 0, sizeof(INTSEQFILE));
  isfp->opstatus = OP_ACTIVE;

  /*
   * Do the file opening and buffer allocation.
   */
  if (mode[0] == 'r') {
    isfp->optype = OP_READ;
    status = intseqf_open(isfp, filename, format);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      intseqf_close(isfp);
      return NULL;
    }
  }
  else {
    isfp->optype = OP_WRITE;
    status = intseqf_open_for_writing(isfp, filename, format, mode);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      intseqf_close(isfp);
      return NULL;
    }
  }

  return (SEQFILE *) isfp;
}


/*
 * seqfopendb
 *
 * Opens a whole database for reading, instead of just a single file.
 * It uses the BIOSEQ procedures to get the database information.
 *
 * Parameters:   dbname   - name of the database (plus optional spec. string)
 *
 * Returns:  A SEQFILE structure.
 */
SEQFILE *seqfopendb(char *dbname)
{
  int status, len;
  char *s;
  INTSEQFILE *isfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(dbname == NULL, return NULL, "seqfopendb", "arg 1 is NULL");
  param_error(dbname[0] == '\0', return NULL, "seqfopendb",
              "arg 1 is an empty string");

  /*
   * Allocate the sequence-file structure, and initialize all the fields.
   */
  isfp = (INTSEQFILE *) malloc(sizeof(INTSEQFILE));
  memory_error(isfp == NULL, return NULL);
  memset(isfp, 0, sizeof(INTSEQFILE));
  isfp->opstatus = OP_ACTIVE;
  isfp->optype = OP_DB;

  isfp->db_spec = mystrdup(dbname);
  if (isfp->db_spec == NULL) {
    free(isfp);
    memory_error(1, return NULL);
  }

  /*
   * Parse the database name (and specification), get the list of files,
   * the database name, the file format and the database title.
   */
  isfp->db_files = bioseq_parse(dbname);
  if (isfp->db_files == NULL) {
    free(isfp->db_spec);
    free(isfp);
    return NULL;
  }
  for (s=isfp->db_files; *s; s++)
    if (*s == '\n')
      *s = '\0';

  isfp->db_name = bioseq_info(dbname, "Name");
  isfp->db_format = bioseq_info(dbname, "Format");
  isfp->db_alpha = bioseq_info(dbname, "Alphabet");
  isfp->db_idprefix = bioseq_info(dbname, "IdPrefix");

  if (isfp->db_format != NULL && !seqfisaformat(isfp->db_format)) {
    set_error(E_INVFORMAT);
    print_error("`%s':  BIOSEQ Entry specifies invalid format `%s'.\n",
                dbname, isfp->db_format);
    intseqf_close(isfp);
    return NULL;
  }
  if (isfp->db_idprefix != NULL) {
    for (s=isfp->db_idprefix,len=0; len < 6 && isalnum(*s); s++,len++)
      if (isupper(*s))
        *s = tolower(*s);
    if (len < 2 || len > 4 || (*s && !isalnum(*s))) {
      set_error(E_INVINFO);
      print_error("`%s':  BIOSEQ Entry specifies invalid id prefix `%s'.\n",
                  dbname, isfp->db_idprefix);
      intseqf_close(isfp);
      return NULL;
    }
  }

  /*
   * Open the first file, and allocate the buffers.
   */
  isfp->db_currentfile = isfp->db_files;
  status = intseqf_open(isfp, isfp->db_currentfile, isfp->db_format);
  switch (status) {
  case STATUS_OK:
  case STATUS_WARNING:
    return isfp;

  case STATUS_FATAL:
    intseqf_close(isfp);
    return NULL;

  case STATUS_EOF:
  case STATUS_ERROR:
    set_error(E_DBFILEERROR);

    /*
     * If searching a database, close the current file and goto the next
     * file in the list.  Return the eof signal if no more files appear
     * in the list.
     */
#ifdef ISMAPABLE
    if (isfp->ismapped) {
      munmap(isfp->fp_buffer, isfp->mapsize);
      isfp->fp_buffer = NULL;
      isfp->fp_bufsize = 0;
      isfp->ismapped = 0;
    }
#endif
    close_raw_file(isfp->input_fd);
    isfp->openflag = 0;

    /*
     * Construct the correct path to the next file, open the file and
     * recursively call seqfread to get the first entry in that file.
     */
    while (1) {
      for (s=isfp->db_currentfile; *s; s++) ;
      isfp->db_currentfile = ++s;
      if (*s == '\0') {
        intseqf_close(isfp);
        return NULL;
      }

      status = intseqf_open(isfp, isfp->db_currentfile, isfp->db_format);
      switch (status) {
      case STATUS_OK:
      case STATUS_WARNING:
        return isfp;

      case STATUS_FATAL:
        intseqf_close(isfp);
        return NULL;

      case STATUS_ERROR:
      case STATUS_EOF:
        set_error(E_DBFILEERROR);
        break;

      default:
        intseqf_close(isfp);
        status_error(return NULL, "seqfopendb");
      }
    }

  default:
    intseqf_close(isfp);
    status_error(return NULL, "seqfopendb");
  }
}


/*
 * seqfopen2
 *
 * A simple interface to open a file/database for reading.  A single
 * string is given as the argument.  If that string specifies a valid
 * file, then that file is opened.  Otherwise, the string is considered
 * a database name and the database is opened.
 *
 * Parameters:   str  -  either a filename or database string
 * 
 * Returns:  A SEQFILE structure.
 */
SEQFILE *seqfopen2(char *str)
{
  if (!ctype_initflag)
    init_ctype();

  if ((str[0] == '-' && str[1] == '\0') || seqfisafile(str))
    return seqfopen(str, "r", NULL);
  else if (bioseq_check(str))
    return seqfopendb(str);
  else {
    set_error(E_OPENFAILED);
    print_error("%s:  No such file or database exists.\n", str);
    return NULL;
  }
}


/*
 * seqfclose
 *
 * Close the open file pointer (if not stdin or stdout), and free the 
 * SEQFILE structure (and its dynamically allocated elements).
 *
 * Parameters:   sfp  -  an opened SEQFILE structure
 * 
 * Returns:  nothing
 */
void seqfclose(SEQFILE *sfp)
{
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return, "seqfclose", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return, "seqfclose",
              "arg 1 is already closed");

  intseqf_close(isfp);
}


/*
 * seqfread
 *
 * Read the next entry in the sequence file/database into memory.
 *
 * Parameters:   sfp  - an opened SEQFILE structure
 *               flag - non-zero says read the next entry, zero for just
 *                      the next sequence.
 *
 * Returns:  a 0 if the read was successful, a -1 on EOF or error.
 */
int seqfread(SEQFILE *sfp, int flag)
{
  int status, offset;
  char *s;
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return -1, "seqfread", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return -1, "seqfread",
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return -1, "seqfread",
              "arg 1 is not open for reading");

  preverror_test(isfp->opstatus == OP_ERROR, return -1);
  eof_test(isfp->opstatus == OP_EOF, return -1);

  if (isfp->opstatus == OP_TEMPERR) {
    if (isfp->optype == OP_READ) {
      isfp->opstatus = OP_EOF;
      set_error(E_EOF);
      return -1;
    }

    while (1) {
      for (s=isfp->db_currentfile; *s; s++) ;
      isfp->db_currentfile = ++s;
      if (*s == '\0') {
        isfp->opstatus = OP_EOF;
        set_error(E_EOF);
        return -1;
      }

      status = intseqf_open(isfp, isfp->db_currentfile, isfp->db_format);
      switch (status) {
      case STATUS_OK:
      case STATUS_WARNING:
        isfp->opstatus = OP_ACTIVE;
        return 0;

      case STATUS_FATAL:
        isfp->opstatus = OP_ERROR;
        return -1;

      case STATUS_ERROR:
      case STATUS_EOF:
        set_error(E_DBFILEERROR);
        break;

      default:
        status_error(return -1, "seqfread");
      }
    }
  }

  /*
   * If we've already read the first entry (as part of the open), just return.
   */
  if (isfp->initreadflag) {
    isfp->initreadflag = 0;
    return 0;
  }

  /*
   * In the basic reading mode or if there are more sequences to read in the
   * current entry, just read the next sequence/entry. 
   *
   * In the random access mode (accessing single entries in a file), first
   * get the byte offset of the next entry to read, seek to that offset,
   * reset all of the file pointers and then read that next entry.
   */
  if (!isfp->randaccessflag ||
      (!flag && isfp->entry_seqno < isfp->entry_numseqs))
    status = intseqf_read(isfp, flag);
  else {
    if (isfp->currentoffset == isfp->num_offsets)
      status = STATUS_EOF;
    else {
      offset = isfp->byteoffsets[isfp->currentoffset++];
      status = seek_raw_file(isfp->input_fd, offset);
      error_test(status != STATUS_OK, E_READFAILED, return -1,
                 print_error("%s:  %s\n", isfp->filename, sys_errlist[errno]));

      isfp->fp_bytepos = offset;
      isfp->fp_current = isfp->fp_top = isfp->fp_buffer;
      isfp->fp_buffer[0] = '\n';
      isfp->isendtagged = 1;

      isfp->fp_entrystart = isfp->fp_seqstart = isfp->fp_entryend = NULL;
      isfp->entry_seqno = isfp->entry_numseqs = 0;
      status = intseqf_read(isfp, 1);
    }
  }

  switch (status) {
  case STATUS_OK:
  case STATUS_WARNING:
    return 0;

  case STATUS_EOF:
    if (isfp->optype == OP_READ) {
      set_error(E_EOF);
      isfp->opstatus = OP_EOF;
      return -1;
    }

    /*
     * If searching a database, close the current file and goto the next
     * file in the list.  Return the eof signal if no more files appear
     * in the list.
     */
#ifdef ISMAPABLE
    if (isfp->ismapped) {
      munmap(isfp->fp_buffer, isfp->mapsize);
      isfp->fp_buffer = NULL;
      isfp->fp_bufsize = 0;
      isfp->ismapped = 0;
    }
#endif
    close_raw_file(isfp->input_fd);
    isfp->openflag = 0;

    /*
     * Construct the correct path to the next file, open the file and
     * recursively call seqfread to get the first entry in that file.
     */
    while (1) {
      for (s=isfp->db_currentfile; *s; s++) ;
      isfp->db_currentfile = ++s;
      if (*s == '\0') {
        set_error(E_EOF);
        isfp->opstatus = OP_EOF;
        return -1;
      }

      status = intseqf_open(isfp, isfp->db_currentfile, isfp->db_format);
      isfp->initreadflag = 0;
      switch (status) {
      case STATUS_OK:
      case STATUS_WARNING:
        return 0;

      case STATUS_FATAL:
        isfp->opstatus = OP_ERROR;
        return -1;

      case STATUS_ERROR:
      case STATUS_EOF:
        set_error(E_DBFILEERROR);
        break;

      default:
        status_error(return -1, "seqfread");
      }
    }
    return -1;

  case STATUS_ERROR:
    isfp->opstatus = OP_TEMPERR;
    return -1;

  case STATUS_FATAL:
    isfp->opstatus = OP_ERROR;
    return -1;

  default:
    status_error(return -1, "seqfread");
  }
}


/*
 * seqfgetseq
 *
 * Read the next entry in the sequence file/database and then get it's
 * sequence.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *              length_out  -  the location to store the sequence length
 *                             (can be NULL)
 *              newbuffer   -  should a new, dynamically allocated buffer
 *                             be created to hold the sequence
 *
 * Returns:  the next entry's sequence.
 */
char *seqfgetseq(SEQFILE *sfp, int *length_out, int newbuffer)
{
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return NULL, "seqfgetseq", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return NULL, "seqfgetseq", 
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return NULL, "seqfgetseq",
              "arg 1 is not open for reading");

  if (seqfread(sfp, 0) == -1) {
    if (isfp->opstatus == OP_EOF || isfp->opstatus == OP_ERROR) 
      return NULL;
    else {
      if (length_out != NULL)
        *length_out = 0;
      return "";
    }
  }
  return seqfsequence(sfp, length_out, newbuffer);
}


/*
 * seqfgetrawseq
 *
 * Read the next entry in the sequence file/database and then get it's
 * raw sequence.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *              length_out  -  the location to store the sequence length
 *                             (can be NULL)
 *              newbuffer   -  should a new, dynamically allocated buffer
 *                             be created to hold the sequence
 *
 * Returns:  the next entry's raw sequence.
 */
char *seqfgetrawseq(SEQFILE *sfp, int *length_out, int newbuffer)
{
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return NULL, "seqfgetrawseq", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return NULL, "seqfgetrawseq", 
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return NULL, "seqfgetrawseq",
              "arg 1 is not open for reading");

  if (seqfread(sfp, 0) == -1) {
    if (isfp->opstatus == OP_EOF || isfp->opstatus == OP_ERROR) 
      return NULL;
    else {
      if (length_out != NULL)
        *length_out = 0;
      return "";
    }
  }
  return seqfrawseq(sfp, length_out, newbuffer);
}


/*
 * seqfgetentry
 *
 * Read the next entry in the sequence file/database and return it.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *              length_out  -  the location to store the entry's length
 *                             (can be NULL)
 *              newbuffer   -  should a new, dynamically allocated buffer
 *                             be created to hold the entry
 *
 * Returns:  the text of the next entry.
 */
char *seqfgetentry(SEQFILE *sfp, int *length_out, int newbuffer)
{
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return NULL, "seqfgetentry", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return NULL, "seqfgetentry", 
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return NULL, "seqfgetentry",
              "arg 1 is not open for reading");

  if (seqfread(sfp, 1) == -1) {
    if (isfp->opstatus == OP_EOF || isfp->opstatus == OP_ERROR) 
      return NULL;
    else {
      if (length_out != NULL)
        *length_out = 0;
      return "";
    }
  }
  return seqfentry(sfp, length_out, newbuffer);
}


/*
 * seqfgetinfo
 *
 * Read the next entry in the sequence file/database, parse it and
 * return the information found in the entry.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *              newbuffer   -  should a new, dynamically allocated buffer
 *                             be created to hold the entry
 *
 * Returns:  the information about the next entry.
 */
SEQINFO *seqfgetinfo(SEQFILE *sfp, int newbuffer)
{
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return NULL, "seqfgetinfo", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return NULL, "seqfgetinfo", 
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return NULL, "seqfgetinfo",
              "arg 1 is not open for reading");

  if (seqfread(sfp, 0) == -1)
    return NULL;
  return seqfinfo(sfp, newbuffer);
}


/*
 * seqfsequence & seqfrawseq
 *
 * Get the sequence for the current entry and return it.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *              length_out  -  the location to store the sequence length
 *                             (can be NULL)
 *              newbuffer   -  should a new, dynamically allocated buffer
 *                             be created to hold the sequence
 *              rawseqflag  -  GETSEQ_SEQUENCE for the basic sequence, or
 *                             GETSEQ_RAWSEQ for the raw seq.
 *                              (there is also GETSEQ_LENGTHS for the lengths
 *                               only, but that should not be used here)
 *              fnname      -  the name of the function.
 *
 * Returns:  the current entry's sequence.
 */
char *intseqf_seq(SEQFILE *sfp, int *length_out, int newbuffer,
                  int rawseqflag, char *fnname)
{
  int status;
  char *seq;
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return NULL, fnname, "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return NULL, fnname, 
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return NULL, fnname,
              "arg 1 is not open for reading");

  preverror_test(isfp->opstatus == OP_ERROR || isfp->opstatus == OP_TEMPERR,
                 return NULL);
  eof_test(isfp->opstatus == OP_EOF, return NULL);

  /*
   * If the sequence has already been read and the sequence in the buffer
   * is the desired sequence, just return it (or a copy of it).
   */
  if (isfp->isseqcurrent && isfp->rawseqflag == rawseqflag) {
    if (length_out != NULL)
      *length_out = isfp->seqlen;

    if (!newbuffer)
      return isfp->seq;
    else {
      seq = (char *) malloc(isfp->seqlen + 1);
      memory_error(seq == NULL, return NULL);
      memcpy(seq, isfp->seq, isfp->seqlen + 1);
      return seq;
    }
  }

  /*
   * Otherwise, allocate space for the sequence, if necessary, and then
   * call the getseq function.
   */
  if (isfp->seq == NULL) {
    isfp->seq = (char *) malloc(INIT_SEQSIZE);
    memory_error(isfp->seq == NULL, return NULL);
    isfp->seqsize = INIT_SEQSIZE;
  }
  isfp->seq[0] = '\0';
  isfp->seqlen = 0;

  status = (*file_table[isfp->format].getseq_fn)(isfp, rawseqflag);
  switch (status) {
  case STATUS_OK:
  case STATUS_WARNING:
    isfp->isseqcurrent = 1;
    isfp->rawseqflag = rawseqflag;
    if (length_out != NULL)
      *length_out = isfp->seqlen;
    seq = isfp->seq;
    if (newbuffer) {
      isfp->seq = NULL;
      isfp->seqlen = isfp->seqsize = 0;
      isfp->isseqcurrent = 0;
    }
    return seq;

  case STATUS_ERROR:
    if (length_out != NULL)
      *length_out = 0;
    return "";

  case STATUS_FATAL:
    return NULL;

  default:
    status_error(return NULL, fnname);
  }
}

char *seqfsequence(SEQFILE *sfp, int *length_out, int newbuffer)
{  return intseqf_seq(sfp, length_out, newbuffer,
                      GETSEQ_SEQUENCE, "seqfsequence");  }
char *seqfrawseq(SEQFILE *sfp, int *length_out, int newbuffer)
{  return intseqf_seq(sfp, length_out, newbuffer,
                      GETSEQ_RAWSEQ, "seqfrawseq");  }


/*
 * seqfentry
 *
 * Return the text for the current entry.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *              length_out  -  the location to store the sequence length
 *                             (can be NULL)
 *              newbuffer   -  should a new, dynamically allocated buffer
 *                             be created to hold the entry
 *
 * Returns:  the current entry's text.
 */
char *seqfentry(SEQFILE *sfp, int *length_out, int newbuffer)
{
  int len;
  char *buffer;
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return NULL, "seqfentry", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return NULL, "seqfentry",
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return NULL, "seqfentry",
              "arg 1 is not open for reading");

  preverror_test(isfp->opstatus == OP_ERROR || isfp->opstatus == OP_TEMPERR,
                 return NULL);
  eof_test(isfp->opstatus == OP_EOF, return NULL);

  len = isfp->fp_entryend - isfp->fp_entrystart;

  /*
   * If the file is mapped and a copy of the current entry is
   * stored in the writable buffer "mapentflag", just use that copy.
   */
  if (isfp->ismapped && isfp->mapentflag) {
    buffer = isfp->mapentry;
    if (newbuffer) {
      isfp->mapentflag = 0;
      isfp->mapentsize = 0;
      isfp->mapentry = NULL;
    }

    if (length_out != NULL)
      *length_out = len;
    return buffer;
  }

  /*
   * If a new buffer is requested, malloc the space and copy the text.
   * Otherwise, if the file is mapped, copy it into a writable buffer,
   * If not, just return a pointer into the internal buffer after
   * '\0'-terminating the entry.
   */
  if (newbuffer) {
    buffer = (char *) malloc(len + 1);
    memory_error(buffer == NULL, return NULL);
    memcpy(buffer, isfp->fp_entrystart, len);
    buffer[len] = '\0';

    if (length_out != NULL)
      *length_out = len;
    return buffer;
  }
  else if (isfp->ismapped) {
    if (isfp->mapentsize < len + 1) {
      isfp->mapentsize += len + 1;
      if (isfp->mapentry == NULL)
        isfp->mapentry = (char *) malloc(isfp->mapentsize);
      else
        isfp->mapentry = (char *) realloc(isfp->mapentry, isfp->mapentsize);

      if (isfp->mapentry == NULL) {
        isfp->mapentsize = 0;
        memory_error(1, return NULL);
      }
    }
    memcpy(isfp->mapentry, isfp->fp_entrystart, len);
    isfp->mapentry[len] = '\0';
    isfp->mapentflag = 1;

    if (length_out != NULL)
      *length_out = len;
    return isfp->mapentry;
  }
  else {
    if (*isfp->fp_entryend != '\0') {
      if (isfp->savech_loc != NULL)
        *isfp->savech_loc = isfp->savech;

      isfp->savech_loc = isfp->fp_entryend;
      isfp->savech = *isfp->fp_entryend;
      *isfp->fp_entryend = '\0';
    }

    if (length_out != NULL)
      *length_out = len;
    return isfp->fp_entrystart;
  }
}


/*
 * intseqf_field1
 *
 * The implementation of seqfinfo, seqfallinfo, seqfdate, seqfmainid,
 * seqfmainacc, seqfidlist, seqfdescription, seqfcomment and seqforganism.
 * The stub functions for those functions occur after the code for
 * intseqf_field1.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *              newbuffer   -  should a new, dynamically allocated buffer
 *                             be created to hold the sequence
 *              fnname      -  the name of the stub function
 *              field       -  the requested information field
 *
 * Returns:  the requested string (or SEQINFO structure)
 */
static char *intseqf_field1(SEQFILE *sfp, int newbuffer, char *fnname,
                            int field)
{
  char *s, *t, *id, *idend, *idlist, *temp;
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return NULL, fnname, "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return NULL, fnname,
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return NULL, fnname,
              "arg 1 is not open for reading");

  preverror_test(isfp->opstatus == OP_ERROR || isfp->opstatus == OP_TEMPERR,
                 return NULL);
  eof_test(isfp->opstatus == OP_EOF, return NULL);

  if (field != SEQINFO_MAINID && field != SEQINFO_MAINACC)
    return intseqf_info(isfp, newbuffer, field);
  else {
    idlist = intseqf_info(isfp, 0, SEQINFO_IDLIST);
    if (idlist == NULL)
      return NULL;

    id = idend = NULL;
    for (s=idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if ((field == SEQINFO_MAINID && !mystreq(t, 'A', "ACC:")) ||
          (field == SEQINFO_MAINACC && mystreq(t, 'A', "ACC:"))) {
        id = t;
        idend = s;
        break;
      }
      if (*s) s++;
    }
    if (id == NULL && field == SEQINFO_MAINID)
      for (id=idend=idlist; *idend && *idend == '|'; idend++) ;

    if (id == NULL)
      return NULL;

    temp = mystrdup2(id, idend);
    memory_error(temp == NULL, return NULL);

    if (!newbuffer) {
      if (isfp->idbuffer != NULL)
        free(isfp->idbuffer);
      isfp->idbuffer = temp;
    }

    return temp;
  }
}

SEQINFO *seqfinfo(SEQFILE *sfp, int newbuffer)
{  return (SEQINFO *) intseqf_field1(sfp, newbuffer, "seqfinfo",
                                     SEQINFO_ALL); }
SEQINFO *seqfallinfo(SEQFILE *sfp, int newbuffer)
{  return (SEQINFO *) intseqf_field1(sfp, newbuffer, "seqfallinfo",
                                     SEQINFO_ALLINFO); }
char *seqfdate(SEQFILE *sfp, int newbuffer)
{  return intseqf_field1(sfp, newbuffer, "seqfdate", SEQINFO_DATE); }
char *seqfmainid(SEQFILE *sfp, int newbuffer)
{  return intseqf_field1(sfp, newbuffer, "seqfmainid", SEQINFO_MAINID); }
char *seqfmainacc(SEQFILE *sfp, int newbuffer)
{  return intseqf_field1(sfp, newbuffer, "seqfmainacc", SEQINFO_MAINACC); }
char *seqfidlist(SEQFILE *sfp, int newbuffer)
{  return intseqf_field1(sfp, newbuffer, "seqfidlist", SEQINFO_IDLIST); }
char *seqfdescription(SEQFILE *sfp, int newbuffer)
{  return intseqf_field1(sfp, newbuffer, "seqfdescription",
                        SEQINFO_DESCRIPTION); }
char *seqfcomment(SEQFILE *sfp, int newbuffer)
{  return intseqf_field1(sfp, newbuffer, "seqfcomment", SEQINFO_COMMENT); }
char *seqforganism(SEQFILE *sfp, int newbuffer)
{  return intseqf_field1(sfp, newbuffer, "seqforganism", SEQINFO_ORGANISM); }


/*
 * intseqf_field2
 *
 * The implementation of seqfdbname, filename and format.  The stub
 * functions for those functions occur after the code for intseqf_field2.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *              newbuffer   -  should a new, dynamically allocated buffer
 *                             be created to hold the sequence
 *              fnname      -  the name of the stub function
 *              field       -  the requested information field
 *
 * Returns:  the requested string
 */
static char *intseqf_field2(SEQFILE *sfp, int newbuffer, char *fnname,
                            int field)
{
  int i;
  char *s;
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return NULL, fnname, "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return NULL, fnname,
              "arg 1 is not an open SEQFILE");

  preverror_test(isfp->opstatus == OP_ERROR || isfp->opstatus == OP_TEMPERR,
                 return NULL);
  eof_test(isfp->opstatus == OP_EOF, return NULL);

  s = NULL;
  switch (field) {
  case SEQINFO_DBNAME:    s = isfp->db_name;  break;
  case SEQINFO_FILENAME:  s = isfp->filename;  break;
  case SEQINFO_FORMAT:
    switch (isfp->format) {
    case FORMAT_GBFAST:  s = file_table[FORMAT_GENBANK].ident;  break;
    case FORMAT_PIRFAST:  s = file_table[FORMAT_PIR].ident;  break;
    case FORMAT_EMBLFAST:  s = file_table[FORMAT_EMBL].ident;  break;
    case FORMAT_SPFAST:  s = file_table[FORMAT_SPROT].ident;  break;
    case FORMAT_GCG:
      if (isfp->gcg_subformat == FORMAT_UNKNOWN)
        s = "GCG";
      else {
        for (i=0; i < gcg_table_size; i++)
          if (isfp->gcg_subformat == gcg_table[i].format)
            break;
        s = (i < gcg_table_size ? gcg_table[i].ident : "GCG");
      }
      break;

    default:
      s = file_table[isfp->format].ident;
      break;
    }
  }

  if (s != NULL && newbuffer) {
    s = mystrdup(s);
    memory_error(s == NULL, return NULL);
  }

  return s;
}

char *seqfdbname(SEQFILE *sfp, int newbuffer)
{  return intseqf_field2(sfp, newbuffer, "seqfdbname", SEQINFO_DBNAME); }
char *seqffilename(SEQFILE *sfp, int newbuffer)
{  return intseqf_field2(sfp, newbuffer, "seqffilename", SEQINFO_FILENAME); }
char *seqfformat(SEQFILE *sfp, int newbuffer)
{  return intseqf_field2(sfp, newbuffer, "seqfformat", SEQINFO_FORMAT); }


/*
 * intseqf_field3
 *
 * The implementation of seqfisfragment, seqfiscircular, seqfalphabet,
 * seqftruelen, seqfentryno, seqfseqno and seqfnumseqs.  The stub functions
 * for those functions occur after the code for intseqf_field3.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *              fnname      -  the name of the stub function
 *              field       -  the requested information field
 *
 * Returns:  the requested string
 */
static int intseqf_field3(SEQFILE *sfp, char *fnname, int field)
{
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return 0, fnname, "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return 0, fnname,
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return 0, fnname,
              "arg 1 is not open for reading");

  preverror_test(isfp->opstatus == OP_ERROR || isfp->opstatus == OP_TEMPERR,
                 return 0);
  eof_test(isfp->opstatus == OP_EOF, return 0);

  if (field == SEQINFO_ENTRYNO ||
      field == SEQINFO_SEQNO || field == SEQINFO_NUMSEQS ||
      (field == SEQINFO_TRUELEN && isfp->iflag_truelen) ||
      (field == SEQINFO_RAWLEN && isfp->iflag_rawlen))
    ;
  else if (intseqf_info(isfp, 0, field) == NULL)
    return 0;

  switch (field) {
  case SEQINFO_FRAGMENT:  return isfp->info->isfragment;
  case SEQINFO_CIRCULAR:  return isfp->info->iscircular;
  case SEQINFO_ALPHABET:  return isfp->info->alphabet;
  case SEQINFO_STARTPOS:  return isfp->info->fragstart;
  case SEQINFO_TRUELEN:   return isfp->entry_truelen;
  case SEQINFO_RAWLEN:    return isfp->entry_rawlen;
  case SEQINFO_ENTRYNO:   return isfp->entry_count;
  case SEQINFO_SEQNO:     return isfp->entry_seqno;
  case SEQINFO_NUMSEQS:   return isfp->entry_numseqs;
  default:                return 0;
  }
}

int seqfisfragment(SEQFILE *sfp)
{  return intseqf_field3(sfp, "seqfisfragment", SEQINFO_FRAGMENT); }
int seqfiscircular(SEQFILE *sfp)
{  return intseqf_field3(sfp, "seqfiscircular", SEQINFO_CIRCULAR); }
int seqfalphabet(SEQFILE *sfp)
{  return intseqf_field3(sfp, "seqfalphabet", SEQINFO_ALPHABET); }
int seqffragstart(SEQFILE *sfp)
{  return intseqf_field3(sfp, "seqffragstart", SEQINFO_STARTPOS); }
int seqftruelen(SEQFILE *sfp)
{  return intseqf_field3(sfp, "seqftruelen", SEQINFO_TRUELEN); }
int seqfrawlen(SEQFILE *sfp)
{  return intseqf_field3(sfp, "seqfrawlen", SEQINFO_RAWLEN); }
int seqfentryno(SEQFILE *sfp)
{  return intseqf_field3(sfp, "seqfentryno", SEQINFO_ENTRYNO); }
int seqfseqno(SEQFILE *sfp)
{  return intseqf_field3(sfp, "seqfseqno", SEQINFO_SEQNO); }
int seqfnumseqs(SEQFILE *sfp)
{  return intseqf_field3(sfp, "seqfnumseqs", SEQINFO_NUMSEQS); }


/*
 * seqfoneline
 *
 * Constructs a one-line description of the information given
 * in the SEQINFO structure.  That description is stored in the
 * given buffer.
 *
 * The description will always be NULL-terminated, and is guaranteed
 * to fit within the buffer length (including the NULL character).
 *
 * Parameter:  info      -  A SEQINFO structure containing information
 *             buffer    -  A character buffer to store the description
 *             buflen    -  The length of the buffer
 *             idonly    -  just return an identifier for the string
 *
 * Returns:  The length of the constructed description, or -1 on error.
 */
int seqfoneline(SEQINFO *info, char *buffer, int buflen, int idonly)
{
  int len, idlen, descrlen, orglen, taillen, fraglen;
  int flag, totallen, trunclen, templen, descrflag, orgflag;
  int midlen, descrslack, orgslack;
  char *s, *t, *t2, *t3, *t4, tailbuf[32], fragbuf[64];

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(info == NULL, return -1, "seqfoneline", "arg 1 is NULL");
  param_error(buffer == NULL, return -1, "seqfoneline", "arg 2 is NULL");
  param_error(buflen <= 0, return -1, "seqfoneline", 
              "arg 3 is not a positive length");

  /*
   * Take the case of just an identifier separately.  Return one of
   * the following (in order):  idlist, description, organism.
   */
  if (idonly) {
    if (info->idlist && info->idlist[0]) {
      s = info->idlist;
      for (t=s,len=0; *t && *t != '|'; t++,len++) ;

      if (buflen - 1 < len || mystreq(t, 'O', "OTH:")) {
        for ( ; s < t && *s != ':'; s++,len--) ;
        if (*s && s[1]) {
          s++;  len--;
        }
        else {
          s = info->idlist;
          len = t - s;
        }
      }
    }
    else if (((s = info->description) && info->description[0]) ||
             ((s = info->organism) && info->organism[0]))
      for (t=s,len=0; *t && !isspace(*t); t++,len++) ;
    else
      return 0;

    if (buflen - 1 < len)
      len = buflen - 1;

    memcpy(buffer, s, len);
    buffer[len] = '\0';
    return len;
  }

  /*
   * The general algorithm, first add one or two identifiers, compute the
   * lengths of the description, organism name, seqlen string and fragment
   * string.  Then, do one of the following things:
   *
   *    1) If only the identifiers fit, just add them.
   *    2) If everything fits, add everything.
   *    3) If 30 chars of the description, 20 chars of the organism and
   *       both the seqlen and fragment string fits, do a construction
   *       truncating the description and organism.
   *    4) Fill things in left to right until less than 15 chars available.
   */
  s = buffer;
  len = buflen - 1;

  /*
   * First, add one or two identifiers from idlist.
   */
  if (info->idlist && info->idlist[0]) {
    for (t2=t=info->idlist; *t2 && *t2 != '|'; t2++) ;
    idlen = t2 - t;

    if (len < idlen) {
      for ( ; t < t2 && *t != ':'; t++,idlen--) ;
      if (t + 1 < t2) {
        t++; idlen--;
      }
      else {
        t = info->idlist;
        idlen = t2 - t;
      }

      if (len < idlen) 
        idlen = len;

      if (idlen < len && t != info->idlist &&
          mystreq(info->idlist, 'A', "ACC:")) {
        *s++ = '~';  len--;
      }
      memcpy(s, t, idlen);
      s[idlen] = '\0';

      s += idlen;
      len -= idlen;
    }
    else if (!*t2) {
      strcpy(s, t);
      s += idlen;
      len -= idlen;
    }
    else {
      for (t3=t4=t2+1; *t4 && *t4 != '|'; t4++) ;
      error_test(t3 == t4, E_INVINFO, return -1,
                 print_error("seqfoneline:  `%s':  Invalid identifier "
                             "format.\n", info->idlist));

      if (len < t4 - t || len < 128) {
        memcpy(s, t, t2 - t);
        s[t2 - t] = '\0';
        s += t2 - t;
        len -= t2 - t;
      }
      else {
        memcpy(s, t, t4 - t);
        s += t4 - t;
        len -= t4 - t;
      }
    }
  }

  /*
   * Construct the seqlen and fragment strings.
   */
  descrlen = orglen = 0;
  if (info->description && info->description[0])
    descrlen = strlen(info->description);
  if (info->organism && info->organism[0])
    orglen = strlen(info->organism);

  taillen = fraglen = 0;
  if (info->truelen > 0) {
    sprintf(tailbuf, ", %d %s", info->truelen, 
            (info->alphabet == PROTEIN ? "aa" 
                : (info->alphabet == DNA || info->alphabet == RNA ? "bp" 
                                                                  : "ch")));
    taillen = strlen(tailbuf);

    if (info->isfragment || info->iscircular ||
        info->alphabet == DNA || info->alphabet == RNA) {
      t = fragbuf;
      *t++ = '(';
      flag = 0;
      if (info->iscircular) {
        strcpy(t, "circular");
        t += 8;
        flag = 1;
      }
      if (info->alphabet == DNA || info->alphabet == RNA) {
        if (flag)
          *t++ = ' ';
        strcpy(t, (info->alphabet == DNA ? "DNA" : "RNA"));
        t += 3;
        flag = 1;
      }
      if (info->isfragment) {
        if (info->fragstart > 0) {
          if (flag) {
            *t++ = ',';
            *t++ = ' ';
          }
          sprintf(t, "f. %d-%d", info->fragstart,
                  info->fragstart + info->truelen - 1);
          while (*t) t++;
        }
        else {
          if (flag)
            *t++ = ' ';
          strcpy(t, "fragment");
          t += 8;
        }
      }
      *t++ = ')';
      *t = '\0';
      fraglen = t - fragbuf;
    }
  }

  /*
   * Decide whether to do the truncated construction or the left to
   * right construction.
   */
  totallen = (descrlen ? descrlen + 1 : 0) + 
             (orglen ? orglen + 3 : 0) + taillen +
             (fraglen ? fraglen + 1 : 0) + 1;
  trunclen = (descrlen < 28 ? descrlen + 2 : 30) +
             (orglen < 16 ? orglen + 4 : 20) + taillen + 
             (fraglen ? fraglen + 1 : 0) + 1;
  if (totallen > len && trunclen <= len) {
    templen = len - taillen - (fraglen ? fraglen + 1 : 0) - 1;
    descrflag = orgflag = 0;
    if (descrlen && orglen) {
      midlen = templen * 6 / 10;
      descrslack = (descrlen < midlen - 1 ? midlen - descrlen - 1 : 0);
      orgslack = (orglen < templen - midlen - 3
                    ? (templen - midlen) - orglen - 3 : 0);

      if (descrlen > midlen + orgslack - 1) {
        descrlen = midlen + orgslack - 2;
        descrflag = 1;
      }
      if (orglen > (templen - midlen) + descrslack - 3) {
        orglen = (templen - midlen) + descrslack - 4;
        orgflag = 1;
      }
    }
    else if (descrlen && descrlen > templen - 1) {
      descrlen = templen - 2;
      descrflag = 1;
    }
    else if (orglen && orglen > templen - 3) {
      orglen = templen - 4;
      orgflag = 1;
    }
        
    if (descrlen) {
      *s++ = ' ';  len--;
      memcpy(s, info->description, descrlen);
      s += descrlen;
      len -= descrlen;
      if (descrflag) {
        *s++ = '*';  len--;
      }
    }
    if (orglen) {
      *s++ = ' ';  len--;
      *s++ = '-';  len--;
      *s++ = ' ';  len--;
      memcpy(s, info->organism, orglen);
      s += orglen;
      len -= orglen;
      if (orgflag) {
        *s++ = '*';  len--;
      }
    }
      
  }
  else {
    if (len >= 15 && descrlen > 0) {
      if (descrlen + 1 > len)
        descrlen = len - 2;
      *s++ = ' ';  len--;
      memcpy(s, info->description, descrlen);
      s += descrlen;
      len -= descrlen;
    }
    if (len >= 15 && orglen > 0) {
      if (orglen + 3 > len)
        orglen = len - 4;
      *s++ = ' ';  len--;
      *s++ = '-';  len--;
      *s++ = ' ';  len--;
      memcpy(s, info->organism, orglen);
      s += orglen;
      len -= orglen;
    }
  }

  if (len >= taillen) { 
    memcpy(s, tailbuf, taillen);
    s += taillen;
    len -= taillen;

    if (fraglen > 0 && len >= fraglen + 1) {
      *s++ = ' ';  len--;
      memcpy(s, fragbuf, fraglen);
      s += fraglen;
      len -= fraglen;
    }
    if (len > 0) {
      *s++ = '.';
      len--;
    }
  }

  *s = '\0';
  return s - buffer;
}


/*
 * seqfsetidpref
 *
 * Sets the identifier prefix used when reading entries.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *
 * Returns:  nothing
 */
void seqfsetidpref(SEQFILE *sfp, char *idprefix)
{
  int len;
  char *s;
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return, "seqfsetidpref", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return, "seqfsetidpref",
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return, "seqfsetidpref",
              "arg 1 is not open for reading");
  if (idprefix != NULL) {
    for (len=0,s=idprefix; len < 6 && *s; s++) ;
    param_error(len == 1, return, "seqfsetidpref", 
                "arg 2 is too short to be a valid identifier prefix");
    param_error(len > 4, return, "seqfsetidpref", 
                "arg 2 is too long to be a valid identifier prefix");
  }

  preverror_test(isfp->opstatus == OP_ERROR, return);
  eof_test(isfp->opstatus == OP_EOF, return);

  /*
   * Free an old idprefix, and then make a copy of the new one.
   */
  if (isfp->db_idprefix != NULL)
    free(isfp->db_idprefix);
  isfp->db_idprefix = NULL;

  if (idprefix != NULL && idprefix[0] != '\0') {
    isfp->db_idprefix = mystrdup(idprefix);
    for (s=isfp->db_idprefix; *s; s++)
      *s = tolower(*s);
    memory_error(isfp->db_idprefix == NULL, return);
  }
}


/*
 * seqfsetdbname
 *
 * Sets the database name used when reading entries.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *
 * Returns:  nothing
 */
void seqfsetdbname(SEQFILE *sfp, char *dbname)
{
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return, "seqfsetdbname", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return, "seqfsetdbname",
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return, "seqfsetdbname",
              "arg 1 is not open for reading");

  preverror_test(isfp->opstatus == OP_ERROR, return);
  eof_test(isfp->opstatus == OP_EOF, return);

  /*
   * Free an old dbname, and then make a copy of the new one.
   */
  if (isfp->db_name != NULL)
    free(isfp->db_name);
  isfp->db_name = NULL;

  if (dbname != NULL && dbname[0] != '\0') {
    isfp->db_name = mystrdup(dbname);
    memory_error(isfp->db_name == NULL, return);
  }
}


/*
 * seqfsetalpha
 *
 * Sets the alphabet used when reading entries.
 *
 * Parameters:  sfp         -  an opened SEQFILE structure
 *
 * Returns:  nothing
 */
void seqfsetalpha(SEQFILE *sfp, char *alphabet)
{
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return, "seqfsetalpha", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return, "seqfsetalpha",
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return, "seqfsetalpha",
              "arg 1 is not open for reading");

  preverror_test(isfp->opstatus == OP_ERROR, return);
  eof_test(isfp->opstatus == OP_EOF, return);

  /*
   * Free an old alphabet, and then make a copy of the new one.
   */
  if (isfp->db_alpha != NULL)
    free(isfp->db_alpha);
  isfp->db_alpha = NULL;

  if (alphabet != NULL && alphabet[0] != '\0') {
    isfp->db_alpha = mystrdup(alphabet);
    memory_error(isfp->db_alpha == NULL, return);
  }
}


/*
 *
 * seqfwrite
 *
 * Writes the given sequence and information to the output file
 * (in the specified format).
 *
 * Parameters:   sfp     - a SEQFILE structure opened for writing
 *               seq     - a sequence
 *               seqlen  - the sequence's length
 *               info    - information about the sequence
 *
 * Returns:  a 0 if the write was successful, a -1 on an error.
 *                                             (seqferrno is set on an error)
 */
int seqfwrite(SEQFILE *sfp, char *seq, int seqlen, SEQINFO *info)
{
  int status;
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return -1, "seqfwrite", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return -1, "seqfwrite",
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype != OP_WRITE, return -1, "seqfwrite",
              "arg 1 is not open for writing");
  param_error(seq == NULL, return -1, "seqfwrite", "arg 2 is NULL");
  param_error(seqlen <= 0, return -1, "seqfwrite",
              "arg 3 is less than or equal to 0");
  param_error(info == NULL, return -1, "seqfwrite", "arg 4 is NULL");

  preverror_test(isfp->opstatus == OP_ERROR, return -1);
  eof_test(isfp->opstatus == OP_EOF, return -1);

  status = (*file_table[isfp->format].putseq_fn)(isfp, seq, seqlen, info);
  switch (status) {
  case STATUS_OK:
  case STATUS_WARNING:
    return 0;

  case STATUS_EOF:
  case STATUS_ERROR:
  case STATUS_FATAL:
    isfp->opstatus = OP_ERROR;
    return -1;

  default:
    status_error(return -1, "seqfwrite");
  }
}


/*
 *
 * seqfconvert
 *
 * Write the sequence and entry information of the current entry to
 * the output file (in the specified format).
 *
 * Parameters:   sfpin  - a SEQFILE structure opened for reading
 *               sfpout - a SEQFILE structure opened for writing
 *
 * Returns:  a 0 if the read was successful, a -1 on an error.
 *                                            (seqferrno is set on an error)
 */
int seqfconvert(SEQFILE *sfpin, SEQFILE *sfpout)
{
  int seqlen, status;
  char *seq;
  SEQINFO *info;
  INTSEQFILE *isfpin, *isfpout;

  isfpin = (INTSEQFILE *) sfpin;
  isfpout = (INTSEQFILE *) sfpout;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfpin == NULL, return -1, "seqfconvert", "arg 1 is NULL");
  param_error(isfpin->opstatus == OP_FREED, return -1, "seqfconvert",
              "arg 1 is not an open SEQFILE");
  param_error(isfpin->optype == OP_WRITE, return -1, "seqfconvert",
              "arg 1 is not open for reading");
  param_error(isfpout == NULL, return -1, "seqfconvert", "arg 2 is NULL");
  param_error(isfpout->opstatus == OP_FREED, return -1, "seqfconvert",
              "arg 2 is not an open SEQFILE");
  param_error(isfpout->optype != OP_WRITE, return -1, "seqfconvert",
              "arg 1 is not open for writing");

  preverror_test(isfpin->opstatus == OP_ERROR ||
                 isfpin->opstatus == OP_TEMPERR, return -1);
  eof_test(isfpin->opstatus == OP_EOF, return -1);

  /*
   * Get the current sequence and info, then call the putseq function.
   */
  if (isfpin->isseqcurrent && !isfpin->rawseqflag) {
    seq = isfpin->seq;
    seqlen = isfpin->seqlen;
  }
  else {
    if ((seq = seqfsequence(sfpin, &seqlen, 0)) == NULL || seqlen == 0)
      return -1;
  }

  if (isfpin->istatus == INFO_ALL)
    info = isfpin->info;
  else {
    if ((info = seqfinfo(sfpin, 0)) == NULL)
      return -1;
  }

  status = (*file_table[isfpout->format].putseq_fn)(isfpout, seq, seqlen,
                                                    info);
  switch (status) {
  case STATUS_OK:
  case STATUS_WARNING:
    return 0;

  case STATUS_EOF:
  case STATUS_ERROR:
  case STATUS_FATAL:
    isfpout->opstatus = OP_ERROR;
    return -1;

  default:
    status_error(return -1, "seqfconvert");
  }
}


/*
 *
 * seqfputs
 *
 * Writes the given string to the output file
 *
 * Parameters:   sfp  - a SEQFILE structure opened for writing
 *               s    - a sequence
 *               len  - the sequence's length
 *
 * Returns:  a 0 if the write was successful, a -1 on an error.
 *                                             (seqferrno is set on an error)
 */
int seqfputs(SEQFILE *sfp, char *s, int len)
{
  int status;
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return -1, "seqfputs", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return -1, "seqfputs",
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype != OP_WRITE, return -1, "seqfputs",
              "arg 1 is not open for writing");
  param_error(s == NULL, return -1, "seqfputs", "arg 2 is NULL");
  param_error(len < 0, return -1, "seqfputs", "arg 3 is less than 0");

  preverror_test(isfp->opstatus == OP_ERROR, return -1);
  eof_test(isfp->opstatus == OP_EOF, return -1);

  if (len == 0) {
    fputs(s, isfp->output_fp);
    return 0;
  }
  else {
    status = fwrite(s, len, 1, isfp->output_fp);
    return (status == 1 ? 0 : -1);
  }
}


/*
 * seqfannotate
 *
 * This function adds extra comment text to an entry, as it's outputting that
 * entry.
 *
 * Parameters:      sfp        -  a SEQFILE structure open for writing
 *                  entry      -  an entry
 *                  entrylen   -  the length of the entry
 *                  newcomment -  the new comment to add to the entry
 *                  flag       -  should an existing comment be retained
 *
 * Returns:  0 on success and -1 on failure
 */
int seqfannotate(SEQFILE *sfp, char *entry, int entrylen, char *newcomment,
                 int flag)
{
  int status;
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return -1, "seqfannotate", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return -1, "seqfannotate",
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype != OP_WRITE, return -1, "seqfannotate",
              "arg 1 is not open for writing");
  param_error(entry == NULL, return -1, "seqfannotate", "arg 2 is NULL");
  param_error(entrylen <= 0, return -1, "seqfannotate",
              "arg 3 is less than or equal to 0");
  param_error(newcomment == NULL, return -1, "seqfannotate", "arg 4 is NULL");
  param_error(newcomment[0] == '\0', return -1, "seqfannotate", 
              "arg 4 is an empty string");

  error_test(file_table[isfp->format].annotate_fn == NULL,
             E_PARAMERROR, return -1,
             print_error("%s:  Cannot annotate entries in this file format.\n",
                         file_table[isfp->format].ident));

  /*
   * Call the annotate function.
   */
  status = (*file_table[isfp->format].annotate_fn)(isfp->output_fp, entry,
                                                   entrylen, newcomment, flag);
  switch (status) {
  case STATUS_OK:
  case STATUS_WARNING:
    return 0;

  case STATUS_ERROR:
  case STATUS_FATAL:
    return -1;

  default:
    status_error(return -1, "seqfannotate");
  }
}


/*
 * seqfgcgify
 *
 * Convert an entry into a GCG entry and output the converted text to
 * the given file.
 *
 * Parameters:  sfp       - A SEQFILE pointer open for writing
 *              entry     - The entry
 *              entrylen  - The entry's length
 *
 * Returns:  0 on success, -1 on EOF or error.
 */
int seqfgcgify(SEQFILE *sfpout, char *entry, int entrylen)
{
  int i, j, k, count, seqlen, checksum, alpha;
  int fmt, dna, others, oldpe, status;
  char ch, *s, *seq, *start, *end, *mainid, *date, buffer[128];
  FILE *fp;
  INTSEQFILE isfbuffer, *isfp;
  INTSEQFILE *isfpout = (INTSEQFILE *) sfpout;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfpout == NULL, return -1, "seqfgcgify", "arg 1 is NULL");
  param_error(isfpout->opstatus == OP_FREED, return -1, "seqfgcgify",
              "arg 1 is not an open SEQFILE");
  param_error(isfpout->optype != OP_WRITE, return -1, "seqfgcgify",
              "arg 1 is not open for writing");
  param_error(entry == NULL, return -1, "seqfgcgify", "arg 2 is NULL");
  param_error(entrylen <= 0, return -1, "seqfgcgify",
              "arg 3 is less than or equal to 0");

  error_test(isfpout->format != FORMAT_GCG, E_INVFORMAT, return -1,
             print_error("%s:  seqfgcgify:  Cannot write GCG entry when "
                         "output format specified as %s.\n",
                         isfpout->filename, seqfformat(isfpout, 0)));

  if (isfpout->entry_count != 0)
    return -1;

  /*
   * Construct an INTSEQFILE structure for the "read" procedure to
   * use while parsing the entry.
   */
  oldpe = pe_flag;
  pe_flag = PE_NONE;

  isfp = &isfbuffer;
  memset(isfp, 0, sizeof(INTSEQFILE));
  isfp->optype = OP_READ;
  isfp->opstatus = OP_ACTIVE;
  isfp->filename = "";
  isfp->fp_buffer = isfp->fp_current = entry;
  isfp->fp_top = entry + entrylen;

  if (isfpout->gcg_subformat != FORMAT_UNKNOWN)
    isfp->format = isfpout->gcg_subformat;
  else {
    status = determine_format(isfp);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      pe_flag = oldpe;
      raise_error(E_PARSEERROR, return -1,
                  print_error("%s:  seqfgcgify:  Parse error while scanning "
                              "entry to output.\n", isfpout->filename));
    }
  }

  /*
   * Reset the error mode to not perform any error actions and then
   * call the read function to parse the entry.
   */
  status = (*file_table[isfp->format].read_fn)(isfp, 0);
  pe_flag = oldpe;

  error_test(status != STATUS_OK && status != STATUS_WARNING,
             E_PARSEERROR, return -1,
             print_error("%s:  seqfgcgify:  Parse error while scanning "
                         "entry to output.\n", isfpout->filename));

  error_test(isfp->fp_seqstart == NULL, E_PARSEERROR, return -1,
             print_error("%s:  seqfgcgify:  Parse error while scanning "
                         "entry to output.\n", isfpout->filename));

  /*
   * Output the header lines.
   */
  isfpout->entry_count++;
  fp = isfpout->output_fp;
  fmt = isfp->format;

  if (isfp->format == FORMAT_GCG) {
    fwrite(entry, entrylen, 1, fp);
    return STATUS_OK;
  }

  fwrite(isfp->fp_entrystart, isfp->fp_seqstart - isfp->fp_entrystart, 1, fp);
  if ((fmt == FORMAT_NBRF || fmt == FORMAT_NBRFOLD) && isfp->nbrf_header) {
    fwrite(isfp->nbrf_header, isfp->fp_entryend - isfp->nbrf_header, 1, fp);
    if (*(isfp->fp_entryend - 1) != '\n')
      fputc('\n', fp);
  }

  /*
   * Find the sequence length, the alphabet type and the checksum.
   */
  start = isfp->fp_seqstart;
  end = isfp->fp_entryend;
  if (fmt == FORMAT_GENBANK || fmt == FORMAT_PIR ||
      fmt == FORMAT_SPROT || fmt == FORMAT_EMBL) {
    if (*(end - 1) == '\n')
      end--;
    while (*(end - 1) != '\n') end--;
  }
  else if (fmt == FORMAT_NBRF && isfp->nbrf_header &&
           isfp->nbrf_header < isfp->fp_entryend)
    end = isfp->nbrf_header;

  checksum = seqlen = 0;
  dna = others = 0;
  for (s=start; s < end; s++) {
    if (!isspace(*s) && !isdigit(*s)) {
      if (((fmt == FORMAT_NBRF || fmt == FORMAT_NBRFOLD) && *s == '*') ||
          ((fmt == FORMAT_STANFORD || fmt == FORMAT_STANFORDOLD) &&
           (*s == '1' || *s == '2')))
        break;
      else if (fmt == FORMAT_PIR && !isalpha(*s))
        continue;

      ch = (*s == '-' ? '.' : toupper(*s));
      checksum += (seqlen % 57 + 1) * ch;
      seqlen++;

      if (ch == 'A' || ch == 'G' || ch == 'C' || ch == 'T' || ch == 'U')
        dna++;
      else if (!isalpha(*s))
        others++;
    }
  }
  checksum %= 10000;

  if (((float) dna / (float) (seqlen - others)) >= 0.85)
    alpha = DNA;
  else
    alpha = PROTEIN;

  /*
   * Output the gcg info line.
   */
  fputc('\n', fp);
  fputs("  ", fp);
  if ((mainid = seqfmainid((SEQFILE *) isfp, 0)) != NULL)
    fputs(mainid, fp);
  fputs("  ", fp);

  fprintf(fp, "Length: %d  ", seqlen);

  date = get_today();
  for (i=1; i <= 12; i++)
    if (myncasecmp(date+3, months[i], 3) == 0)
      break;
  if (i <= 12)
    fprintf(fp, "%s %c%c, %s %s  ", gcg_full_months[i], date[0], date[1],
            date+7, date+12);

  if (alpha == DNA)
    fputs("Type: N  ", fp);
  else if (alpha == PROTEIN)
    fputs("Type: P  ", fp);

  fprintf(fp, "Check: %d  ..\n\n", checksum);

  /*
   * Print the gcg sequence lines.
   */
  s = NULL;
  j = k = 0;
  count = 1;
  for (seq=start; seq < end; seq++) {
    if (!isspace(*seq) && !isdigit(*seq)) {
      if (((fmt == FORMAT_NBRF || fmt == FORMAT_NBRFOLD) && *seq == '*') ||
          ((fmt == FORMAT_STANFORD || fmt == FORMAT_STANFORDOLD) &&
           (*seq == '1' || *seq == '2')))
        break;
      else if (fmt == FORMAT_PIR && !isalpha(*seq))
        continue;

      if (j == 0 && k == 0) {
        sprintf(buffer, "%8d  ", count);
        s = buffer + 9;
      }

      *s++ = (*seq == '-' ? '.' : *seq);
      count++;

      if (++k == 10) {
        *s++ = ' ';
        k = 0;
        if (++j == 5) {
          *s++ = '\n';
          *s++ = '\n';
          *s = '\0';
          fputs(buffer, fp);
          j = 0;
        }
      }
    }
  }
  if (j != 0 || k != 0) {
    *s++ = '\n';
    *s++ = '\n';
    *s = '\0';
    fputs(buffer, fp);
  }

  /*
   * Free up the SEQINFO structure that may have been allocated when
   * the main identifier was extracted.
   */
  if (isfp->info != NULL)
    free(isfp->info);

  return 0;
}


/*
 * seqfungcgify
 *
 * Convert an entry from a GCG entry and output the converted text to
 * the given file.
 *
 * Parameters:  sfp       - A SEQFILE structure open for writing
 *              entry     - The entry
 *              entrylen  - The entry's length
 *
 * Returns:  0 on success, -1 on error.
 */
int seqfungcgify(SEQFILE *sfpout, char *entry, int entrylen)
{
  int i, j, k, count, fmt, status, oldpe;
  char ch, *s, *seq, *end;
  FILE *fp;
  INTSEQFILE isfbuffer, *isfp;
  INTSEQFILE *isfpout = (INTSEQFILE *) sfpout;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfpout == NULL, return -1, "seqfungcgify", "arg 1 is NULL");
  param_error(isfpout->opstatus == OP_FREED, return -1, "seqfungcgify",
              "arg 1 is not an open SEQFILE");
  param_error(isfpout->optype != OP_WRITE, return -1, "seqfungcgify",
              "arg 1 is not open for writing");
  param_error(entry == NULL, return -1, "seqfungcgify", "arg 2 is NULL");
  param_error(entrylen <= 0, return -1, "seqfungcgify",
              "arg 3 is less than or equal to 0");

  for (i=0; i < gcg_table_size; i++)
    if (isfpout->format == gcg_table[i].format &&
        gcg_table[i].format != FORMAT_MSF)
      break;
  error_test(i == gcg_table_size && isfpout->format != FORMAT_RAW,
             E_INVFORMAT, return -1,
             print_error("%s:  seqfungcgify:  Cannot convert GCG entry to "
                         "output format %s.\n", isfpout->filename,
                         seqfformat(isfpout, 0)));

  /*
   * Construct an INTSEQFILE structure for the "read" procedure to
   * use while parsing the entry.
   */
  isfp = &isfbuffer;
  memset(isfp, 0, sizeof(INTSEQFILE));
  isfp->optype = OP_READ;
  isfp->opstatus = OP_ACTIVE;
  isfp->filename = "";

  isfp->format = FORMAT_GCG;
  isfp->gcg_subformat = isfpout->format;
  isfp->fp_buffer = isfp->fp_current = entry;
  isfp->fp_top = entry + entrylen;

  /*
   * Reset the error mode to not perform any error actions and then
   * call the read function to parse the entry.
   */
  oldpe = pe_flag;
  pe_flag = PE_NONE;
  status = (*file_table[isfpout->format].read_fn)(isfp, 0);
  pe_flag = oldpe;

  error_test(status != STATUS_OK && status != STATUS_WARNING,
             E_PARSEERROR, return -1,
             print_error("%s:  seqfungcgify:  Parse error while scanning "
                         "entry to output.\n", isfpout->filename));

  error_test(isfp->fp_seqstart == NULL || isfp->gcg_infoline == NULL,
             E_PARSEERROR, return -1,
             print_error("%s:  seqfungcgify:  Parse error while scanning "
                         "entry to output.\n", isfpout->filename));

  /*
   * Output the header lines.
   */
  fp = isfpout->output_fp;
  fmt = isfpout->format;
  isfpout->entry_count++;

  if (fmt != FORMAT_RAW && fmt != FORMAT_PLAIN) {
    if (fmt != FORMAT_NBRF || isfp->nbrf_header == NULL)
      for (s=isfp->gcg_infoline; *(s-2) == '\n'; s--) ;
    else
      s = isfp->nbrf_header;

    fwrite(isfp->fp_entrystart, s - isfp->fp_entrystart, 1, fp);
  }

  /*
   * Output the sequence.
   */
  seq = isfp->fp_seqstart;
  end = isfp->fp_entryend;

  switch (fmt) {
  case FORMAT_RAW:
    for ( ; seq < end; seq++) {
      if (*seq == '>' || *seq == '<' || *seq == '$') {
        for (ch=*seq++; seq < end && *seq != ch; seq++) ;
      }
      else if (!isspace(*seq) && !isdigit(*seq)) {
        ch = (*seq == '.' ? '-' : *seq);
        fputc(ch, fp);
      }
    }
    break;
    
  case FORMAT_PLAIN:
  case FORMAT_FASTA:
  case FORMAT_FASTAOLD:
  case FORMAT_STANFORD:
  case FORMAT_STANFORDOLD:
  case FORMAT_NBRF:
  case FORMAT_NBRFOLD:
    count = 0;
    for ( ; seq < end; seq++) {
      if (*seq == '>' || *seq == '<' || *seq == '$') {
        for (ch=*seq++; seq < end && *seq != ch; seq++) ;
      }
      else if (!isspace(*seq) && !isdigit(*seq)) {
        ch = (*seq == '.' ? '-' : *seq);

        if (count == 60) {
          fputc('\n', fp);
          count = 0;
        }
        fputc(ch, fp);
        count++;
      }
    }

    if (fmt == FORMAT_STANFORD || fmt == FORMAT_STANFORDOLD)
      fputc('1', fp);
    else if (fmt == FORMAT_NBRF || fmt == FORMAT_NBRFOLD)
      fputc('*', fp);

    if (count != 0)
      fputc('\n', fp);

    if ((fmt == FORMAT_NBRF || fmt == FORMAT_NBRFOLD) &&
        isfp->nbrf_header != NULL) {
      for (s=isfp->gcg_infoline; *(s-2) == '\n'; s--) ;
      fwrite(isfp->nbrf_header, s - isfp->nbrf_header, 1, fp);
    }
    break;

  case FORMAT_GENBANK:
    j = k = count = 0;
    for ( ; seq < end; seq++) {
      if (*seq == '>' || *seq == '<' || *seq == '$') {
        for (ch=*seq++; seq < end && *seq != ch; seq++) ;
      }
      else if (!isspace(*seq) && !isdigit(*seq)) {
        ch = (*seq == '.' ? '-' : *seq);

        if (j == 0 && k == 0)
          fprintf(fp, "   %6d", count+1);
        if (k == 0)
          fputc(' ', fp);

        fputc(ch, fp);
        count++;

        if (++k == 10) {
          k = 0;
          if (++j == 6) {
            fputc('\n', fp);
            j = 0;
          }
        }
      }
    }
    if (j != 0 || k != 0)
      fputc('\n', fp);
    fputs("//\n", fp);
    break;

  case FORMAT_PIR:
    fputs("                5        10        15"
          "        20        25        30\n", fp);
    j = count = 0;
    for ( ; seq < end; seq++) {
      if (*seq == '>' || *seq == '<' || *seq == '$') {
        for (ch=*seq++; seq < end && *seq != ch; seq++) ;
      }
      else if (!isspace(*seq) && !isdigit(*seq)) {
        ch = (*seq == '.' ? '-' : *seq);

        if (j == 0)
          fprintf(fp, "%7d", count + 1);

        fputc(' ', fp);
        fputc(ch, fp);
        count++;

        if (++j == 30) {
          j = 0;
          fputc('\n', fp);
        }
      }
    }
    if (j != 0)
      fputc('\n', fp);
    fputs("///\n", fp);
    break;

  case FORMAT_EMBL:
    j = k = count = 0;
    for ( ; seq < end; seq++) {
      if (*seq == '>' || *seq == '<' || *seq == '$') {
        for (ch=*seq++; seq < end && *seq != ch; seq++) ;
      }
      else if (!isspace(*seq) && !isdigit(*seq)) {
        ch = (*seq == '.' ? '-' : *seq);

        if (j == 0 && k == 0)
          fputs("    ", fp);
        if (k == 0)
          fputc(' ', fp);

        fputc(ch, fp);
        count++;

        if (++k == 10) {
          k = 0;
          if (++j == 6) {
            fprintf(fp, "%10d\n", count);
            j = 0;
          }
        }
      }
    }
    if (j != 0 || k != 0) {
      while (j != 0 && k != 0) {
        if (k == 0)
          fputc(' ', fp);
        fputc(' ', fp);
        if (++k == 10) {
          k = 0;
          if (++j == 6) {
            fprintf(fp, "%10d\n", count);
            j = 0;
          }
        }
      }
    }
    fputs("//\n", fp);
    break;

  case FORMAT_SPROT:
    j = k = 0;
    for ( ; seq < end; seq++) {
      if (*seq == '>' || *seq == '<' || *seq == '$') {
        for (ch=*seq++; seq < end && *seq != ch; seq++) ;
      }
      else if (!isspace(*seq) && !isdigit(*seq)) {
        ch = (*seq == '.' ? '-' : *seq);

        if (j == 0 && k == 0)
          fputs("    ", fp);
        if (k == 0)
          fputc(' ', fp);

        fputc(ch, fp);

        if (++k == 10) {
          k = 0;
          if (++j == 6) {
            fputc('\n', fp);
            j = 0;
          }
        }
      }
    }
    if (j != 0 || k != 0)
      fputc('\n', fp);
    fputs("//\n", fp);
    break;

  default:
    status_error(return -1, "seqfungcgify");
  }

  return 0;
}

  
/*
 * seqfparseent
 *
 * This function parses a given entry and constructs a SEQINFO structure
 * containing the information about that entry.
 *
 * Parameters:      entry     -  an entry
 *                  entrylen  -  the length of the entry
 *                  format    -  the file format for the entry
 *
 * Returns:  a SEQINFO structure, or NULL on an error
 */
SEQINFO *seqfparseent(char *entry, int entrylen, char *format)
{
  int i, status, gcgflag;
  INTSEQFILE istruct;
  INTSEQFILE *isfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(entry == NULL, return NULL, "seqfparseent", "arg 1 is NULL");
  param_error(entrylen <= 0, return NULL, "seqfparseent",
              "arg 2 is less than or equal to 0");
  param_error(format == NULL, return NULL, "seqfparseent", "arg 3 is NULL");
  param_error(format[0] == '\0', return NULL, "seqfparseent",
              "arg 3 is an empty string");

  error_test(!seqfcanparseent(format), E_PARAMERROR, return NULL,
             print_error("%s:  Cannot parse entries in this file format.\n",
                         format));

  /*
   * Figure out the specified format.
   */
  if (myncasecmp(format, "GCG-", 4) == 0) {
    for (i=0; i < gcg_table_size; i++)
      if (mycasecmp(format+4, gcg_table[i].ident+4) == 0)
        break;
    i = gcg_table[i].format;
    gcgflag = 1;
  }
  else {
    for (i=0; i < file_table_size; i++)
      if (mycasecmp(format, file_table[i].ident) == 0)
        break;
    i = file_table[i].format;
    gcgflag = 0;
  }

  /*
   * Create a dummy INTSEQFILE struct to give to the getinfo function.
   */
  isfp = &istruct;
  memset(isfp, 0, sizeof(INTSEQFILE));

  isfp->optype = OP_READ;
  isfp->filename = "";
  if (!gcgflag) {
    isfp->format = i;
    isfp->gcg_subformat = FORMAT_UNKNOWN;
  }
  else {
    isfp->format = FORMAT_GCG;
    isfp->gcg_subformat = i;
  }
  
  isfp->infosize = isfp->infobufsize = sizeof(SEQINFO);
  isfp->info = (SEQINFO *) malloc(isfp->infobufsize);
  memory_error(isfp->info == NULL, return NULL);
  memset(isfp->info, isfp->infosize, 0);

  isfp->iflag_rawlen = isfp->iflag_truelen = 1;

  /*
   * Call the getinfo function.
   */
  status = (*file_table[i].getinfo_fn)(isfp, entry, entrylen, SEQINFO_ALL);
  switch (status) {
  case STATUS_OK:
  case STATUS_WARNING:
    return isfp->info;

  case STATUS_ERROR:
  case STATUS_FATAL:
    if (isfp->info != NULL)
      free(isfp->info);
    return NULL;

  default:
    status_error(return NULL, "seqfparseent");
  }
}


/*
 * seqfsetpretty
 *
 * This function sets the pretty flag used by the Plain, FASTA, NBRF and
 * IG/Stanford putseq functions.
 *
 * Parameters:      sfp    -  a SEQFILE structure open for writing
 *                  value  -  value to set the pretty flag to
 *
 * Returns:  nothing.
 */
void seqfsetpretty(SEQFILE *sfp, int value)
{
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return, "seqfsetpretty", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return, "seqfsetpretty",
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype != OP_WRITE, return, "seqfsetpretty",
              "arg 1 is not open for writing");

  isfp->prettyflag = (value != 0 ? 2 : 1);
}


/*
 * seqfisaformat
 *
 * This function tests whether the given string is a valid file format.
 *
 * Parameters:      format  -  a file format string
 *
 * Returns:  non-zero if the string is a file format, zero otherwise.
 */
int seqfisaformat(char *format)
{
  int i;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(format == NULL, return 0, "seqfisaformat", "arg 1 is NULL");
  param_error(format[0] == '\0', return 0, "seqfisaformat",
              "arg 1 is an empty string");
  
  /*
   * Test the format string.
   */
  if (myncasecmp(format, "GCG-", 4) == 0) {
    for (i=0; i < gcg_table_size; i++)
      if (mycasecmp(format + 4, gcg_table[i].ident + 4) == 0)
        break;

    return (i < gcg_table_size);
  }
  else {
    for (i=0; i < file_table_size; i++)
      if (mycasecmp(format, file_table[i].ident) == 0)
        break;

    return (i < file_table_size);
  }
}


/*
 * seqffmttype
 *
 * This function returns some type information about a format.
 *
 * Parameters:      format  -  a file format string
 *
 * Returns:  a type define value if the string is a file format,
 *           zero otherwise.
 */
int seqffmttype(char *format)
{
  int i;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(format == NULL, return T_INVFORMAT, "seqffmttype",
              "arg 1 is NULL");
  param_error(format[0] == '\0', return T_INVFORMAT, "seqffmttype",
              "arg 1 is an empty string");
  
  /*
   * Test the format string.
   */
  if (myncasecmp(format, "GCG-", 4) == 0) {
    for (i=0; i < gcg_table_size; i++)
      if (mycasecmp(format + 4, gcg_table[i].ident + 4) == 0)
        break;

    if (i < gcg_table_size)
      return file_table[gcg_table[i].format].type;
  }
  else {
    for (i=0; i < file_table_size; i++)
      if (mycasecmp(format, file_table[i].ident) == 0)
        break;

    if (i < file_table_size)
      return file_table[file_table[i].format].type;
  }

  set_error(E_INVFORMAT);
  return T_INVFORMAT;
}


/*
 * seqfcanwrite
 *
 * This function tests whether the format has a putseq function.
 *
 * Parameters:      format  -  a file format string
 *
 * Returns:  non-zero if the string is a file format with a putseq
             function, zero otherwise.
 */
int seqfcanwrite(char *format)
{
  int i;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(format == NULL, return 0, "seqfcanwrite", "arg 1 is NULL");
  param_error(format[0] == '\0', return 0, "seqfcanwrite",
              "arg 1 is an empty string");
  
  /*
   * Test the format string.
   */
  if (myncasecmp(format, "GCG-", 4) == 0) {
    for (i=0; i < gcg_table_size; i++)
      if (mycasecmp(format + 4, gcg_table[i].ident + 4) == 0)
        break;

    if (i < gcg_table_size)
      return (file_table[gcg_table[i].format].putseq_fn != NULL);
  }
  else {
    for (i=0; i < file_table_size; i++)
      if (mycasecmp(format, file_table[i].ident) == 0)
        break;

    if (i < file_table_size)
      return (file_table[file_table[i].format].putseq_fn != NULL);
  }

  set_error(E_INVFORMAT);
  return 0;
}


/*
 * seqfcanannotate
 *
 * This function tests whether the format has an annotate function.
 *
 * Parameters:      format  -  a file format string
 *
 * Returns:  non-zero if the string is a file format with an annotate
             function, zero otherwise.
 */
int seqfcanannotate(char *format)
{
  int i;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(format == NULL, return 0, "seqfcanannotate", "arg 1 is NULL");
  param_error(format[0] == '\0', return 0, "seqfcanannotate",
              "arg 1 is an empty string");
  
  /*
   * Test the format string.
   */
  if (myncasecmp(format, "GCG-", 4) == 0) {
    for (i=0; i < gcg_table_size; i++)
      if (mycasecmp(format + 4, gcg_table[i].ident + 4) == 0)
        break;

    if (i < gcg_table_size)
      return (file_table[gcg_table[i].format].annotate_fn != NULL);
  }
  else {
    for (i=0; i < file_table_size; i++)
      if (mycasecmp(format, file_table[i].ident) == 0)
        break;

    if (i < file_table_size)
      return (file_table[file_table[i].format].annotate_fn != NULL);
  }

  set_error(E_INVFORMAT);
  return 0;
}


/*
 * seqfcanparseent
 *
 * This function tests whether the format is parsable from the raw entry
 * text.
 *
 * Parameters:      format  -  a file format string
 *
 * Returns:  non-zero if the string is parseable, zero otherwise.
 */
int seqfcanparseent(char *format)
{
  int i;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(format == NULL, return 0, "seqfcanparseent", "arg 1 is NULL");
  param_error(format[0] == '\0', return 0, "seqfcanparseent",
              "arg 1 is an empty string");
  
  /*
   * Test the format string.
   */
  if (myncasecmp(format, "GCG-", 4) == 0) {
    for (i=0; i < gcg_table_size; i++)
      if (mycasecmp(format + 4, gcg_table[i].ident + 4) == 0)
        break;

    if (i < gcg_table_size) {
      if (gcg_table[i].format == FORMAT_MSF)
        return 0;
      else
        return 1;
    }
  }
  else {
    for (i=0; i < file_table_size; i++)
      if (mycasecmp(format, file_table[i].ident) == 0)
        break;

    if (i < file_table_size) {
      if (i == FORMAT_FOUT || i == FORMAT_PHYLIP || i == FORMAT_PHYSEQ ||
          i == FORMAT_PHYINT || i == FORMAT_CLUSTAL || i == FORMAT_MSF)
        return 0;
      else
        return 1;
    }
  }

  set_error(E_INVFORMAT);
  return 0;
}


/*
 * seqfcangcgify
 *
 * This function tests whether the seqfgcgify and seqfungcgify can be
 * used with entries in the given format.
 *
 * Parameters:      format  -  a file format string
 *
 * Returns:  non-zero if the format is gcgify'able, zero otherwise.
 */
int seqfcangcgify(char *format)
{
  int i;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(format == NULL, return 0, "seqfcangcgify", "arg 1 is NULL");
  param_error(format[0] == '\0', return 0, "seqfcangcgify",
              "arg 1 is an empty string");
  
  /*
   * Test the format string.
   */
  if (myncasecmp(format, "GCG-", 4) == 0)
    format += 4;

  for (i=0; i < gcg_table_size; i++)
    if (mycasecmp(format, gcg_table[i].ident + 4) == 0)
      break;

  if (i == gcg_table_size || gcg_table[i].format == FORMAT_MSF)
    return 0;
  else
    return 1;
}


/*
 * seqfbytepos
 *
 * This function returns the byte position of the current entry when reading.
 *
 * Parameters:      sfp  -  An open SEQFILE structure
 *
 * Returns:  The byte position, or a -1 on an error or if there is no
 *           current entry.
 */
int seqfbytepos(SEQFILE *sfp)
{
  INTSEQFILE *isfp = (INTSEQFILE *) sfp;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(isfp == NULL, return -1, "seqfbytepos", "arg 1 is NULL");
  param_error(isfp->opstatus == OP_FREED, return -1, "seqfbytepos",
              "arg 1 is not an open SEQFILE");
  param_error(isfp->optype == OP_WRITE, return -1, "seqfbytepos",
              "arg 1 is not open for reading");

  preverror_test(isfp->opstatus == OP_ERROR, return -1);
  preverror_test(isfp->opstatus == OP_TEMPERR, return -1);
  eof_test(isfp->opstatus == OP_EOF, return -1);

  if (isfp->fp_entrystart == NULL)
    return -1;
  else
    return isfp->fp_bytepos + (isfp->fp_entrystart - isfp->fp_buffer);
}


/*
 * seqfisafile
 *
 * This function checks an input string to see if it's an existing file.
 * This function is needed because a valid filename may contain a '@'
 * suffix specifying the entries in the file to get.  Thus, the file check
 * has to strip off that suffix before calling stat.
 *
 * Parameters:  file  -  A filename
 *
 * Returns:  Non-zero if the file specifies a valid and existing file,
 *           zero otherwise.
 */
int seqfisafile(char *file)
{
  char *s;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(file == NULL, return 0, "seqfisafile", "arg 1 is NULL");
  for (s=file; *s && *s != '@'; s++) ;

  return isa_file(get_truename(file, s));
}




/*
 *
 * 
 * Internal procedures to do some of the grunge work of the interface
 * procedures.
 *
 *    intseqf_open, intseqf_open_for_writing, determine_format,
 *    intseqf_close, intseqf_read, intseqf_info
 *
 *
 *
 */

/*
 * intseqf_open
 *
 * Opens the specified file for reading, determines the file format (if
 * not specified), sets the appropriate INTSEQFILE fields and reads the
 * first entry.
 *
 * If the filename parameter is the string "-", then standard input
 * is used.
 *
 * If the format parameter is NULL, the file format is determined from
 * the first line of the first that begins with a non-whitespace character.
 *
 * Returns STATUS_ERROR when the file can't be opened or read, but
 * returns STATUS_FATAL on a malloc error or if "format" isn't valid.
 *
 * Parameters:      isfp      -  an INTSEQFILE structure
 *                  filename  -  the name of the file to open for reading
 *                  format    -  the file format to use (may be NULL)
 *
 * Returns:  a STATUS value
 */
static int intseqf_open(INTSEQFILE *isfp, char *filename, char *format)
{
  int id, status, subid, offset;
  char *s, *offset_string;

  id = subid = FORMAT_UNKNOWN;

  /*
   * Figure out the format, if given.
   */
  if (format != NULL) {
    if (myncasecmp(format, "GCG-", 4) == 0) {
      for (id=0; id < gcg_table_size; id++)
        if (mycasecmp(format + 4, gcg_table[id].ident + 4) == 0)
          break;

      error_test(id == gcg_table_size, E_INVFORMAT, return STATUS_FATAL,
                 print_error("Read Error:  `%s' is not a valid file format.\n",
                             format));

      subid = gcg_table[id].format;
      id = (subid == FORMAT_MSF ? FORMAT_MSF : FORMAT_GCG);
    }
    else {
      for (id=0; id < file_table_size; id++)
        if (mycasecmp(format, file_table[id].ident) == 0)
          break;

      error_test(id == file_table_size, E_INVFORMAT, return STATUS_FATAL,
                 print_error("Read Error:  `%s' is not a valid file format.\n",
                             format));

      subid = FORMAT_UNKNOWN;
      id = file_table[id].format;
    }
  }

  /*
   * Open the specified file or standard input, check for random access
   * mode and make a local copy of the filename.
   */
  if (isfp->filename != NULL)
    free(isfp->filename);

  if (filename[0] == '-' && filename[1] == '\0') {
    status = open_raw_stdin(&isfp->input_fd);
    error_test(status != STATUS_OK, E_OPENFAILED, return STATUS_ERROR,
               print_error("Read Error:  Standard input is not available.\n"));

    isfp->filename = mystrdup("(stdin)");
    memory_error(isfp->filename == NULL, return STATUS_FATAL);

    isfp->openflag = 0;
    isfp->randaccessflag = 0;
    offset_string = NULL;
  }
  else {
    for (s=filename; *s && *s != '@'; s++) ;
    isfp->randaccessflag = (*s == '@');
    offset_string = (*s == '@' ? s + 1 : NULL);

    isfp->filename = mystrdup2(filename, s);
    memory_error(isfp->filename == NULL, return STATUS_FATAL);

    status = open_raw_file(get_truename(isfp->filename, NULL),
                           &isfp->input_fd);
    error_test(status != STATUS_OK, E_OPENFAILED, return STATUS_ERROR,
               print_error("%s:  %s\n", filename, sys_errlist[errno]));
    isfp->openflag = 1;

#ifdef ISMAPABLE
    if (!isfp->randaccessflag) {
      caddr_t addr;

      isfp->filesize = get_filesize(get_truename(isfp->filename, NULL));
      isfp->filepos = 0;
      isfp->mapsize = (isfp->filesize < MAXMAPSIZE 
                         ? isfp->filesize : MAXMAPSIZE);
      addr = mmap(0, isfp->mapsize, PROT_READ, MAP_SHARED, isfp->input_fd, 0);
      if (addr != (caddr_t) -1) {
        if (isfp->fp_buffer != NULL)
          free(isfp->fp_buffer);
        isfp->fp_buffer = addr;
        isfp->fp_bufsize = isfp->mapsize;
        isfp->ismapped = 1;
      }
    }
#endif
  }
  isfp->entry_count = 0;

  /*
   * Allocate the space for the internal file buffer and
   * initialize the entry markers, if necessary.
   */
  if (isfp->ismapped) {
    isfp->fp_current = isfp->fp_buffer;
    isfp->fp_top = isfp->fp_buffer + isfp->fp_bufsize;
    isfp->isendtagged = 0;
  }
  else {
    if (isfp->fp_buffer == NULL) {
      isfp->fp_buffer = (char *) malloc(INIT_BUFSIZE);
      memory_error(isfp->fp_buffer == NULL, return STATUS_FATAL);
      isfp->fp_bufsize = INIT_BUFSIZE;
    }
    isfp->fp_current = isfp->fp_top = isfp->fp_buffer;
    isfp->fp_buffer[0] = '\n';
    isfp->isendtagged = 1;
  }
  isfp->fp_bytepos = 0;

  isfp->fp_entrystart = isfp->fp_entryend = isfp->fp_seqstart = NULL;

  /*
   * Set the file format for the sequence file, either by the passed
   * in argument or by looking at the first line of the file beginning
   * with a non-space character.
   */
  if (format != NULL) {
    isfp->format = id;
    isfp->gcg_subformat = subid;
  }
  else {
    isfp->autodetermined = 1;
    status = determine_format(isfp);
    if (status != STATUS_OK && status != STATUS_WARNING)
      return status;
  }

  /*
   * If the filename specifies the entries to read, then scan that
   * list to resolve the entries.
   */
  if (isfp->randaccessflag) {
    if (isfp->format == FORMAT_ASN || isfp->format == FORMAT_CLUSTAL ||
        isfp->format == FORMAT_FOUT || isfp->format == FORMAT_BOUT) {
      raise_error(E_FILEERROR, return STATUS_ERROR,
                  print_error("%s:  Cannot access single entries of %s "
                              "formatted files.\n", filename,
                              file_table[isfp->format].ident));
    }

    status = resolve_offsets(isfp, filename, offset_string);
    if (status != STATUS_OK && status != STATUS_WARNING)
      return status;

    /*
     * Seek to the first specified entry, and reset all of the
     * appropriate fields.
     */
    isfp->currentoffset = 0;
    offset = isfp->byteoffsets[isfp->currentoffset++];
    status = seek_raw_file(isfp->input_fd, offset);
    error_test(status != STATUS_OK, E_READFAILED, return -1,
               print_error("%s:  %s\n", isfp->filename, sys_errlist[errno]));

    isfp->fp_bytepos = offset;
    isfp->entry_count = 0;

    isfp->fp_current = isfp->fp_top = isfp->fp_buffer;
    isfp->fp_buffer[0] = '\n';
    isfp->isendtagged = 1;

    isfp->savech_loc = NULL;
    isfp->savech = '\0';
  }

  isfp->initreadflag = 1;
  return intseqf_read(isfp, 0);
}


/*
 * intseqf_open_for_writing
 *
 * Opens the specified file for writing, and sets the appropriate
 * INTSEQFILE fields.
 *
 * If the filename parameter is the string "-", then standard output
 * is used.
 *
 * Parameters:      isfp      -  an INTSEQFILE structure
 *                  filename  -  the name of the file to open for reading
 *                  format    -  the file format to use
 *                  mode      -  whether to write or append
 *
 * Returns:  a STATUS value
 */
static int intseqf_open_for_writing(INTSEQFILE *isfp, char *filename,
                                    char *format, char *mode)
{
  int id, status, subid;

  /*
   * Figure out the format.
   */
  if (myncasecmp(format, "GCG-", 4) == 0) {
    for (id=0; id < gcg_table_size; id++)
      if (mycasecmp(format + 4, gcg_table[id].ident + 4) == 0)
        break;

    error_test(id == gcg_table_size, E_INVFORMAT, return STATUS_FATAL,
               print_error("Write Error:  `%s' is not a valid file format.\n",
                           format));

    subid = gcg_table[id].format;
    id = (subid == FORMAT_MSF ? FORMAT_MSF : FORMAT_GCG);
  }
  else {
    for (id=0; id < file_table_size; id++)
      if (mycasecmp(format, file_table[id].ident) == 0)
        break;

    error_test(id == file_table_size, E_INVFORMAT, return STATUS_FATAL,
               print_error("Write Error:  `%s' is not a valid file format.\n",
                           format));

    subid = FORMAT_UNKNOWN;
    id = file_table[id].format;
  }

  error_test(file_table[id].putseq_fn == NULL,
             E_INVFORMAT, return STATUS_FATAL,
             print_error("Write Error:  Format `%s' is a read-only format.\n",
                         format));

  /*
   * Open the specified file or standard output.
   */
  if (filename[0] == '-' && filename[1] == '\0') {
    status = open_stdout(&isfp->output_fp);
    error_test(status != STATUS_OK, E_OPENFAILED, return STATUS_ERROR,
               print_error("Write Error:  Standard output is not "
                           "available.\n"));
    filename = "(stdout)";
    isfp->openflag = 0;
  }
  else {
    isfp->output_fp = fopen(get_truename(filename, NULL), mode);
    error_test(isfp->output_fp == NULL, E_OPENFAILED, return STATUS_ERROR,
               print_error("%s:  %s\n", filename, sys_errlist[errno]));
    isfp->openflag = 1;
  }

  isfp->entry_count = 0;
  isfp->format = id;
  isfp->gcg_subformat = subid;

  /*
   * Make a local copy of the filename.
   */
  if (isfp->filename != NULL)
    free(isfp->filename);

  isfp->filename = mystrdup(filename);
  memory_error(isfp->filename == NULL, return STATUS_FATAL);

  return STATUS_OK;
}


/*
 * determine_format
 *
 * Look for the first line in the file which does not begin with a 
 * space, tab or newline character.  Check the beginning of that line
 * against the determinants listed in the file table (where a '?' in
 * a determinant is a wildcard match to any character in the line).
 * If a match occurs, set the file format.  Otherwise, return an error.
 *
 * This procedure will also skip over an e-mail header, if it exists, before
 * determining the file format.
 *
 * Parameters:  isfp  -  an INTSEQFILE structure
 *
 * Returns: a STATUS value
 */
static int determine_format(INTSEQFILE *isfp)
{
  int i, status;
  char *s, *line, *start, *end, *text, *det;

  start = isfp->fp_current;

  /*
   * Skip the e-mail header if it exists.
   */
  status = fp_get_line(isfp, &line, &end);
  if (status == STATUS_OK && strncmp(line, "From ", 5) == 0) {
    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
      if (line == end)
        break;
  }

  /*
   * Skip any blank lines (or lines of whitespace).
   */
  s = NULL;
  while (status == STATUS_OK) {
    for (s=line; s < end && isspace(*s); s++) ;
    if (s < end)
      break;

    status = fp_get_line(isfp, &line, &end);
  }

  /*
   * Test to see if we've hit an error or reached the end of the file.
   */
  switch (status) {
  case STATUS_OK:     break;
  case STATUS_EOF:    raise_error(E_DETFAILED, return STATUS_ERROR,
                                  print_error("%s:  File contains no "
                                              "sequence entries.\n",
                                              isfp->filename));
  case STATUS_ERROR:  return STATUS_ERROR;
  case STATUS_FATAL:  return STATUS_FATAL;
  default:            status_error(return STATUS_ERROR, "determine_format");
  }

  /*
   * Search through the list of determinants to find one which
   * matches the beginning of the non-whitespace characters.
   * If none match, then assume the file contains plaintext.
   */
  text = s;
  for (i=0; i < file_table_size; i++) {
    det = file_table[i].determinant;
    if (det == NULL)
      continue;

    while (*det != '\0') {
      /*
       * Try to match the determinant, where '?' in the determinant
       * matches any character and '|' divides alternative determinants.
       */
      for (s=text; *s && *det && *det != '|'; s++,det++)
        if (*det != '?' && toupper(*s) != toupper(*det))
          break;

      /*
       * If a match did occur, then set the file format accordingly and
       * set the current file pointer to point to the beginning of the line.
       *
       * We have to special case the EMBL/Swiss-Prot determination here
       * because it depends on the structure of the first line.
       */
      if (!*det || *det == '|') {
        isfp->format = file_table[i].format;
        isfp->fp_current = line;
        return STATUS_OK;
      }
      else {
        while (*det && *det != '|')
          det++;
        if (*det)
          det++;
      }
    }
  }

  /*
   * Otherwise, if the line matches none of the determinants, 
   * assume that the file contains only the sequence, and
   * reset the current file pointer to the beginning of the file.
   */
  set_error(E_DETFAILED);
  isfp->format = FORMAT_PLAIN;
  isfp->fp_current = start;

  return STATUS_WARNING;
}


/*
 * resolve_offsets
 *
 * When a filename is specified in "random access" mode, i.e. with one or
 * more specific entries or byte offsets, setup the INTSEQFILE structure
 * to read in random access mode and resolve all of the entries/offsets
 * specified to byte offsets (this may require reading an initial portion
 * of the file in order to convert an identifier or entry number to a 
 * byte offset).
 *
 * Parameters:  isfp      -  An INTSEQFILE structure being opened
 *              filename  -  The complete filename (with entry/offset specs)
 *              offsets   -  The part of `filename' with the entries/offsets
 *
 * Returns:  a STATUS value.
 */
static int resolve_offsets(INTSEQFILE *isfp, char *filename,
                           char *offset_string)
{
  int i, size, offset, count, scan_mode, oldpe, minentry, mincell;
  int status, flag, isacolon, match, len;
  char *s, *t, *s2, *t2, *t3, *idlist, buffer[1024];
  SEQFILE *sfp = (SEQFILE *) isfp;

  size = get_filesize(get_truename(isfp->filename, NULL));

  /*
   * Make an initial count of the number of offsets to find, and malloc
   * the array that will hold the offsets.
   */
  for (s=offset_string,count=1; *s; s++)
    if (*s == ',')
      count++;

  isfp->byteoffsets = (int *) malloc(count * sizeof(int));
  memory_error(isfp->byteoffsets == NULL, return STATUS_FATAL);

  /*
   * Make a more complete parse of the offset string, converting any
   * byte offsets found (of the form "#%d"), marking any identifiers
   * found using a -1 in the offsets array, and semi-converting any
   * entry position numbers found (i.e., if the 5th entry is specified,
   * put  -5 - 1 = -6 in the byte array, so that the negative values
   * from -2 downward specify entry numbers from 1 upward).
   *
   * Also, set the `scan_mode' flag if any entry numbers or identifiers
   * are found.  That will mean that we will have to read the file
   * to convert them into byte offsets.
   */
  count = 0;
  scan_mode = 0;
  for (s=offset_string; *s; ) {
    error_test(*s == ',', E_FILEERROR, return STATUS_ERROR,
               print_error("%s:  Parse error in filename at `%.10s'.\n",
                           filename, s));

    if (*s == '#') {
      for (t=++s; *s && *s != ','; s++) {
        error_test(!isdigit(*s), E_FILEERROR, return STATUS_ERROR,
                   print_error("%s:  Parse error in filename at `%.10s'.\n",
                               filename, t));
      }
      error_test(t == s, E_FILEERROR, return STATUS_ERROR,
                 print_error("%s:  Parse error at end of filename.\n",
                             filename, t));

      offset = myatoi(t, 10, '0');
      if (offset < 0) {
        memcpy(buffer, t-1, s - t + 1);
        buffer[s - t + 1] = '\0';
        raise_error(E_FILEERROR, return STATUS_ERROR,
                    print_error("%s:  Invalid byte offset `%s'.\n", filename,
                                buffer));
      }
      if (offset >= size) {
        memcpy(buffer, t-1, s - t + 1);
        buffer[s - t + 1] = '\0';
        raise_error(E_FILEERROR, return STATUS_ERROR,
                    print_error("%s:  Byte offset `%s' larger than file "
                                "size.\n", filename, buffer));
      }

      isfp->byteoffsets[count++] = offset;
    }
    else if (isdigit(*s)) {
      for (t=s; *s && *s != ','; s++) {
        error_test(!isdigit(*s), E_FILEERROR, return STATUS_ERROR,
                   print_error("%s:  Parse error in filename at `%.10s'.\n",
                               filename, t));
      }

      offset = myatoi(t, 10, '0');
      if (offset <= 0) {
        memcpy(buffer, t-1, s - t + 1);
        buffer[s - t + 1] = '\0';
        raise_error(E_FILEERROR, return STATUS_ERROR,
                    print_error("%s:  Invalid entry number `%s'.\n", filename,
                                buffer));
      }

      isfp->byteoffsets[count++] = - offset - 1;
      if (!scan_mode)
        scan_mode = 1;
    }
    else {
      while (*s && *s != ',') s++;
      isfp->byteoffsets[count++] = -1;
      scan_mode = 2;
    }

    if (*s) s++;
  }
  error_test(count == 0, E_FILEERROR, return STATUS_ERROR,
             print_error("%s:  Invalid entry specifiers:  `%s'.\n",
                         filename, offset_string-1));

  isfp->num_offsets = count;

  if (!scan_mode)
    return STATUS_OK;

  /*
   * Start reading the file, and trying to convert the entry numbers
   * and entry identifiers into byte offsets.
   */
  oldpe = pe_flag;
  pe_flag = PE_NONE;

  /*
   * Use minentry and mincell to always keep track of the smallest
   * entry number that has not been resolved.
   */
  minentry = -1;
  mincell = 0;
  for (i=0,count=0; i < isfp->num_offsets; i++) {
    if (isfp->byteoffsets[i] < 0)
      count++;
    if (isfp->byteoffsets[i] <= -2 &&
        (minentry == -1 || minentry > - (isfp->byteoffsets[i] + 1))) {
      minentry = - (isfp->byteoffsets[i] + 1);
      mincell = i;
    }
  }

  /*
   * The main reading loop.
   */
  status = STATUS_OK;
  while (count > 0) {
    status = intseqf_read(isfp, (scan_mode == 1 ? 0 : 1));
    if (status != STATUS_OK && status != STATUS_WARNING)
      break;
      
    while (count && minentry != -1 && isfp->entry_count == minentry) {
      isfp->byteoffsets[mincell] = seqfbytepos(sfp);
      count--;
      minentry = -1;
      for (i=0; i < isfp->num_offsets; i++) {
        if (isfp->byteoffsets[i] <= -2 &&
            (minentry == -1 || minentry >= - (isfp->byteoffsets[i] + 1))) {
          minentry = - (isfp->byteoffsets[i] + 1);
          mincell = i;
        }
      }
    }

    /*
     * Check all of the identifiers that can be found in the entry to
     * to see if they match any of the identifier that have yet to be
     * resolved.  If an outstanding identifier does not have an
     * identifier prefix, then check it against all of the identifiers
     * in each read-in entry (regardless of their identifier prefix).
     */
    if (scan_mode == 2) {
      idlist = seqfidlist(sfp, 0);
      flag = 0;
      for (i=0,s=offset_string; count > 0 && i < isfp->num_offsets; i++,s++) {
        isacolon = 0;
        for (t=s; *s && *s != ','; s++)
          if (!isacolon && *s == ':')
            isacolon = 1;

        len = s - t;
        if (isfp->byteoffsets[i] == -1) {
          flag++;

          match = 0;
          if (idlist != NULL) {
            for (s2=idlist; *s2; ) {
              for (t2=s2; *s2 && *s2 != '|'; s2++) ;
              if (!isacolon) {
                for (t3=t2; t2 < s2 && *t2 != ':'; t2++) ;
                if (t2 + 1 < s2)
                  t2++;
                else
                  t2 = t3;
              }

              if (len == s2 - t2 && myncasecmp(t2, t, len) == 0) {
                match = 1;
                break;
              }

              if (*s2) s2++;
            }
          }
          
          if (match) {
            isfp->byteoffsets[i] = seqfbytepos(sfp);
            count--;
            flag--;
          }
        }
      }

      if (flag == 0)
        scan_mode = 1;
    }
  }
  pe_flag = oldpe;

  /*
   * Check for unresolved entry numbers or identifiers.
   */
  if (count > 0) {
    if (status == STATUS_ERROR || status == STATUS_FATAL) {
      print_error("%s", seqferrstr);
      return status;
    }

    error_test(minentry > -1, E_FILEERROR, return STATUS_ERROR,
               print_error("%s:  File only contains %d entries, not %d.\n",
                           filename, isfp->entry_count, minentry));

    t = NULL;
    for (i=0,s=offset_string; i < isfp->num_offsets; i++,s++) {
      for (t=s; *s && *s != ','; s++) ;
      if (isfp->byteoffsets[i] < 0)
        break;
    }
    error_test(i == isfp->num_offsets, E_FILEERROR, return STATUS_ERROR,
               print_error("%s:  Unable to resolve entry names/offsets in "
                           "filename.\n", filename));

    memcpy(buffer, t, s - t);
    buffer[s - t] = '\0';
    raise_error(E_FILEERROR, return STATUS_ERROR,
                print_error("%s:  Unable to resolve entry for `%s'.\n",
                            filename, buffer));
  }

  return status;
}


/*
 * intseqf_close
 * 
 * Performs the actual closing of the file.  Used both by the interface
 * procedure seqfclose and the file/database opening procedures when an
 * error occurs (so not all fields of the structure are guaranteed to
 * be allocated).
 *
 * Parameters:  isfp  - an INTSEQFILE structure
 *
 * Returns:  nothing
 */
static void intseqf_close(INTSEQFILE *isfp)
{
  int i;

  if (isfp->optype != OP_WRITE && isfp->openflag) {

#ifdef ISMAPABLE
    if (isfp->ismapped) {
      munmap(isfp->fp_buffer, isfp->mapsize);
      isfp->fp_buffer = NULL;
    }
#endif

    close_raw_file(isfp->input_fd);
    isfp->input_fd = 0;
    isfp->openflag = 0;
  }

  if (isfp->optype == OP_WRITE) {
    if (isfp->opstatus == OP_ACTIVE) {
      if (isfp->format == FORMAT_ASN)
        asn_putseqend(isfp);
      else if (isfp->format == FORMAT_PHYSEQ)
        physeq_putseqend(isfp);
      else if (isfp->format == FORMAT_PHYINT || isfp->format == FORMAT_PHYLIP)
        phyint_putseqend(isfp);
      else if (isfp->format == FORMAT_CLUSTAL)
        clustal_putseqend(isfp);
      else if (isfp->format == FORMAT_MSF)
        msf_putseqend(isfp);
    }

    if (isfp->openflag)
      fclose(isfp->output_fp);
    isfp->output_fp = NULL;
    isfp->openflag = 0;
  }

  if (isfp->filename != NULL) {
    free(isfp->filename);
    isfp->filename = NULL;
  }
  if (isfp->fp_buffer != NULL) {
    free(isfp->fp_buffer);
    isfp->fp_buffer = NULL;
  }
  if (isfp->seq != NULL) {
    free(isfp->seq);
    isfp->seq = NULL;
  }
  if (isfp->info != NULL) {
    free(isfp->info);
    isfp->info = NULL;
  }
  if (isfp->idbuffer != NULL) {
    free(isfp->idbuffer);
    isfp->idbuffer = NULL;
  }
  if (isfp->db_files != NULL) {
    free(isfp->db_files);
    isfp->db_files = NULL;
  }
  if (isfp->db_spec != NULL) {
    free(isfp->db_spec);
    isfp->db_spec = NULL;
  }
  if (isfp->db_name != NULL) {
    free(isfp->db_name);
    isfp->db_name = NULL;
  }
  if (isfp->db_format != NULL) {
    free(isfp->db_format);
    isfp->db_format = NULL;
  }
  if (isfp->db_alpha != NULL) {
    free(isfp->db_alpha);
    isfp->db_alpha = NULL;
  }
  if (isfp->db_idprefix != NULL) {
    free(isfp->db_idprefix);
    isfp->db_idprefix = NULL;
  }
  if (isfp->fout_id1 != NULL) {
    free(isfp->fout_id1);
    isfp->fout_id1 = NULL;
  }
  if (isfp->fout_descr1 != NULL) {
    free(isfp->fout_descr1);
    isfp->fout_descr1 = NULL;
  }
  if (isfp->fout_id2 != NULL) {
    free(isfp->fout_id2);
    isfp->fout_id2 = NULL;
  }
  if (isfp->fout_descr2 != NULL) {
    free(isfp->fout_descr2);
    isfp->fout_descr2 = NULL;
  }
  if (isfp->malign_size > 0) {
    if (isfp->malign_seqs != NULL) {
      for (i=0; i < isfp->malign_count; i++)
        if (isfp->malign_seqs[i] != NULL)
          free(isfp->malign_seqs[i]);
      free(isfp->malign_seqs);
      isfp->malign_seqs = NULL;
    }
    if (isfp->malign_ids != NULL) {
      for (i=0; i < isfp->malign_count; i++)
        if (isfp->malign_ids[i] != NULL)
          free(isfp->malign_ids[i]);
      free(isfp->malign_ids);
      isfp->malign_ids = NULL;
    }
    if (isfp->malign_seqlens != NULL) {
      free(isfp->malign_seqlens);
      isfp->malign_seqlens = NULL;
    }
    isfp->malign_count = isfp->malign_size = 0;
  }
  if (isfp->byteoffsets != NULL) {
    free(isfp->byteoffsets);
    isfp->byteoffsets = NULL;
  }
  isfp->opstatus = OP_FREED;
  free(isfp);
}


/*
 * intseqf_read
 *
 * Read the next entry or sequence in the file.
 *
 * Parameters:  isfp  -  An INTSEQFILE structure open for reading
 *              flag  -  Read the next sequence or entry.
 *
 * Returns:  a status value
 */
static int intseqf_read(INTSEQFILE *isfp, int flag)
{
  /*
   * If a character was replaced with a '\0' to NULL terminate an entry,
   * restore the original character.
   */
  if (isfp->savech_loc != NULL) {
    *isfp->savech_loc = isfp->savech;
    isfp->savech_loc = NULL;
  }

  /*
   * Reset all of the flags marking valid information about the current
   * sequence to false.
   */
  isfp->mapentflag = isfp->isseqcurrent = isfp->rawseqflag = 0;
  isfp->iflag_truelen = isfp->iflag_rawlen = 0;
  if (isfp->istatus != INFO_NONE) {
    isfp->istatus = INFO_NONE;
    isfp->iflag_date = isfp->iflag_idlist = isfp->iflag_description = 0;
    isfp->iflag_comment = isfp->iflag_organism = isfp->iflag_fragment = 0;
    isfp->iflag_circular = isfp->iflag_alphabet = isfp->iflag_fragstart = 0;
    if (isfp->info != NULL) {
      memset(isfp->info, 0, sizeof(SEQINFO));
      isfp->infosize = sizeof(SEQINFO);
    }
  }

  /*
   * Call the "read_fn" function to read the next sequence.
   */
  return (*file_table[isfp->format].read_fn)(isfp, flag);
}


/*
 * intseqf_info
 *
 * The internal procedure to get either one piece of or all of the current
 * entry's information.  This procedure also handles the buffering of
 * that information (i.e., whether it's stored in the SEQFILE's info
 * structure or if a copy is made).
 *
 * Parameters:  isfp       - an open INTSEQFILE structure
 *              newbuffer  - flag telling whether to make a copy of the info
 *              flag       - flag telling what information to retrieve
 *
 * Returns:  a pointer to the information, or NULL on an error
 *                                              (seqferrno set on error)
 *           (Special return values:
 *              1) When computing all of the information from an entry,
 *                 the pointer to the SEQINFO structure is cast and returned.
 *              2) When computing the `iscircular', `isfragment' or
 *                 `alphabet' information, the return value only signals
 *                 whether the information has been stored in the SEQINFO
 *                 structure.)
 */
static char *intseqf_info(INTSEQFILE *isfp, int newbuffer, int flag)
{
  int status, exists, len;
  char *s, *buf;
  SEQINFO *newinfo;

  if ((isfp->istatus == INFO_ALL && flag != SEQINFO_ALLINFO) ||
      (isfp->istatus == INFO_ALLINFO &&
       flag != SEQINFO_ALL && flag != SEQINFO_COMMENT) ||
      (isfp->istatus == INFO_ANY &&
       ((flag == SEQINFO_DATE && isfp->iflag_date) ||
        (flag == SEQINFO_IDLIST && isfp->iflag_idlist) ||
        (flag == SEQINFO_DESCRIPTION && isfp->iflag_description) ||
        (flag == SEQINFO_COMMENT && isfp->iflag_comment) ||
        (flag == SEQINFO_ORGANISM && isfp->iflag_organism) ||
        (flag == SEQINFO_FRAGMENT && isfp->iflag_fragment) ||
        (flag == SEQINFO_CIRCULAR && isfp->iflag_circular) ||
        (flag == SEQINFO_ALPHABET && isfp->iflag_alphabet) ||
        (flag == SEQINFO_STARTPOS && isfp->iflag_fragstart) ||
        (flag == SEQINFO_TRUELEN && isfp->iflag_truelen) ||
        (flag == SEQINFO_RAWLEN && isfp->iflag_rawlen))))
    exists = 1;
  else
    exists = 0;

  if (!exists) {
    if (isfp->info == NULL) {
      isfp->info = (SEQINFO *) malloc(sizeof(SEQINFO));
      memory_error(isfp->info == NULL, return NULL);
      isfp->infosize = isfp->infobufsize = sizeof(SEQINFO);
      memset(isfp->info, 0, isfp->infosize);
    }

    if (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO) {
      memset(isfp->info, 0, sizeof(SEQINFO));
      isfp->infosize = sizeof(SEQINFO);
      isfp->istatus = INFO_NONE;
      isfp->iflag_date = isfp->iflag_idlist = isfp->iflag_description = 0;
      isfp->iflag_comment = isfp->iflag_organism = isfp->iflag_fragment = 0;
      isfp->iflag_circular = isfp->iflag_alphabet = isfp->iflag_fragstart = 0;
    }
    else {
      if (isfp->istatus == INFO_ALLINFO) {
        isfp->info->comment = NULL;
        isfp->iflag_comment = 0;
        isfp->istatus = INFO_ANY;
      }

      if (flag == SEQINFO_IDLIST) {
        isfp->info->idlist = NULL;
        isfp->iflag_idlist = 0;
        isfp->istatus = INFO_ANY;
      }
    }

    /*
     * Call the format specific getinfo procedure to get the information.
     */
    len = isfp->fp_entryend - isfp->fp_entrystart;
    status = (*file_table[isfp->format].getinfo_fn)(isfp, isfp->fp_entrystart,
                                                    len, flag);
    switch (status) {
    case STATUS_OK:
    case STATUS_WARNING:
      break;

    case STATUS_ERROR:
    case STATUS_FATAL:
      return NULL;

    default:
      status_error(return NULL, "intseqf_info");
    }
  }

  if (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO) {
    isfp->istatus = (flag == SEQINFO_ALL ? INFO_ALL : INFO_ALLINFO);
    if (!newbuffer)
      return (char *) isfp->info;

    s = (char *) isfp->info;
    len = isfp->infosize;
    buf = (char *) malloc(len);
    memory_error(buf == NULL, return NULL);
    memcpy(buf, s, len);
    
    newinfo = (SEQINFO *) buf;
    if (isfp->info->dbname)
      newinfo->dbname = buf + (isfp->info->dbname - s);
    if (isfp->info->filename)
      newinfo->filename = buf + (isfp->info->filename - s);
    if (isfp->info->format)
      newinfo->format = buf + (isfp->info->format - s);
    if (isfp->info->date)
      newinfo->date = buf + (isfp->info->date - s);
    if (isfp->info->idlist)
      newinfo->idlist = buf + (isfp->info->idlist - s);
    if (isfp->info->description)
      newinfo->description = buf + (isfp->info->description - s);
    if (isfp->info->comment)
      newinfo->comment = buf + (isfp->info->comment - s);
    if (isfp->info->organism)
      newinfo->organism = buf + (isfp->info->organism - s);
    if (isfp->info->history)
      newinfo->history = buf + (isfp->info->history - s);

    return buf;
  }
  else {
    s = NULL;
    isfp->istatus = INFO_ANY;
    switch (flag) {
    case SEQINFO_DATE:  isfp->iflag_date = 1;  
                        s = isfp->info->date;
                        break;
    case SEQINFO_IDLIST:  isfp->iflag_idlist = 1;
                          s = isfp->info->idlist;
                          break;
    case SEQINFO_DESCRIPTION:  isfp->iflag_description = 1;
                               s = isfp->info->description;
                               break;
    case SEQINFO_COMMENT: isfp->iflag_comment = 1;
                          s = isfp->info->comment;
                          break;
    case SEQINFO_ORGANISM:  isfp->iflag_organism = 1;
                            s = isfp->info->organism;
                            break;
    case SEQINFO_FRAGMENT:  isfp->iflag_fragment = 1;
                            return "";
    case SEQINFO_CIRCULAR:  isfp->iflag_circular = 1;
                            return "";
    case SEQINFO_ALPHABET:  isfp->iflag_alphabet = 1;
                            return "";
    case SEQINFO_STARTPOS:  isfp->iflag_fragstart = 1;
                            return "";
    case SEQINFO_TRUELEN:
    case SEQINFO_RAWLEN:    return "";
    default:
      program_error(1, return NULL, 
                    print_error("   Illegal flag value %d in intseqf_info\n",
                                flag));
    }
    if (s == NULL || !newbuffer)
      return s;

    len = strlen(s) + 1;
    buf = (char *) malloc(len);
    memory_error(buf == NULL, return NULL);
    memcpy(buf, s, len);

    return buf;
  }
}




/*
 *
 *
 * Section for error reporting procedures, both interface procedures
 * and internal procedures.
 *
 *
 *
 */


/*
 * seqfperror
 * 
 * Prints the error string for the last error that occurred.
 * Similar to the `perror' function.
 *
 * Parameters:  s  - a character string (could be NULL)
 *
 * Returns:  nothing
 */
void seqfperror(char *s)
{
  if (!ctype_initflag)
    init_ctype();

  if (s != NULL) {
    (*perror_fn)(s);
    (*perror_fn)(":  ");
  }

  if (seqferrno != E_NOERROR)
    (*perror_fn)(seqferrstr);
}


/*
 * seqferrpolicy
 * 
 * Sets the error policy for the SEQIO package.  When the package
 * hits either an error or a warning condition, it does up to three
 * things:
 *    1) set seqferrno to the appropriate value
 *    2) print a descriptive error message to stderr
 *    3) if an error, either call exit or return an error value
 *
 * The different error policies allow the user to specify whether
 * a descriptive message is printed and whether `exit' is called
 * on an error.
 * 
 * Parameters:  pe  - an integer describing the new error policy
 *
 * Returns:  the old error policy
 */
int seqferrpolicy(int pe)
{
  int oldpe;

  if (!ctype_initflag)
    init_ctype();

  param_error(pe < PE_NONE || pe > PE_ALL, return -1, "seqferrpolicy",
              "arg 1 is an invalid error policy");

  oldpe = pe_flag;
  pe_flag = pe;

  return oldpe;
}


/*
 * seqfsetperror
 * 
 * Sets the function called to output all of the error messages.
 *
 * Parameters:  perror_fn  - a pointer to an error function (or NULL
 *                           to use the default printing)
 *
 * Returns:  nothing
 */
void seqfsetperror(void (*perr_fn)(char *))
{
  if (!ctype_initflag)
    init_ctype();

  param_error(perr_fn == NULL, return, "seqfsetperror", "arg 1 is NULL");

  if (perr_fn == NULL)
    perror_fn = puterror;
  else
    perror_fn = perr_fn;
}


/*
 * print_fatal
 * 
 * An internal procedure which does the printing and exiting on
 * an error.  Its arguments are similar to that of printf.
 *
 * Parameters:  format, ...   -  similar to printf
 *
 * Returns:  nothing
 */
static void print_fatal(char *format, ...)
{
  char *s;
  va_list ap;

  for (s=seqferrstr; err_batchmode && *s; s++) ;
  va_start(ap, format);
  vsprintf(s, format, ap);
  va_end(ap);

  if (pe_flag == PE_ALL || pe_flag == PE_NOEXIT ||
      pe_flag == PE_NOWARN || pe_flag == PE_ERRONLY) {
    (*perror_fn)(seqferrstr);
  }
  if (pe_flag == PE_ALL || pe_flag == PE_NOWARN)
    exit(1);
}


/*
 * print_error
 * 
 * An internal procedure which does the printing on an error.
 * Its arguments are similar to that of printf.
 *
 * Parameters:  format, ...   -  similar to printf
 *
 * Returns:  nothing
 */
static void print_error(char *format, ...)
{
  char *s;
  va_list ap;

  for (s=seqferrstr; err_batchmode && *s; s++) ;
  va_start(ap, format);
  vsprintf(s, format, ap);
  va_end(ap);

  if (pe_flag == PE_ALL || pe_flag == PE_NOEXIT ||
      pe_flag == PE_NOWARN || pe_flag == PE_ERRONLY) {
    (*perror_fn)(seqferrstr);
  }
}


/*
 * print_warning
 * 
 * An internal procedure which does the printing of a descriptive
 * message about a warning.  Its arguments are similar to that of printf.
 *
 * Parameters:  format, ...   -  similar to printf
 *
 * Returns:  nothing
 */
static void print_warning(char *format, ...)
{
  char *s;
  va_list ap;

  for (s=seqferrstr; err_batchmode && *s; s++) ;
  va_start(ap, format);
  vsprintf(s, format, ap);
  va_end(ap);

  if (pe_flag == PE_ALL || pe_flag == PE_NOEXIT || pe_flag == PE_WARNONLY) {
    (*perror_fn)(seqferrstr);
  }
}



/*
 *
 *  Section for procedures that read the various file formats.
 *
 *
 */


/*
 * raw_read
 *
 * The input file consists of a single entry (i.e., the characters of
 * the file are the characters of the entry).  This read function is
 * used for the "Raw", "Plain", "GCG" and "MSF" file formats.
 * 
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *
 * Return: a STATUS value
 */
static int raw_read(INTSEQFILE *isfp, int flag)
{
  int status, gcglen, msfflag;
  char *s, *t, *end, *line;

  if (isfp->entry_count != 0)
    return STATUS_EOF;

  isfp->entry_count++;
  isfp->entry_seqlen = isfp->entry_truelen = isfp->entry_rawlen = 0;
  isfp->iflag_truelen = isfp->iflag_rawlen = 0;
  isfp->entry_seqno = isfp->entry_numseqs = 1;

  isfp->fp_entrystart = isfp->fp_seqstart = isfp->fp_current;
  isfp->fp_entryend = NULL;

  status = fp_read_all(isfp);
  switch (status) {
  case STATUS_OK:    break;
  case STATUS_EOF:   raise_error(E_PARSEERROR, return STATUS_ERROR,
                            print_error("%s:  Empty file.\n", isfp->filename));
  case STATUS_ERROR: return STATUS_ERROR;
  case STATUS_FATAL: return STATUS_FATAL;
  default:           status_error(return STATUS_ERROR, "raw_read");
  }

  isfp->fp_entryend = isfp->fp_top;

  if (isfp->format == FORMAT_RAW) {
    isfp->entry_truelen = isfp->entry_rawlen = isfp->fp_entryend -
                                               isfp->fp_entrystart;
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;

    return STATUS_OK;
  }

  /*
   * Check to see if the file is a GCG or MSF file by looking for a line
   * ending with "..".
   */
  for (s=isfp->fp_entrystart; s < isfp->fp_entryend; s++) {
    if (*s == '.' && s[1] == '.') {
      for (s+=2; s < isfp->fp_entryend && *s != '\n'; s++)
        if (!isspace(*s))
          break;
      if (s < isfp->fp_entryend && *s == '\n')
        break;
    }
  }

  if (s >= isfp->fp_entryend ||
      (!isfp->autodetermined && isfp->format == FORMAT_PLAIN)) {
    if (isfp->format == FORMAT_PLAIN) {
      if (!isfp->autodetermined)
        return STATUS_OK;
      else {
        set_error(E_DETFAILED);
        print_warning("%s:  Cannot determine file format.  Using `plain'.\n",
                      isfp->filename);
        return STATUS_WARNING;
      }
    }

    raise_error(E_PARSEERROR, return STATUS_ERROR,
                print_error("%s, entry 1:  Parse error in GCG entry:  "
                            "no `..' dividing line.\n", isfp->filename));
  }

  /*
   * If it is a GCG or MSF file, determine which one by looking for the
   * information line preceeding the `..' and looking for "MSF: ".
   */
  gcglen = msfflag = 0;
  for (t=s-3; t >= isfp->fp_entrystart && *t != '\n'; t--) {
    if (gcglen == 0 && mystreq(t, 'M', "MSF: ")) {
      gcglen = myatoi(t + 4, 10, '0');
      msfflag = (toupper(*t) == 'M');
    }
    else if (gcglen == 0 && mystreq(t, 'L', "LENGTH: ")) {
      gcglen = myatoi(t + 8, 10, '0');
      msfflag = (toupper(*t) == 'M');
    }
  }
  isfp->gcg_infoline = t+1;

  if (!msfflag || isfp->format == FORMAT_GCG) {
    isfp->format = FORMAT_GCG;
    if (isfp->autodetermined)
      isfp->gcg_subformat = FORMAT_UNKNOWN;
    isfp->fp_seqstart = s + 1;
    if (gcglen > 0) {
      isfp->entry_rawlen = isfp->entry_seqlen = gcglen;
      isfp->iflag_rawlen = 1;
    }

    return STATUS_OK;
  }

  /*
   * Find the number of sequences in the MSF entry and the length
   * of the first sequence.
   */
  isfp->format = FORMAT_MSF;
  isfp->fp_seqstart = NULL;
  isfp->entry_numseqs = 0;
  isfp->malign_seqno = 1;

  s++;
  end = isfp->fp_entryend;
  while (s < end) {
    for (line=s; s < end && *s != '\n' && isspace(*s); s++) ;
    if (s >= end)
      break;

    if (*s != '\n') {
      if (*s == '/' && s[1] == '/')
        break;

      error_test(!mystreq(s, 'N', "NAME: "), E_PARSEERROR, return STATUS_ERROR,
                 print_error("%s, entry 1:  Parse error in MSF sequence "
                             "header lines.\n", isfp->filename));

      isfp->entry_numseqs++;
      if (isfp->fp_seqstart == NULL) {
        isfp->fp_seqstart = line;

        /*
         * Find the length of the first sequence.
         */
        for ( ; s + 5 < end && *s != '\n'; s++) {
          if (mystreq(s, 'L', "LEN: ")) {
            isfp->entry_rawlen = isfp->entry_seqlen = myatoi(s + 4, 10, '0');
            isfp->iflag_rawlen = 1;
            break;
          }
        }
        error_test(isfp->entry_seqlen <= 0, E_PARSEERROR, return STATUS_ERROR,
                   print_error("%s, entry 1:  Invalid format of MSF sequence "
                               "header lines.\n", isfp->filename));
      }

      while (s < end && *s != '\n') s++;
      if (s >= end)
        break;
    }
    s++;
  }
  error_test(s >= end, E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry 1:  Parse error in MSF entry:  "
                         "no `//' dividing line.\n", isfp->filename));
  error_test(isfp->entry_numseqs == 0, E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry 1:  Parse error in MSF entry:  "
                         "no sequence headers between `..' and `//'.\n",
                         isfp->filename));
  return STATUS_OK;
}


/*
 * raw_getseq  (Raw file-format)
 *
 * In the raw file format, the "entry" contains only the sequence.
 *
 * Parameters:  isfp        -  an INTSEQFILE structure that has read an
 *                             entry from a sequence file.
 *              rawseqflag  -  should the actual sequence (when 0) or
 *                             the raw sequence text be extracted (when 1),
 *                             or should just the lengths be set (when 2).
 *
 * Return: a STATUS value
 */
static int raw_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  int len;

  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));

  if (rawseqflag == GETSEQ_LENGTHS)
    return STATUS_OK;

  if (isfp->fp_seqstart == NULL) {
    isfp->seqlen = 0;
    isfp->seq[0] = '\0';
    set_error(E_NOSEQ);
    print_error("%s, entry %d:  Entry contains no sequence.\n",
                isfp->filename, isfp->entry_count);
    return STATUS_ERROR;
  }

  len = isfp->fp_entryend - isfp->fp_seqstart;
  if (len + 1 >= isfp->seqsize) {
    isfp->seqsize += len + 1;
    isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
    if (isfp->seq == NULL) {
      isfp->seqsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
  }

  memcpy(isfp->seq, isfp->fp_seqstart, len);
  isfp->seq[len] = '\0';
  isfp->seqlen = len;

  return STATUS_OK;
}


/*
 * databank_read
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *
 * Return: a STATUS value
 */
static int databank_read(INTSEQFILE *isfp, int flag)
{
  int status, format, period, count, sqflag;
  char *s, *t, *line, *end, *keyword;

  isfp->fp_entrystart = isfp->fp_seqstart = isfp->fp_entryend = NULL;
  isfp->entry_seqlen = isfp->entry_truelen = isfp->entry_rawlen = 0;
  isfp->iflag_truelen = isfp->iflag_rawlen = 0;
  isfp->entry_count++;
  isfp->entry_seqno = isfp->entry_numseqs = 1;
  isfp->gcg_infoline = NULL;

  format = (isfp->format == FORMAT_GCG ? isfp->gcg_subformat : isfp->format);

  /*
   * Scan to the first line.
   */
  switch (format) {
  case FORMAT_GENBANK:  keyword = "LOCUS";  break;
  case FORMAT_PIR:      keyword = "ENTRY";  break;
  case FORMAT_EMBL:     keyword = "ID   ";  break;
  case FORMAT_SPROT:    keyword = "ID   ";  break;
  default:
    status_error(return STATUS_ERROR, "databank_read");
  }

  while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
    if (mystreq(line, *keyword, keyword))
      break;

  if (status == STATUS_EOF) {
    error_test(isfp->entry_count == 1,
               E_PARSEERROR, return STATUS_ERROR,
               print_error("%s:  File contains no %s entries.\n",
                           isfp->filename, file_table[format].ident));
    return STATUS_EOF;
  }

  /*
   * Read the rest of the entry, if no error has occurred.  While reading
   * the entry, look for a line containing the sequence's length, as well
   * as the first line containing the sequence.
   */
  if (status == STATUS_OK) {
    isfp->fp_entrystart = line;

    switch (format) {
    case FORMAT_GENBANK:
      isfp->entry_truelen = isfp->entry_seqlen = myatoi(line + 22, 10, '0');
      isfp->iflag_truelen = 1;

      while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
        if (line[0] == '/' && line[1] == '/')
          break;
        else if (mystreq(line, 'O', "ORIGIN"))
          isfp->fp_seqstart = end + 1;
        else if (end - line >= 2 && end[-1] == '.' && end[-2] == '.') {
          if (isfp->fp_seqstart && !isfp->gcg_infoline &&
              (isfp->format == FORMAT_GCG || isfp->autodetermined)) {
            error_test(isfp->entry_count != 1,
                       E_PARSEERROR, return STATUS_ERROR,
                       print_error("%s, entry %d:  GCG entry found, but not "
                                   "as only entry in file.\n", isfp->filename,
                                   isfp->entry_count));

            if (isfp->format != FORMAT_GCG) {
              isfp->gcg_subformat = isfp->format;
              isfp->format = FORMAT_GCG;
            }
            isfp->gcg_infoline = line;
            isfp->fp_seqstart = end + 1;
          }
        }
      }
      break;

    case FORMAT_PIR:
      while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
        if (line[0] == '/' && line[1] == '/' && line[2] == '/')
          break;
        else if (mystreq(line, 'S', "SEQUENCE"))
          isfp->fp_seqstart = end + 1;
        else if (mystreq(line, 'S', "SUMMARY")) {
          for (t=line+7; t < end && !mystreq(t, '#', "#LENGTH"); t++) ;
          isfp->entry_truelen = isfp->entry_seqlen = myatoi(t + 7, 10, '0');
          isfp->iflag_truelen = 1;
        }
        else if (end - line >= 2 && end[-1] == '.' && end[-2] == '.') {
          if (isfp->fp_seqstart && !isfp->gcg_infoline &&
              (isfp->format == FORMAT_GCG || isfp->autodetermined)) {
            error_test(isfp->entry_count != 1,
                       E_PARSEERROR, return STATUS_ERROR,
                       print_error("%s, entry %d:  GCG entry found, but not "
                                   "as only entry in file.\n", isfp->filename,
                                   isfp->entry_count));

            if (isfp->format != FORMAT_GCG) {
              isfp->gcg_subformat = isfp->format;
              isfp->format = FORMAT_GCG;
            }
            isfp->gcg_infoline = line;
            isfp->fp_seqstart = end + 1;
          }
        }
      }
      break;

    case FORMAT_EMBL:
    case FORMAT_SPROT:
      /*
       * If this is the first entry read in, and the format was
       * specified as EMBL, check that entry to see if it is in fact
       * a Swiss-Prot entry.
       */
      if (isfp->entry_count == 1 && isfp->format == FORMAT_EMBL) {
        period = count = 0;
        t = NULL;
        for (s=line+5; s < end; s++) {
          if (*s == ';') {
            count++;
            t = s;
          }
          else if (*s == '.')
            period = 1;
        }

        if (count == 2 && period && mystreq(t-3, 'P', "PRT;"))
          isfp->format = FORMAT_SPROT;
      }

      sqflag = 0;
      while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
        if (line[0] == '/' && line[1] == '/')
          break;
        else if (isfp->fp_seqstart == NULL && mystreq(line, ' ', "     "))
          isfp->fp_seqstart = line;
        else if (mystreq(line, 'S', "SQ")) {
          if (end - line > 13) {
            isfp->entry_truelen = 
              isfp->entry_seqlen = myatoi(line+13, 10, '0');
            isfp->iflag_truelen = 1;
          }
          sqflag = 1;
        }
        else if (end - line >= 2 && end[-1] == '.' && end[-2] == '.') {
          if ((isfp->fp_seqstart || sqflag) && !isfp->gcg_infoline &&
              (isfp->format == FORMAT_GCG || isfp->autodetermined)) {
            error_test(isfp->entry_count != 1,
                       E_PARSEERROR, return STATUS_ERROR,
                       print_error("%s, entry %d:  GCG entry found, but not "
                                   "as only entry in file.\n", isfp->filename,
                                   isfp->entry_count));

            if (isfp->format != FORMAT_GCG) {
              isfp->gcg_subformat = isfp->format;
              isfp->format = FORMAT_GCG;
            }
            isfp->gcg_infoline = line;
            isfp->fp_seqstart = end + 1;
          }
        }
      }
      break;

    default:
      status_error(return STATUS_ERROR, "databank_read");
    }
  }

  /*
   * Check for errors during the read.
   */
  switch (status) {
  case STATUS_OK:    break;
  case STATUS_EOF:   error_test(isfp->format != FORMAT_GCG,
                                E_PARSEERROR, return STATUS_ERROR,
                                print_error("%s, entry %d:  Premature EOF "
                                            "reached.\n", isfp->filename,
                                            isfp->entry_count));
                     break;
  case STATUS_ERROR: return STATUS_ERROR;
  case STATUS_FATAL: return STATUS_FATAL;
  default:           status_error(return STATUS_ERROR, "databank_read");
  }

  /*
   * Set the final values for the entry.
   */
  if (isfp->format == FORMAT_GCG) {
    error_test(isfp->gcg_infoline == NULL, E_PARSEERROR, return STATUS_ERROR,
               print_error("%s, entry %d:  No `..' dividing line in GCG "
                           "entry.\n", isfp->filename, isfp->entry_count));

    isfp->fp_entryend = isfp->fp_top;
    for (s=isfp->gcg_infoline; *s != '\n'; s++) {
      if (mystreq(s, 'L', "LENGTH: ")) {
        isfp->entry_rawlen = myatoi(s + 8, 10, '0');
        isfp->iflag_rawlen = 1;
        if (isfp->entry_seqlen <= 0)
          isfp->entry_seqlen = isfp->entry_rawlen;
        break;
      }
    }
  }
  else {
    isfp->fp_entryend = end + 1;
    if (isfp->fp_seqstart == NULL)
      isfp->entry_seqno = isfp->entry_numseqs = 0;
  }

  return STATUS_OK;
}


/*
 * basic_read
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *
 * Return: a STATUS value
 */
static int basic_read(INTSEQFILE *isfp, int flag)
{
  int status, format, header_offset, descr_line, descr_end;
  char keych, *s, *line, *end;

  isfp->fp_entrystart = isfp->fp_seqstart = isfp->fp_entryend = NULL;
  isfp->entry_seqlen = isfp->entry_truelen = isfp->entry_rawlen = 0;
  isfp->iflag_truelen = isfp->iflag_rawlen = 0;
  isfp->entry_count++;
  isfp->entry_seqno = isfp->entry_numseqs = 1;
  isfp->gcg_infoline = NULL;

  format = (isfp->format == FORMAT_GCG ? isfp->gcg_subformat : isfp->format);
  descr_line = descr_end = -1;

  if (format == FORMAT_NBRF || format == FORMAT_NBRFOLD)
    isfp->nbrf_header = NULL;

  /*
   * Scan to the first line.
   */
  switch (format) {
  case FORMAT_FASTA:
  case FORMAT_FASTAOLD:
  case FORMAT_NBRF:
  case FORMAT_NBRFOLD:
    keych = '>';
    break;

  case FORMAT_STANFORD:
  case FORMAT_STANFORDOLD:
    keych = ';';
    break;

  default:
    status_error(return STATUS_ERROR, "basic_read");
  }

  while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
    if (line[0] == keych)
      break;

  if (status == STATUS_EOF) {
    error_test(isfp->entry_count == 1, E_PARSEERROR, return STATUS_ERROR,
               print_error("%s:  File contains no %s entries.\n",
                           isfp->filename, file_table[format].ident));
    return STATUS_EOF;
  }

  /*
   * Read the rest of the header, if no error has occurred.
   */
  if (status == STATUS_OK) {
    isfp->fp_entrystart = line;

    if (format == FORMAT_FASTA) {
      descr_line = line - isfp->fp_entrystart;
      descr_end = end - isfp->fp_entrystart;
    }

    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
      if (line[0] != keych)
        break;
  }

  /*
   * Read any extra lines to the beginning of the sequence.
   */
  if (status == STATUS_OK) {
    switch (format) {
    case FORMAT_FASTA:
    case FORMAT_FASTAOLD:
      if (line[0] == ';') {
        while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
          if (line[0] != ';')
            break;
      }
      break;

    case FORMAT_NBRF:
    case FORMAT_NBRFOLD:
    case FORMAT_STANFORD:
    case FORMAT_STANFORDOLD:
      descr_line = line - isfp->fp_entrystart;
      descr_end = end - isfp->fp_entrystart;
      status = fp_get_line(isfp, &line, &end);
      break;
    }
  }

  /*
   * If the file ends before any sequence lines are given, return
   * with the entry consisting of just a header.
   */
  if (status == STATUS_EOF) {
    isfp->entry_seqno = isfp->entry_numseqs = 0;
    isfp->fp_entryend = isfp->fp_top;
    return STATUS_OK;
  }

  /*
   * Read the sequence lines up to the next line beginning with 'keych'.
   */
  if (status == STATUS_OK && line[0] != keych) {
    isfp->fp_seqstart = line;

    switch (format) {
    case FORMAT_FASTA:
    case FORMAT_FASTAOLD:
    case FORMAT_STANFORD:
    case FORMAT_STANFORDOLD:
      while (status == STATUS_OK && line[0] != keych) {
        if (end - line >= 2 && end[-1] == '.' && end[-2] == '.') {
          if (!isfp->gcg_infoline &&
              (isfp->format == FORMAT_GCG || isfp->autodetermined)) {
            error_test(isfp->entry_count != 1,
                       E_PARSEERROR, return STATUS_ERROR,
                       print_error("%s, entry %d:  GCG entry found, but not "
                                   "as only entry in file.\n", isfp->filename,
                                   isfp->entry_count));

            if (isfp->format != FORMAT_GCG) {
              isfp->gcg_subformat = isfp->format;
              isfp->format = FORMAT_GCG;
            }
            isfp->gcg_infoline = line;
            isfp->fp_seqstart = end + 1;
          }
        }
        status = fp_get_line(isfp, &line, &end);
      }
      break;

    case FORMAT_NBRF:
    case FORMAT_NBRFOLD:
      header_offset = 0;
      while (status == STATUS_OK && line[0] != keych) {
        if (!header_offset && line[1] == ';')
          header_offset = line - isfp->fp_entrystart;
        else if (end - line >= 2 && end[-1] == '.' && end[-2] == '.') {
          if (!isfp->gcg_infoline &&
              (isfp->format == FORMAT_GCG || isfp->autodetermined)) {
            error_test(isfp->entry_count != 1,
                       E_PARSEERROR, return STATUS_ERROR,
                       print_error("%s, entry %d:  GCG entry found, but not "
                                   "as only entry in file.\n", isfp->filename,
                                   isfp->entry_count));

            if (isfp->format != FORMAT_GCG) {
              isfp->gcg_subformat = isfp->format;
              isfp->format = FORMAT_GCG;
            }
            isfp->gcg_infoline = line;
            isfp->fp_seqstart = end + 1;
          }
        }
        status = fp_get_line(isfp, &line, &end);
      }
      if (header_offset)
        isfp->nbrf_header = isfp->fp_entrystart + header_offset;
      break;
    }
  }

  /*
   * Check for errors during the read.
   */
  switch (status) {
  case STATUS_OK:    isfp->fp_entryend = isfp->fp_current = line;
                     break;
  case STATUS_EOF:   isfp->fp_entryend = isfp->fp_current = isfp->fp_top;
                     break;
  case STATUS_ERROR: return STATUS_ERROR;
  case STATUS_FATAL: return STATUS_FATAL;
  default:           status_error(return STATUS_ERROR, "basic_read");
  }

  if (isfp->format == FORMAT_GCG) {
    error_test(status == STATUS_OK, E_PARSEERROR, return STATUS_ERROR,
               print_error("%s, entry %d:  Improperly formatted sequence "
                           "lines of GCG entry.\n", isfp->filename,
                           isfp->entry_count));
    error_test(isfp->gcg_infoline == NULL, E_PARSEERROR, return STATUS_ERROR,
               print_error("%s, entry %d:  No `..' dividing line in GCG "
                           "entry.\n", isfp->filename, isfp->entry_count));
  }

  /*
   * Find the sequence length.
   */
  if (isfp->gcg_infoline) {
    end = isfp->fp_entryend;
    for (s=isfp->gcg_infoline; s < end && *s != '\n'; s++) {
      if (mystreq(s, 'L', "LENGTH: ")) {
        isfp->entry_rawlen = myatoi(s + 8, 10, '0');
        isfp->iflag_rawlen = 1;
        break;
      }
    }
  }
  if (descr_line != -1) {
    line = isfp->fp_entrystart + descr_line;
    end = isfp->fp_entrystart + descr_end;

    flag = 0;
    for (s=end-1; s >= line && isspace(*s); s--) ;
    if (s >= line && *s == '.')
      for (s--; s >= line && isspace(*s); s--) ;
    if (s >= line && *s == ')') {
      for (s--; s >= line && *s != '('; s--) ;
      for (s--; s >= line && isspace(*s); s--) ;
    }
    if (s - line >= 3 && isspace(*(s-2)) &&
        ((toupper(*(s-1)) == 'B' && toupper(*s) == 'P') ||
         (toupper(*(s-1)) == 'A' && toupper(*s) == 'A') ||
         (toupper(*(s-1)) == 'C' && toupper(*s) == 'H'))) {
      flag++;
      for (s-=3; s >= line && isspace(*s); s--) ;
    }
    if (s >= line && isdigit(*s)) {
      flag++;
      while (s >= line && isdigit(*s)) s--;
      while (s >= line && isspace(*s)) s--;
    }
    if (s >= line && flag == 2 && *s == ',') {
      isfp->entry_truelen = myatoi(s + 1, 10, '0');
      isfp->iflag_truelen = 1;
    }
  }

  return STATUS_OK;
}


/*
 * basic_getseq
 *
 * The basic method for getting the sequence from an entry.  The pointer
 * "fp_seqstart" points to the first line of the sequence, and all of
 * the alphabetic characters from there to "fp_entryend" make up the
 * sequence.  The procedure copies those characters into the sequence
 * buffer.
 *
 * There are a couple minor variations included in the function.  For
 * GenBank, EMBL, PIR and Swissprot, the last line of the entry is ignored
 * (since it contains the "//" or "///" terminator).  For FASTA, the
 * function ignores any text after a semi-colon on a line.  For NBRF, the
 * function stops either at the first asterisk, the first "header" line or
 * the end of the entry.
 *
 * Parameters:  isfp        -  an INTSEQFILE structure that has read an
 *                             entry from a sequence file.
 *              rawseqflag  -  should the actual sequence or the raw
 *                             sequence text be extracted.
 *
 * Returns:  a STATUS value.
 */
static int basic_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  int format;
  char *s, *end, *seq;

  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));

  if (isfp->fp_seqstart == NULL) {
    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_rawlen = isfp->entry_truelen = 0;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;
      return STATUS_OK;
    }
    else {
      isfp->seqlen = 0;
      isfp->seq[0] = '\0';
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
  }

  s = isfp->fp_seqstart;
  end = isfp->fp_entryend;
  format = isfp->format;

  /*
   * For GenBank, PIR, EMBL and Swissprot, move the end in front of the
   * "//" or "///" line.
   *
   * For NBRF, move the end to the first "header" line that appears after
   * the sequence.
   */
  if (format == FORMAT_GENBANK || format == FORMAT_PIR ||
      format == FORMAT_EMBL || format == FORMAT_SPROT) {
    if (*(end-1) == '\n') end--;
    while (end > s && *(end-1) != '\n') end--;
  }
  else if (format == FORMAT_NBRF) {
    if (isfp->nbrf_header != NULL)
      end = isfp->nbrf_header;
  }

  /*
   * Reallocate the sequence buffer, if necessary.
   */
  if (rawseqflag != GETSEQ_LENGTHS && end - s + 1 >= isfp->seqsize) {
    isfp->seqsize += end - s + 1;
    isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
    if (isfp->seq == NULL) {
      isfp->seqsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
  }
  seq = isfp->seq;

  /*
   * Extract the sequence characters.
   */
  switch (format) {
  case FORMAT_FASTA:
    switch (rawseqflag) {
    case GETSEQ_SEQUENCE:
      for ( ; s < end; s++) {
        if (*s == ';')
          while (s < end && *s != '\n') s++;
        else if (isalpha(*s))
          *seq++ = *s;
      }
      break;

    case GETSEQ_RAWSEQ:
      for ( ; s < end && *s != '*'; s++) {
        if (*s == ';')
          while (s < end && *s != '\n') s++;
        else if (!(isspace(*s) || isdigit(*s)))
          *seq++ = *s;
      }
      break;

    case GETSEQ_LENGTHS:
      isfp->entry_truelen = isfp->entry_rawlen = 0;
      for ( ; s < end && *s != '*'; s++) {
        if (*s == ';')
          while (s < end && *s != '\n') s++;
        else if (!(isspace(*s) || isdigit(*s))) {
          isfp->entry_rawlen++;
          if (isalpha(*s))
            isfp->entry_truelen++;
        }
      }
      isfp->iflag_truelen = isfp->iflag_rawlen = 1;
      break;
    }
    break;

  case FORMAT_NBRF:
    switch (rawseqflag) {
    case GETSEQ_SEQUENCE:
      for ( ; s < end && *s != '*'; s++)
        if (isalpha(*s))
          *seq++ = *s;
      break;

    case GETSEQ_RAWSEQ:
      for ( ; s < end && *s != '*'; s++)
        if (!(isspace(*s) || isdigit(*s)))
          *seq++ = *s;
      break;

    case GETSEQ_LENGTHS:
      isfp->entry_truelen = isfp->entry_rawlen = 0;
      for ( ; s < end && *s != '*'; s++) {
        if (!(isspace(*s) || isdigit(*s))) {
          isfp->entry_rawlen++;
          if (isalpha(*s))
            isfp->entry_truelen++;
        }
      }
      isfp->iflag_truelen = isfp->iflag_rawlen = 1;
      break;
    }
    break;

  default:
    switch (rawseqflag) {
    case GETSEQ_SEQUENCE:
      for ( ; s < end; s++)
        if (isalpha(*s))
          *seq++ = *s;
      break;

    case GETSEQ_RAWSEQ:
      for ( ; s < end; s++)
        if (!(isspace(*s) || isdigit(*s)))
          *seq++ = *s;
      break;

    case GETSEQ_LENGTHS:
      isfp->entry_truelen = isfp->entry_rawlen = 0;
      for ( ; s < end; s++) {
        if (!(isspace(*s) || isdigit(*s))) {
          isfp->entry_rawlen++;
          if (isalpha(*s))
            isfp->entry_truelen++;
        }
      }
      break;
    }
  }

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;
  else {
    *seq = '\0';
    isfp->seqlen = seq - isfp->seq;
    
    /*
     * Perform checks on the sequence length.
     */
    if (isfp->seqlen == 0) {
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
    if (rawseqflag == GETSEQ_SEQUENCE &&
        isfp->entry_seqlen > 0 && isfp->entry_seqlen != isfp->seqlen) {
      set_error(E_DIFFLENGTH);
      print_warning("Warning:  %s, entry %d:  Entry gives seq. length of %d, "
                    "but %d characters found.\n", isfp->filename,
                    isfp->entry_count, isfp->entry_seqlen, isfp->seqlen);
      return STATUS_WARNING;
    }

    if (rawseqflag == GETSEQ_SEQUENCE) {
      isfp->entry_truelen = isfp->seqlen;
      isfp->iflag_truelen = 1;
    }
    else if (rawseqflag == GETSEQ_RAWSEQ) {
      isfp->entry_rawlen = isfp->seqlen;
      isfp->iflag_rawlen = 1;
    }
  }

  return STATUS_OK;
}


/*
 * databank_fast_read
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *
 * Return: a STATUS value
 */
static int databank_fast_read(INTSEQFILE *isfp, int flag)
{
  static int jt_flag = 0;
  static int origin_jump_table[128], summary_jump_table[128];
  static int sequenc_jump_table[128];
  register char *s;
  int i, count, status, shift, format, width, thin, wide, num;
  char *t, *line, *end, *stemp, *top;

  if (!jt_flag) {
    for (i=0; i < 128; i++)
      origin_jump_table[i] = 7;
    origin_jump_table['\n'] = 6;
    origin_jump_table['O'] = 5;
    origin_jump_table['R'] = 4;
    origin_jump_table['G'] = 2;
    origin_jump_table['I'] = 1;
    origin_jump_table['N'] = 0;

    for (i=0; i < 128; i++)
      summary_jump_table[i] = 8;
    summary_jump_table['\n'] = 7;
    summary_jump_table['S'] = 6;
    summary_jump_table['U'] = 5;
    summary_jump_table['M'] = 3;
    summary_jump_table['A'] = 2;
    summary_jump_table['R'] = 1;
    summary_jump_table['Y'] = 0;

    for (i=0; i < 128; i++)
      sequenc_jump_table[i] = 13;
    sequenc_jump_table['\n'] = 12;
    sequenc_jump_table[' '] = 7;
    sequenc_jump_table['S'] = sequenc_jump_table['s'] = 6;
    sequenc_jump_table['Q'] = sequenc_jump_table['q'] = 4;
    sequenc_jump_table['U'] = sequenc_jump_table['u'] = 3;
    sequenc_jump_table['E'] = sequenc_jump_table['e'] = 2;
    sequenc_jump_table['N'] = sequenc_jump_table['n'] = 1;
    sequenc_jump_table['C'] = sequenc_jump_table['c'] = 0;

    jt_flag = 1;
  }

  isfp->fp_entrystart = isfp->fp_seqstart = isfp->fp_entryend = NULL;
  isfp->entry_seqlen = isfp->entry_truelen = isfp->entry_rawlen = 0;
  isfp->iflag_truelen = isfp->iflag_rawlen = 0;
  isfp->entry_count++;
  isfp->entry_seqno = isfp->entry_numseqs = 1;

  format = isfp->format;

  /*
   * Simple line-by-line scanning to look for the LOCUS line.
   */
  switch (format) {
  case FORMAT_GBFAST:
    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
      if (line[0] == 'L' && line[1] == 'O' && line[2] == 'C' &&
          line[3] == 'U' && line[4] == 'S')
        break;
    }
    break;

  case FORMAT_PIRFAST:
    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
      if (line[0] == 'E' && line[1] == 'N' && line[2] == 'T' &&
          line[3] == 'R' && line[4] == 'Y')
        break;
    }
    break;

  case FORMAT_EMBLFAST:
  case FORMAT_SPFAST:
    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
      if (line[0] == 'I' && line[1] == 'D' && line[2] == ' ' &&
          line[3] == ' ' && line[4] == ' ')
        break;
    }
    break;

  default:
    status_error(return STATUS_ERROR, "databank_fast_read");
  }

  if (status == STATUS_EOF) {
    error_test(isfp->entry_count == 1, E_PARSEERROR, return STATUS_ERROR,
               print_error("%s:  File contains no %s entries.\n",
                           isfp->filename, file_table[format].ident));
    return STATUS_EOF;
  }

  if (status == STATUS_OK) {
    isfp->fp_entrystart = line;

    if (format == FORMAT_GBFAST) {
      isfp->entry_seqlen = myatoi(line + 22, 10, '0');
      isfp->entry_rawlen = isfp->entry_truelen = isfp->entry_seqlen;
      isfp->iflag_truelen = isfp->iflag_rawlen = 1;
    }

    /*
     * Perform a simplified Boyer-Moore search for the "ORIGIN", "SUMMARY"
     * or "SQ   Sequenc" line.
     *
     * During the Boyer-Moore search, "s" and "top" will be local copies of
     * the fp_current and fp_top values (normally hidden inside fp_get_line).
     */
    line = s = isfp->fp_current;
    top = isfp->fp_top;

    width = 0;
    switch (format) {
    case FORMAT_GBFAST:
      s += 5;
      while (status == STATUS_OK) {
        while (s < top && (shift = origin_jump_table[(int) *s]))
          s += shift;

        if (s < top) {
          if (s[-6] == '\n' && s[-5] == 'O' && s[-4] == 'R' &&
              s[-3] == 'I' && s[-2] == 'G' && s[-1] == 'I')
            break;
          else
            s += 7;
        }
        else {
          stemp = s;
          status = fp_read_more(isfp, &line, &stemp, &top);
          s = stemp;
        }
      }
      width = 5;
      break;

    case FORMAT_PIRFAST:
      s += 6;
      while (status == STATUS_OK) {
        while (s < top && (shift = summary_jump_table[(int) *s]))
          s += shift;

        if (s < top) {
          if (s[-7] == '\n' && s[-6] == 'S' && s[-5] == 'U' &&
              s[-4] == 'M' && s[-3] == 'M' && s[-2] == 'A' && s[-1] == 'R')
            break;
          else
            s += 8;
        }
        else {
          stemp = s;
          status = fp_read_more(isfp, &line, &stemp, &top);
          s = stemp;
        }
      }
      width = 6;
      break;

    case FORMAT_EMBLFAST:
    case FORMAT_SPFAST:
      s += 11;
      while (status == STATUS_OK) {
        while (s < top && (shift = sequenc_jump_table[(int) *s]))
          s += shift;

        if (s < top) {
          if (s[-12] == '\n' && s[-11] == 'S' && s[-10] == 'Q' &&
              s[-9] == ' ' && s[-8] == ' ' && s[-7] == ' ' && s[-6] == 'S' &&
              toupper(s[-5]) == 'E' && toupper(s[-4]) == 'Q' &&
              toupper(s[-3]) == 'U' && toupper(s[-2]) == 'E' &&
              toupper(s[-1]) == 'N')
            break;
          else
            s += 13;
        }
        else {
          stemp = s;
          status = fp_read_more(isfp, &line, &stemp, &top);
          s = stemp;
        }
      }
      width = 11;
      break;
    }

    if (status == STATUS_OK) {
      isfp->fp_current = s - width;
      isfp->fp_top = top;
    }
  }

  if (status == STATUS_OK && 
      (status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
    if (format == FORMAT_EMBLFAST || format == FORMAT_SPFAST) {
      isfp->entry_seqlen = myatoi(line + 14, 10, '0');
      isfp->entry_truelen = isfp->entry_rawlen = isfp->entry_seqlen;
      isfp->iflag_truelen = isfp->iflag_rawlen = 1;
    }
    else if (format == FORMAT_PIRFAST) {
      if (mystreq(line + 17, '#', "#LENGTH")) {
        isfp->entry_truelen = isfp->entry_seqlen = myatoi(line + 25, 10, '0');
        isfp->iflag_truelen = 1;
      }
      else {
        for (t=line + 7; t < end; t++) {
          if (*t == '#' && mystreq(t+1, 'L', "LENGTH")) {
            isfp->entry_truelen = isfp->entry_seqlen = myatoi(t + 8, 10, '0');
            isfp->iflag_truelen = 1;
            break;
          }
        }
      }

      if ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
        status = fp_get_line(isfp, &line, &end);
    }
  }

  if (status == STATUS_OK) {
    /*
     * The entry length must have been in the header for this code to work.
     */
    error_test(isfp->entry_seqlen == 0, E_PARSEERROR, return STATUS_ERROR,
               print_error("%s, entry %d:  Sequence length is missing from ",
                           "entry.\n", isfp->filename, isfp->entry_count));
    /*
     * Skip past the complete sequence lines (i.e., the lines containing
     * a full 60 characters of the sequence, and then restore the values
     * of fp_current and fp_top.
     */
    line = s = isfp->fp_current;
    top = isfp->fp_top;
    
    isfp->fp_seqstart = line;

    switch (format) {
    case FORMAT_GBFAST:    thin = 76;  wide = 76;  num = 60;  break;
    case FORMAT_PIRFAST:   thin = 68;  wide = 69;  num = 30;  break;
    case FORMAT_EMBLFAST:  thin = 71;  wide = 81;  num = 60;  break;
    case FORMAT_SPFAST:    thin = 71;  wide = 81;  num = 60;  break;
    default:
      status_error(return STATUS_ERROR, "databank_fast_read");
    }

    count = num;
    while (count < isfp->entry_seqlen) {
      if (s + wide < top || (s + thin < top && s[thin-1] == '\n')) {
        count += num;
        s += (s[thin-1] == '\n' ? thin : wide);
      }
      else {
        stemp = s;
        status = fp_read_more(isfp, &line, &stemp, &top);
        s = stemp;
        if (status != STATUS_OK)
          break;
      }
    }

    if (status == STATUS_OK) {
      isfp->fp_current = s;
      isfp->fp_top = top;

      error_test(*(s-1) != '\n', E_PARSEERROR, return STATUS_ERROR,
                 print_error("%s, entry %d:  Improper format of entry's "
                             "sequence lines.\n", isfp->filename,
                             isfp->entry_count));
    }
  }

  /*
   * Finally, look for the "//" line to end the sequence.  It must be
   * either on the current line or the next line (all previous lines were
   * skipped above).
   */
  if (status == STATUS_OK) {
    status = fp_get_line(isfp, &line, &end);
    if (status == STATUS_OK && (line[0] != '/' || line[1] != '/'))
      status = fp_get_line(isfp, &line, &end);

    error_test(status == STATUS_OK && (line[0] != '/' || line[1] != '/'),
               E_PARSEERROR, return STATUS_ERROR,
               print_error("%s, entry %d:  Improper format of entry's sequence"
                           " lines.\n", isfp->filename, isfp->entry_count));
  }

  switch (status) {
  case STATUS_OK:    break;
  case STATUS_EOF:   raise_error(E_PARSEERROR, return STATUS_ERROR,
                       print_error("%s, entry %d:  Premature EOF reached.\n",
                                   isfp->filename, isfp->entry_count));
  case STATUS_ERROR: return STATUS_ERROR;
  case STATUS_FATAL: return STATUS_FATAL;
  default:           status_error(return STATUS_ERROR, "databank_fast_read");
  }

  isfp->fp_entryend = end + 1;

  return STATUS_OK;
}


/*
 * databank_fast_getseq
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *
 * Return: a STATUS value
 */
static int databank_fast_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  int count, format;
  register char *s, *seq;

  program_error(isfp->fp_seqstart == NULL, return STATUS_ERROR,
                print_error("  fp_seqstart not set by %s's read function\n",
                            file_table[isfp->format].ident));
  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));
  program_error(isfp->entry_seqlen <= 0, return STATUS_ERROR,
                print_error("   Entry %d's sequence length not set by %s's "
                            "read function\n", isfp->entry_count,
                            file_table[isfp->format].ident));

  if (rawseqflag == GETSEQ_LENGTHS)
    return STATUS_OK;

  /*
   * Reallocate the sequence buffer, if necessary.
   */
  if (isfp->entry_seqlen >= isfp->seqsize) {
    isfp->seqsize += isfp->entry_seqlen;
    isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
    if (isfp->seq == NULL) {
      isfp->seqsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
  }

  /*
   * Scan the "full" sequence lines, i.e., lines containing 60 characters
   * of the sequence.  Such lines contain 76 characters overall (including
   * the newline).
   */
  s = isfp->fp_seqstart;
  seq = isfp->seq;
  format = isfp->format;

  switch (format) {
  case FORMAT_GBFAST:
    for (count=60; count <= isfp->entry_seqlen; count+=60) {
      seq[0] = s[10];  seq[1] = s[11];  seq[2] = s[12];  seq[3] = s[13];
      seq[4] = s[14];  seq[5] = s[15];  seq[6] = s[16];  seq[7] = s[17];
      seq[8] = s[18];  seq[9] = s[19];  seq[10] = s[21]; seq[11] = s[22];
      seq[12] = s[23]; seq[13] = s[24]; seq[14] = s[25]; seq[15] = s[26];
      seq[16] = s[27]; seq[17] = s[28]; seq[18] = s[29]; seq[19] = s[30];

      seq[20] = s[32]; seq[21] = s[33]; seq[22] = s[34]; seq[23] = s[35];
      seq[24] = s[36]; seq[25] = s[37]; seq[26] = s[38]; seq[27] = s[39];
      seq[28] = s[40]; seq[29] = s[41]; seq[30] = s[43]; seq[31] = s[44];
      seq[32] = s[45]; seq[33] = s[46]; seq[34] = s[47]; seq[35] = s[48];
      seq[36] = s[49]; seq[37] = s[50]; seq[38] = s[51]; seq[39] = s[52];

      seq[40] = s[54]; seq[41] = s[55]; seq[42] = s[56]; seq[43] = s[57];
      seq[44] = s[58]; seq[45] = s[59]; seq[46] = s[60]; seq[47] = s[61];
      seq[48] = s[62]; seq[49] = s[63]; seq[50] = s[65]; seq[51] = s[66];
      seq[52] = s[67]; seq[53] = s[68]; seq[54] = s[69]; seq[55] = s[70];
      seq[56] = s[71]; seq[57] = s[72]; seq[58] = s[73]; seq[59] = s[74];

      seq += 60;
      s += 76;
    }
    break;

  case FORMAT_PIRFAST:
    for (count=30; count <= isfp->entry_seqlen; count+=30) {
      seq[0] = s[8];   seq[1] = s[10];  seq[2] = s[12];  seq[3] = s[14];
      seq[4] = s[16];  seq[5] = s[18];  seq[6] = s[20];  seq[7] = s[22];
      seq[8] = s[24];  seq[9] = s[26];  seq[10] = s[28]; seq[11] = s[30];
      seq[12] = s[32]; seq[13] = s[34]; seq[14] = s[36]; seq[15] = s[38];
      seq[16] = s[40]; seq[17] = s[42]; seq[18] = s[44]; seq[19] = s[46];

      seq[20] = s[48]; seq[21] = s[50]; seq[22] = s[52]; seq[23] = s[54];
      seq[24] = s[56]; seq[25] = s[58]; seq[26] = s[60]; seq[27] = s[62];
      seq[28] = s[64]; seq[29] = s[66];

      seq += 30;
      s += (s[67] == '\n' ? 68 : 69);
    }
    break;

  case FORMAT_EMBLFAST:
  case FORMAT_SPFAST:
    for (count=60; count <= isfp->entry_seqlen; count+=60) {
      seq[0] = s[5];   seq[1] = s[6];   seq[2] = s[7];   seq[3] = s[8];
      seq[4] = s[9];   seq[5] = s[10];  seq[6] = s[11];  seq[7] = s[12];
      seq[8] = s[13];  seq[9] = s[14];  seq[10] = s[16]; seq[11] = s[17];
      seq[12] = s[18]; seq[13] = s[19]; seq[14] = s[20]; seq[15] = s[21];
      seq[16] = s[22]; seq[17] = s[23]; seq[18] = s[24]; seq[19] = s[25];

      seq[20] = s[27]; seq[21] = s[28]; seq[22] = s[29]; seq[23] = s[30];
      seq[24] = s[31]; seq[25] = s[32]; seq[26] = s[33]; seq[27] = s[34];
      seq[28] = s[35]; seq[29] = s[36]; seq[30] = s[38]; seq[31] = s[39];
      seq[32] = s[40]; seq[33] = s[41]; seq[34] = s[42]; seq[35] = s[43];
      seq[36] = s[44]; seq[37] = s[45]; seq[38] = s[46]; seq[39] = s[47];

      seq[40] = s[49]; seq[41] = s[50]; seq[42] = s[51]; seq[43] = s[52];
      seq[44] = s[53]; seq[45] = s[54]; seq[46] = s[55]; seq[47] = s[56];
      seq[48] = s[57]; seq[49] = s[58]; seq[50] = s[60]; seq[51] = s[61];
      seq[52] = s[62]; seq[53] = s[63]; seq[54] = s[64]; seq[55] = s[65];
      seq[56] = s[66]; seq[57] = s[67]; seq[58] = s[68]; seq[59] = s[69];

      seq += 60;
      s += (s[70] == '\n' ? 71 : 81);
    }
    break;
  }


  /*
   * Scan the last line of the sequence, if it exists.
   */
  if (*s != '/') {
    while (*s != '\n') {
      if (isalpha(*s))
        *seq++ = *s;
      s++;
    }
  }

  *seq = '\0';
  isfp->seqlen = seq - isfp->seq;

  /*
   * Check the length of the sequence.
   */
  if (isfp->seqlen == 0) {
    set_error(E_NOSEQ);
    print_error("%s, entry %d:  Entry contains no sequence.\n",
                isfp->filename, isfp->entry_count);
    return STATUS_ERROR;
  }
  if (isfp->entry_seqlen > 0 && isfp->entry_seqlen != isfp->seqlen) {
    set_error(E_DIFFLENGTH);
    print_warning("Warning: %s, entry %d:  Entry gives seq. length of %d, "
                  "but %d characters found.\n", isfp->filename,
                  isfp->entry_count, isfp->entry_seqlen, isfp->seqlen);
    return STATUS_WARNING;
  }
    
  return STATUS_OK;
}


/*
 * gcg_getseq
 *
 * Get the sequence for a GCG entry.  This is very similar to the 
 * basic_getseq function, except that all periods read from the
 * sequence are automatically replaced with dashes.
 *
 * Parameters:  isfp        -  an INTSEQFILE structure that has read an
 *                             entry from a sequence file.
 *              rawseqflag  -  should the actual sequence or the raw
 *                             sequence text be extracted.
 *
 * Returns:  a STATUS value.
 */
static int gcg_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  int format;
  char ch, *s, *end, *seq;

  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));

  if (isfp->fp_seqstart == NULL) {
    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_rawlen = isfp->entry_truelen = 0;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;
      return STATUS_OK;
    }
    else {
      isfp->seqlen = 0;
      isfp->seq[0] = '\0';
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
  }

  s = isfp->fp_seqstart;
  end = isfp->fp_entryend;
  format = isfp->format;

  /*
   * Reallocate the sequence buffer, if necessary.
   */
  if (rawseqflag != GETSEQ_LENGTHS && end - s + 1 >= isfp->seqsize) {
    isfp->seqsize += end - s + 1;
    isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
    if (isfp->seq == NULL) {
      isfp->seqsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
  }
  seq = isfp->seq;

  /*
   * Extract the sequence characters.
   */
  switch (rawseqflag) {
  case GETSEQ_SEQUENCE:
    for ( ; s < end; s++) {
      if (*s == '>' || *s == '<' || *s == '$') {
        for (ch=*s++; s < end && *s != ch; s++) ;
      }
      else if (isalpha(*s))
        *seq++ = *s;
    }
    break;

  case GETSEQ_RAWSEQ:
    for ( ; s < end; s++) {
      if (*s == '>' || *s == '<' || *s == '$') {
        for (ch=*s++; s < end && *s != ch; s++) ;
      }
      else if (!isspace(*s) && !isdigit(*s))
        *seq++ = (*s == '.' ? '-' : *s);
    }
    break;

  case GETSEQ_LENGTHS:
    isfp->entry_truelen = isfp->entry_rawlen = 0;
    for ( ; s < end; s++) {
      if (*s == '>' || *seq == '<' || *s == '$') {
        for (ch=*s++; s < end && *s != ch; s++) ;
      }
      else if (!isspace(*s) && !isdigit(*s)) {
        isfp->entry_rawlen++;
        if (isalpha(*s))
          isfp->entry_truelen++;
      }
    }
    break;
  }

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;
  else {
    *seq = '\0';
    isfp->seqlen = seq - isfp->seq;

    /*
     * Perform checks on the sequence length.
     */
    if (isfp->seqlen == 0) {
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
    if (rawseqflag == GETSEQ_RAWSEQ &&
        isfp->entry_seqlen > 0 && isfp->entry_seqlen != isfp->seqlen) {
      set_error(E_DIFFLENGTH);
      print_warning("Warning:  %s, entry %d:  Entry gives seq. length of %d, "
                    "but %d characters found.\n", isfp->filename,
                    isfp->entry_count, isfp->entry_seqlen, isfp->seqlen);
      return STATUS_WARNING;
    }

    if (rawseqflag == GETSEQ_SEQUENCE) {
      isfp->entry_truelen = isfp->seqlen;
      isfp->iflag_truelen = 1;
    }
    else if (rawseqflag == GETSEQ_RAWSEQ) {
      isfp->entry_rawlen = isfp->seqlen;
      isfp->iflag_rawlen = 1;
    }
  }

  return STATUS_OK;
}


/*
 * msf_read
 *
 * Reads GCG's MSF format.
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *
 * Return: a STATUS value
 */
static int msf_read(INTSEQFILE *isfp, int flag)
{
  char *s, *t;

  if (isfp->entry_count == 0)
    return raw_read(isfp, flag);

  if (!flag && isfp->entry_seqno < isfp->entry_numseqs) {
    isfp->entry_seqno++;

    /*
     * Advance the `fp_seqstart' pointer so that it's pointing to the
     * header line for the now current sequence, and then get the sequence
     * length.
     */
    while (isfp->malign_seqno < isfp->entry_seqno) {
      for (s=isfp->fp_seqstart; *s != '\n'; s++) ;
      isfp->fp_seqstart = s + 1;
      for (t=s+1; *t != '\n' && isspace(*t); t++) ;
      if (*t != '\n')
        isfp->malign_seqno++;
    }

    isfp->entry_seqlen = 0;
    for (s=isfp->fp_seqstart; *s != '\n'; s++) {
      if (mystreq(s, 'L', "LEN: ")) {
        isfp->entry_rawlen = isfp->entry_seqlen = myatoi(s + 4, 10, '0');
        isfp->iflag_rawlen = 1;
        break;
      }
    }
    error_test(isfp->entry_seqlen == 0, E_PARSEERROR, return STATUS_ERROR,
               print_error("%s, entry 1:  Invalid format of MSF sequence "
                           "header lines.\n", isfp->filename));
               
    return STATUS_OK;
  }

  return STATUS_EOF;
}


static int msf_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  int i, namelen;
  char *s, *t, *end, *seq, *name;

  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));

  if (isfp->fp_seqstart == NULL) {
    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_rawlen = isfp->entry_truelen = 0;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;
      return STATUS_OK;
    }
    else {
      isfp->seqlen = 0;
      isfp->seq[0] = '\0';
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
  }

  while (isfp->malign_seqno < isfp->entry_seqno) {
    for (s=isfp->fp_seqstart; *s != '\n'; s++) ;
    for (t=s; *t == '\n'; )
      for (s=t++; *t != '\n' && isspace(*t); t++) ;
    isfp->fp_seqstart = s + 1;
    isfp->malign_seqno++;
  }

  /*
   * Reallocate the sequence buffer, if necessary.
   */
  if (rawseqflag != GETSEQ_LENGTHS && isfp->entry_seqlen >= isfp->seqsize) {
    isfp->seqsize += isfp->entry_seqlen;
    isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
    if (isfp->seq == NULL) {
      isfp->seqsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
  }
  seq = isfp->seq;

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->entry_truelen = isfp->entry_rawlen = 0;

  /*
   * Extract the sequence characters.  Every line that begins with
   * the name of the sequence contains sequence characters.
   */
  s = isfp->fp_seqstart;
  end = isfp->fp_entryend;

  for (s=isfp->fp_seqstart; isspace(*s); s++) ;
  for (s+=5; isspace(*s); s++) ;
  for (name=s; !isspace(*s); s++) ;
  namelen = s - name + 1;

  /*
   * Skip to the beginning of the sequence lines.
   */
  while (s < end && (*s != '\n' || s[1] != '/' || s[2] != '/')) s++;
  for (s++; s < end && *s != '\n'; s++) ;
  s++;

  /*
   * Extract the characters.
   */
  while (s < end) {
    while (s < end && *s != '\n' && isspace(*s)) s++;
    if (s >= end || *s == '\n') {
      s++;
      continue;
    }

    for (i=0; s < end && i < namelen && name[i] == *s; i++,s++) ;
    if (i < namelen)
      while (s < end && *s != '\n') s++;
    else {
      switch (rawseqflag) {
      case GETSEQ_SEQUENCE:
        for ( ; s < end && *s != '\n'; s++)
          if (isalpha(*s))
            *seq++ = *s;
        break;

      case GETSEQ_RAWSEQ:
        for ( ; s < end && *s != '\n'; s++)
          if (!(isspace(*s) || isdigit(*s)))
            *seq++ = (*s == '.' ? '-' : *s);
        break;

      case GETSEQ_LENGTHS:
        for ( ; s < end && *s != '\n'; s++) {
          if (!(isspace(*s) || isdigit(*s))) {
            isfp->entry_rawlen++;
            if (isalpha(*s))
              isfp->entry_truelen++;
          }
        }
        break;
      }
    }
    s++;
  }

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;
  else {
    *seq = '\0';
    isfp->seqlen = seq - isfp->seq;

    /*
     * Perform a check on the sequence length.
     */
    if (isfp->seqlen == 0) {
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
    if (rawseqflag == GETSEQ_RAWSEQ &&
        isfp->entry_seqlen > 0 && isfp->entry_seqlen != isfp->seqlen) {
      set_error(E_DIFFLENGTH);
      print_warning("Warning:  %s, entry %d:  Entry gives seq. length of %d, "
                    "but %d characters found.\n", isfp->filename,
                    isfp->entry_count, isfp->entry_seqlen, isfp->seqlen);
      return STATUS_WARNING;
    }

    if (rawseqflag == GETSEQ_SEQUENCE) {
      isfp->entry_truelen = isfp->seqlen;
      isfp->iflag_truelen = 1;
    }
    else if (rawseqflag == GETSEQ_RAWSEQ) {
      isfp->entry_rawlen = isfp->seqlen;
      isfp->iflag_rawlen = 1;
    }
  }

  return STATUS_OK;
}

  
/*
 * fastaout_read     (FASTA output formats)
 *
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *
 * Return: a STATUS value
 */
#define FASTA_MODE 1
#define LFASTA_MODE 2
#define ALIGN_MODE 3

#define MARKX0 1
#define MARKX2 2
#define MARKX3 3
#define MARKX10 10
#define NONPARSABLE -1

static int fastaout_read(INTSEQFILE *isfp, int flag)
{
  int state, runflag, markx, mode, count, status, fasta_out_bug_flag;
  char *s, *t, *s2, *line, *end;

  if (isfp->entry_count == 0) {
    state = 0;
    runflag = 1;
    while (runflag && (status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
      switch (state) {
      case -1:
        if (line == end)
          state = 0;
        break;

      case 0:
        if (strncmp(line, "From ", 5) == 0)
          state = -1;
        else if (mystreq(line, ' ', " FASTA") ||
                 mystreq(line, ' ', " FASTX") ||
                 mystreq(line, ' ', " TFASTA") ||
                 mystreq(line, ' ', " SSEARCH")) {
          isfp->fout_mode = FASTA_MODE;
          for (s=isfp->fout_progname,t=line+1; !isspace(*t); s++,t++)
            *s = *t;
          *s = '\0';
          state = 1;
        }
        else if (mystreq(line, ' ', " LFASTA") ||
                 mystreq(line, ' ', " LALIGN")) {
          isfp->fout_mode = LFASTA_MODE;
          for (s=isfp->fout_progname,t=line+1; !isspace(*t); s++,t++)
            *s = *t;
          *s = '\0';
          state = 4;
        }
        else if (mystreq(line, 'A', "ALIGN")) {
          isfp->fout_mode = ALIGN_MODE;
          for (s=isfp->fout_progname,t=line; !isspace(*t); s++,t++)
            *s = *t;
          *s = '\0';
          state = 8;
        }
        else
          state = 11;
        break;

      case 11:
        if (strncmp(line, "The best scores are:", 20) == 0) {
          isfp->fout_mode = FASTA_MODE;
          strcpy(isfp->fout_progname, "FASTA/FASTX/TFASTA/SSEARCH");
          state = 2;
        }
        else if (strncmp(line, " Comparison of:", 15) == 0) {
          isfp->fout_mode = LFASTA_MODE;
          strcpy(isfp->fout_progname, "LFASTA/LALIGN");
          state = 5;
        }
        else if (line[0] == '>' && line[1] == '>' && line[2] == '>') {
          isfp->fout_mode = LFASTA_MODE;
          strcpy(isfp->fout_progname, "LFASTA/LALIGN");
          runflag = 0;
        }
        break;

      case 1:
        if (line[0] == ' ' && line[1] == '>') {
          error_test(isfp->fout_id1 != NULL,
                     E_PARSEERROR, return STATUS_ERROR,
                     print_error("%s, entry %d:  Text differs from %s "
                                 " output format.\n", isfp->filename,
                                 isfp->entry_count, isfp->fout_progname));

          for (t=s=line+2; s < end && !isspace(*s); s++) ;
          if (s < end) {
            isfp->fout_id1 = mystrdup2(t, s);
            for (s++; s < end && isspace(*s); s++);
          }

          if (s < end) {
            for (t=s,s=end-1; s >= t && *s != ':'; s--) ;
            if (t < s) {
              for (s2=s; s2 > t && isspace(*(s2-1)); s2--) ;
              isfp->fout_descr1 = mystrdup2(t, s2);

              s += 2;
              if (s < end && isdigit(*s)) {
                isfp->fout_len1 = *s - '0';
                for (s++; s < end && isdigit(*s); s++) {
                  isfp->fout_len1 *= 10;
                  isfp->fout_len1 += *s - '0';
                }

                s++;
                if (*s == 'a' && s[1] == 'a')
                  isfp->fout_alpha1 = PROTEIN;
                else if (*s == 'n' && s[1] == 't')
                  isfp->fout_alpha1 = DNA;
              }
            }
          }
        }
        else if (strncmp(line, "The best scores are:", 20) == 0)
          state = 2;
        break;

      case 2:
        if (line == end || isspace(line[0]))
          state = 3;
        break;

      case 3:
        runflag = 0;
        break;

      case 4:
        if (strncmp(line, " Comparison of:", 15) == 0)
          state = 5;
        else if (line[0] == '>' && line[1] == '>' && line[2] == '>')
          runflag = 0;
        break;

      case 5:
        error_test(isfp->fout_id1 != NULL, E_PARSEERROR, return STATUS_ERROR,
                   print_error("%s, entry %d:  Text differs from %s "
                               " output format.\n", isfp->filename,
                               isfp->entry_count, isfp->fout_progname));
        for (s=line+2; s < end && *s != '>'; s++) ;
        for (t=++s; s < end && !isspace(*s); s++) ;
        if (s < end) {
          isfp->fout_id1 = mystrdup2(t, s);
          for (s++; s < end && isspace(*s); s++);
        }
        if (s < end) {
          for (t=s,s=end-1; s >= t && *s != '-'; s--) ;
          if (t < s) {
            for (s2=s; s2 > t && isspace(*(s2-1)); s2--) ;
            isfp->fout_descr1 = mystrdup2(t, s2);
            s += 2;
            if (s < end && isdigit(*s)) {
              isfp->fout_len1 = *s - '0';
              for (s++; s < end && isdigit(*s); s++) {
                isfp->fout_len1 *= 10;
                isfp->fout_len1 += *s - '0';
              }
              s++;
              if (*s == 'a' && s[1] == 'a')
                isfp->fout_alpha1 = PROTEIN;
              else if (*s == 'n' && s[1] == 't')
                isfp->fout_alpha1 = DNA;
            }
          }
        }
        state = 6;
        break;

      case 6:
        error_test(isfp->fout_id2 != NULL, E_PARSEERROR, return STATUS_ERROR,
                   print_error("%s, entry %d:  Text differs from %s "
                               " output format.\n", isfp->filename,
                               isfp->entry_count, isfp->fout_progname));
        for (s=line+2; s < end && *s != '>'; s++) ;
        for (t=++s; s < end && !isspace(*s); s++) ;
        if (s < end) {
          isfp->fout_id2 = mystrdup2(t, s);
          for (s++; s < end && isspace(*s); s++);
        }
        if (s < end) {
          for (t=s,s=end-1; s >= t && *s != '-'; s--) ;
          if (t < s) {
            for (s2=s; s2 > t && isspace(*(s2-1)); s2--) ;
            isfp->fout_descr2 = mystrdup2(t, s2);
            s += 2;
            if (s < end && isdigit(*s)) {
              isfp->fout_len2 = *s - '0';
              for (s++; s < end && isdigit(*s); s++) {
                isfp->fout_len2 *= 10;
                isfp->fout_len2 += *s - '0';
              }
              s++;
              if (*s == 'a' && s[1] == 'a')
                isfp->fout_alpha2 = PROTEIN;
              else if (*s == 'n' && s[1] == 't')
                isfp->fout_alpha2 = DNA;
            }
          }
        }
        state = 7;
        break;

      case 7:
        state = 2;
        break;

      case 8:
        if (line[0] == '>') {
          error_test(isfp->fout_id1 != NULL || end - line <= 54, 
                     E_PARSEERROR, return STATUS_ERROR,
                     print_error("%s, entry %d:  Text differs from %s "
                                 " output format.\n", isfp->filename,
                                 isfp->entry_count, isfp->fout_progname));
          for (t=s=line+1; s < end && !isspace(*s); s++) ;
          if (s < end) {
            isfp->fout_id1 = mystrdup2(t, s);
            for (s++; s < end && isspace(*s); s++);
          }
          if (s < end) {
            for (s2=line+52; s2 > t && isspace(*(s2-1)); s2--) ;
            isfp->fout_descr1 = mystrdup2(s, s2);
            s = line + 52;
            if (s < end && isdigit(*s)) {
              isfp->fout_len1 = *s - '0';
              for (s++; s < end && isdigit(*s); s++) {
                isfp->fout_len1 *= 10;
                isfp->fout_len1 += *s - '0';
              }
              s++;
              if (*s == 'a' && s[1] == 'a')
                isfp->fout_alpha1 = PROTEIN;
              else if (*s == 'n' && s[1] == 't')
                isfp->fout_alpha1 = DNA;
            }
          }
          state = 9;
        }
        break;

      case 9:
        error_test(line[0] != '>' || isfp->fout_id2 != NULL || end-line <= 54,
                   E_PARSEERROR, return STATUS_ERROR,
                   print_error("%s, entry %d:  Text differs from %s "
                               " output format.\n", isfp->filename,
                               isfp->entry_count, isfp->fout_progname));
        for (t=s=line+1; s < end && !isspace(*s); s++) ;
        if (s < end) {
          isfp->fout_id2 = mystrdup2(t, s);
          for (s++; s < end && isspace(*s); s++);
        }
        if (s < end) {
          for (s2=line+52; s2 > t && isspace(*(s2-1)); s2--) ;
          isfp->fout_descr2 = mystrdup2(s, s2);
          s = line + 52;
          if (s < end && isdigit(*s)) {
            isfp->fout_len2 = *s - '0';
            for (s++; s < end && isdigit(*s); s++) {
              isfp->fout_len2 *= 10;
              isfp->fout_len2 += *s - '0';
            }
            s++;
            if (*s == 'a' && s[1] == 'a')
              isfp->fout_alpha2 = PROTEIN;
            else if (*s == 'n' && s[1] == 't')
              isfp->fout_alpha2 = DNA;
          }
        }
        state = 10;
        break;

      case 10:
        runflag = 0;
        break;

      }
    }
    switch (status) {
    case STATUS_OK:    break;
    case STATUS_EOF:   error_test(state == 11,
                                  E_PARSEERROR, return STATUS_ERROR,
                                  print_error("%s:  Cannot determine which "
                                              "FASTA program generated the "
                                              "text.\n", isfp->filename));
                       raise_error(E_PARSEERROR, return STATUS_ERROR,
                                   print_error("%s:  File contains no "
                                               "entries.\n", isfp->filename));
    case STATUS_ERROR: return STATUS_ERROR;
    case STATUS_FATAL: return STATUS_FATAL;
    default:           status_error(return STATUS_ERROR, "fastaout_read");
    }

    error_test(line == end, E_PARSEERROR, return STATUS_ERROR, 
               print_error("%s, entry %d:  Text differs from %s "
                           " output format.\n", isfp->filename,
                           isfp->entry_count, isfp->fout_progname));

    isfp->fout_markx = (line[0] == '>' ? MARKX10 : NONPARSABLE);
    isfp->fp_current = line;
    isfp->entry_seqno = isfp->entry_numseqs = 0;
  }

  if (!flag && isfp->entry_seqno < isfp->entry_numseqs) {
    isfp->entry_seqno++;
    if (isfp->entry_truelen > 0)
      isfp->iflag_truelen = 1;
    if (isfp->entry_rawlen > 0)
      isfp->iflag_rawlen = 1;
    return STATUS_OK;
  }

  if (isfp->fout_mode == ALIGN_MODE && isfp->entry_count > 0)
    return STATUS_EOF;

  isfp->fp_entrystart = isfp->fp_seqstart = isfp->fp_entryend = NULL;
  isfp->entry_seqlen = isfp->entry_truelen = isfp->entry_rawlen = 0;
  isfp->iflag_truelen = isfp->iflag_rawlen = 0;
  isfp->entry_count++;
  isfp->entry_seqno = isfp->entry_numseqs = 0;

  markx = isfp->fout_markx;
  mode = isfp->fout_mode;

  if (markx == NONPARSABLE || markx == MARKX0 ||
      markx == MARKX2 || markx == MARKX3) {
    if (mode == LFASTA_MODE && isfp->entry_count > 1) {
      status = fp_get_line(isfp, &line, &end);
      switch (status) {
      case STATUS_OK:    break;
      case STATUS_EOF:   return STATUS_EOF;
      case STATUS_ERROR: return STATUS_ERROR;
      case STATUS_FATAL: return STATUS_FATAL;
      default:           status_error(return STATUS_ERROR, "fastaout_read");
      }

      error_test(line != end && !isspace(*line),
                 E_PARSEERROR, return STATUS_ERROR, 
                 print_error("%s, entry %d:  Text differs from %s "
                             " output format.\n", isfp->filename,
                             isfp->entry_count, isfp->fout_progname));
    }

    state = 0;
    count = 0;
    fasta_out_bug_flag = 0;
    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
      if (state == 0 && count == 0) {
        if (strncmp(line, "Library scan:", 13) == 0) {
          error_test(isfp->entry_count == 1, E_PARSEERROR, return STATUS_ERROR,
                     print_error("%s:  File contains no entries.\n",
                                 isfp->filename));
          return STATUS_EOF;
        }
        else
          isfp->fp_entrystart = line;
      }
      count++;

      if (state == 0 && (line == end || (line[0] == '>' && line[1] != '>'))) {
        if (markx == NONPARSABLE && line[0] == '>')
          markx = isfp->fout_markx = MARKX3;

        error_test((markx != MARKX3 && line != end) ||
                   (markx == MARKX3 && line[0] != '>'),
                   E_PARSEERROR, return STATUS_ERROR, 
                   print_error("%s, entry %d:  Text differs from %s "
                               " output format.\n", isfp->filename,
                               isfp->entry_count, isfp->fout_progname));
        state = (markx == MARKX3 ? 2 : 1);
        count = 0;
        if (isfp->fp_seqstart == NULL)
          isfp->fp_seqstart = line;
      }
      else if (state == 1) {
        if (count == 1) {
          if (line == end) {
            if (markx == MARKX2)
              fasta_out_bug_flag = 1;
            else {
              raise_error(E_PARSEERROR, return STATUS_ERROR, 
                          print_error("%s, entry %d:  Text differs from %s "
                                      " output format.\n", isfp->filename,
                                      isfp->entry_count, isfp->fout_progname));
            }
          }
          else if (!isspace(*line)) {
            if ((line[0] == '-' && strncmp(line, "----", 4) == 0) ||
                (line[0] == 'L' && strncmp(line, "Library ", 8) == 0))
              break;
            for (s=end-1; s > line && isspace(*s); s--) ;
            if (s - 3 > line && *s == ')' &&
                ((*(s-2) == 'a' && *(s-1) == 'a') ||
                 (*(s-2) == 'n' && *(s-1) == 't')))
              break;
            count = 4;
          }
        }
        else if (count == 2 && fasta_out_bug_flag) {
          /*
           * In this FASTA bug, an extra blank line is output at the
           * end of the entry (appearing where the first line of the
           * next entry should be).  So, we need to delay the text
           * for the next entry beginning to the next line.
           */
          if (!isspace(*line)) {
            if ((line[0] == '-' && strncmp(line, "----", 4) == 0) ||
                (line[0] == 'L' && strncmp(line, "Library ", 8) == 0))
              break;
            for (s=end-1; s > line && isspace(*s); s--) ;
            if (s - 3 > line && *s == ')' &&
                ((*(s-2) == 'a' && *(s-1) == 'a') ||
                 (*(s-2) == 'n' && *(s-1) == 't')))
              break;
          }
          raise_error(E_PARSEERROR, return STATUS_ERROR, 
                      print_error("%s, entry %d:  Text differs from %s "
                                  " output format.\n", isfp->filename,
                                  isfp->entry_count, isfp->fout_progname));
        }
        else if (count == 3) {
          if (line == end)
            count = 0;
          else {
            if (markx == NONPARSABLE)
              markx = isfp->fout_markx = (!isspace(*line) ? MARKX2 : MARKX0);

            if (markx == MARKX2)
              count+=2;
          }
        }
        else if (count == 6)
          count = 0;
      }
      else if (state == 2 && line[0] == '>')
        state = 3;
      else if (state == 3 && line == end)
        state = 4;
      else if (state == 4 && line != end)
        break;
    }
    if (status == STATUS_EOF && 
        ((state >= 3 && (markx == MARKX3 || mode == ALIGN_MODE)) ||
         (mode == ALIGN_MODE && state == 1 && (count == 0 || count == 5)))) {
      isfp->entry_seqno = 1;
      isfp->entry_numseqs = 2;
      isfp->fp_entryend = isfp->fp_top;
      return STATUS_OK;
    }
    switch (status) {
    case STATUS_OK:    break;
    case STATUS_EOF:   raise_error(E_PARSEERROR, return STATUS_ERROR,
                         print_error("%s, entry %d:  Premature EOF reached.\n",
                                     isfp->filename, isfp->entry_count));
    case STATUS_ERROR: return STATUS_ERROR;
    case STATUS_FATAL: return STATUS_FATAL;
    default:           status_error(return STATUS_ERROR, "fastaout_read");
    }

    error_test(markx == NONPARSABLE, E_PARSEERROR, return STATUS_ERROR, 
               print_error("%s, entry %d:  Text differs from %s "
                           " output format.\n", isfp->filename,
                           isfp->entry_count, isfp->fout_progname));

    if (mode == LFASTA_MODE)
      isfp->fp_entryend = isfp->fp_current = end + 1;
    else
      isfp->fp_entryend = isfp->fp_current = line;
  }
  else if (markx == MARKX10) {
    status = fp_get_line(isfp, &line, &end);
    switch (status) {
    case STATUS_OK:    break;
    case STATUS_EOF:   return STATUS_EOF;
    case STATUS_ERROR: return STATUS_ERROR;
    case STATUS_FATAL: return STATUS_FATAL;
    default:           status_error(return STATUS_ERROR, "fastaout_read");
    }

    if (strncmp(line, "Library scan:", 13) == 0)
      return STATUS_EOF;

    error_test(line[0] != '>' || line[1] != '>',
               E_PARSEERROR, return STATUS_ERROR, 
               print_error("%s, entry %d:  Text differs from %s "
                           " output format.\n", isfp->filename,
                           isfp->entry_count, isfp->fout_progname));
    error_test(isfp->entry_count == 1 && line[2] != '>',
               E_PARSEERROR, return STATUS_ERROR, 
               print_error("%s, entry %d:  Text differs from %s "
                           " output format.\n", isfp->filename,
                           isfp->entry_count, isfp->fout_progname));

    if (isfp->entry_count == 1) {
      while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
        if (line[0] == '>' && line[1] == '>')
          break;
      switch (status) {
      case STATUS_OK:    break;
      case STATUS_EOF:   raise_error(E_PARSEERROR, return STATUS_ERROR,
                         print_error("%s, entry %d:  Premature EOF reached.\n",
                                     isfp->filename, isfp->entry_count));
      case STATUS_ERROR: return STATUS_ERROR;
      case STATUS_FATAL: return STATUS_FATAL;
      default:           status_error(return STATUS_ERROR, "fastaout_read");
      }
    }

    isfp->fp_entrystart = line;
    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
      if ((line[0] == '>' && line[1] == '>') ||
          strncmp(line, "Library scan:", 13) == 0)
        break;
      else if (isfp->fp_seqstart == NULL && line[0] == '>')
        isfp->fp_seqstart = line;
    }
    switch (status) {
    case STATUS_OK:    break;
    case STATUS_EOF:   isfp->fp_entryend = isfp->fp_top;
                       isfp->entry_seqno = 1;
                       isfp->entry_numseqs = 2;
                       return STATUS_OK;
    case STATUS_ERROR: return STATUS_ERROR;
    case STATUS_FATAL: return STATUS_FATAL;
    default:           status_error(return STATUS_ERROR, "fastaout_read");
    }
    isfp->fp_entryend = isfp->fp_current = line;
  }

  isfp->entry_seqno = 1;
  isfp->entry_numseqs = 2;
  return STATUS_OK;
}



static int fastaout_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  int markx, initflag, seq1only, seq2only, templen;
  char *s, *end, *seq, *last, *seqline1;

  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));

  if (isfp->fp_seqstart == NULL) {
    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_rawlen = isfp->entry_truelen = 0;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;
      return STATUS_OK;
    }
    else {
      isfp->seqlen = 0;
      isfp->seq[0] = '\0';
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
  }

  s = isfp->fp_seqstart;
  end = isfp->fp_entryend;
  
  /*
   * Reallocate the sequence buffer, if necessary.
   */
  if (rawseqflag != GETSEQ_LENGTHS && (end - s + 1) / 2 >= isfp->seqsize) {
    isfp->seqsize += (end - s + 1) / 2;
    isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
    if (isfp->seq == NULL) {
      isfp->seqsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
  }
  seq = isfp->seq;

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->entry_truelen = isfp->entry_rawlen = 0;

  /*
   * Extract the sequence characters.
   */
  last = NULL;
  markx = isfp->fout_markx;
  if (markx == MARKX0 || markx == MARKX2) {
    initflag = 1;
    while (1) {
      while (s < end && *s != '\n') s++;    /* Skip blank line */
      s++;

      error_test(s >= end && initflag, E_PARSEERROR, return STATUS_ERROR, 
                 print_error("%s, entry %d:  Entry contains no sequences.\n",
                             isfp->filename, isfp->entry_count));
      initflag = 0;
      if (s >= end || (*s == '-' && s[1] == '-' && s[2] == '-'))
        break;

      if (*s == '\n' && s+1 == end)  /* This handles the FASTA output bug */
        break;                       /* described in fastaout_read        */

      seq1only = seq2only = 0;

      if (s < end && isspace(*s)) {
        while (s < end && *s != '\n') s++;    /* Skip seq 1 positions line */
        s++;
      }
      else
        seq2only = 1;

      seqline1 = s;

      if (isfp->entry_seqno == 2) {
        if (!seq2only) { 
          while (s < end && *s != '\n') s++;  /* Skip sequence 1 line */
          s++;

          if (s < end && *s == '\n') {
            seq1only = 1;
            s = seqline1;
          }
          else if (markx == MARKX0) {
            while (s < end && *s != '\n') s++;  /* Skip line of matches */
            s++;
          }
        }
      }

      error_test(s >= end, E_PARSEERROR, return STATUS_ERROR, 
                 print_error("%s, entry %d:  Premature end of entry.\n",
                             isfp->filename, isfp->entry_count));

      if (!seq1only && !seq2only &&
          markx == MARKX2 && isfp->entry_seqno == 2) {
        switch (rawseqflag) {
        case GETSEQ_SEQUENCE:
          for (s+=7,last=seqline1+7; s < end && *s != '\n'; s++,last++) {
            if (*s == '.' && isalpha(*last))
              *seq++ = *last;
            else if (isalpha(*s))
              *seq++ = *s;
          }
          break;

        case GETSEQ_RAWSEQ:
          for (s+=7,last=seqline1+7; s < end && *s != '\n'; s++,last++) {
            if (*s == '.' && !(isspace(*last) || isdigit(*last)))
              *seq++ = *last;
            else if (*s != ' ')
              *seq++ = *s;
            else
              *seq++ = '-';
          }
          break;

        case GETSEQ_LENGTHS:
          for (s+=7,last=seqline1+7; s < end && *s != '\n'; s++,last++) {
            isfp->entry_rawlen++;
            if (isalpha(*s) || (*s == '.' && isalpha(*last)))
              isfp->entry_truelen++;
          }
          break;
        }
      }
      else if ((!seq1only && !seq2only) ||
               (seq1only && isfp->entry_seqno == 1) ||
               (seq2only && isfp->entry_seqno == 2)) {
        switch (rawseqflag) {
        case GETSEQ_SEQUENCE:
          for (s+=7; s < end && *s != '\n'; s++)
            if (isalpha(*s))
              *seq++ = *s;
          break;

        case GETSEQ_RAWSEQ:
          for (s+=7; s < end && *s != '\n'; s++)
            *seq++ = (*s == ' ' ? '-' : *s);
          break;

        case GETSEQ_LENGTHS:
          for (s+=7; s < end && *s != '\n'; s++) {
            isfp->entry_rawlen++;
            if (isalpha(*s))
              isfp->entry_truelen++;
          }
          break;
        }
      }
      else {
        switch (rawseqflag) {
        case GETSEQ_SEQUENCE:
        case GETSEQ_RAWSEQ:
          for (s+=7; s < end && *s != '\n'; s++)
            *seq++ = '-';
          break;

        case GETSEQ_LENGTHS:
          for (s+=7; s < end && *s != '\n'; s++) {
            isfp->entry_rawlen++;
            if (isalpha(*s))
              isfp->entry_truelen++;
          }
          break;
        }
      }
      s++;

      error_test(s >= end && initflag, E_PARSEERROR, return STATUS_ERROR, 
                 print_error("%s, entry %d:  Premature end of entry.\n",
                             isfp->filename, isfp->entry_count));

      if (*s == '\n' && !seq1only && isfp->entry_seqno == 1)
        seq1only = 1;

      if (seq1only)
        ;                          /* We're at the blank line, so do nothing */
      else if (seq2only) {
        while (s < end && *s != '\n') s++;      /* Skip seq 2 positions line */
        s++;
      }
      else if (isfp->entry_seqno == 1) {
        if (markx == MARKX0) {
          while (s < end && *s != '\n') s++;    /* Skip line of matches */
          s++;
        }
        while (s < end && *s != '\n') s++;      /* Skip sequence 2 line */
        s++;
        if (markx == MARKX0) {
          while (s < end && *s != '\n') s++;    /* Skip seq 2 positions line */
          s++;
        }
      }
      else {   /* isfp->entry_seqno == 2 */
        if (markx == MARKX0) {
          while (s < end && *s != '\n') s++;    /* Skip seq 2 positions line */
          s++;
        }
      }
    }
  }
  else if (markx == MARKX3 || markx == MARKX10) {
    while (s < end && *s != '\n') s++;

    templen = 0;
    if (isfp->entry_seqno == 2) {
      if (markx == MARKX3)
        for (s++; s < end && (*s != '\n' || s[1] != '>'); s++) ;
      else {
        while (s < end && (*s != '\n' || s[1] != '>')) {
          if (s[1] == ';')
            for (s++; s < end && *s != '\n'; s++) ;
          else
            for (s++; s < end && *s != '\n'; s++)
              templen++;
        }
      }

      for (s++; s < end && *s != '\n'; s++) ;
    }

    for (s++; s < end && *s != '>' && *s != '\n'; s++) {
      if (*s == ';')
        while (s < end && *s != '\n') s++;
      else {
        switch (rawseqflag) {
        case GETSEQ_SEQUENCE:
          for ( ; s < end && *s != '\n'; s++)
            if (isalpha(*s))
              *seq++ = *s;
          break;

        case GETSEQ_RAWSEQ:
          for ( ; s < end && *s != '\n'; s++)
            *seq++ = (*s == ' ' ? '-' : *s);
          break;

        case GETSEQ_LENGTHS:
          for ( ; s < end && *s != '\n'; s++) {
            isfp->entry_rawlen++;
            if (isalpha(*s))
              isfp->entry_truelen++;
          }
          break;
        }
      }
    }

    if (markx == MARKX10) {
      if (isfp->entry_seqno == 1) {
        error_test(s >= end, E_PARSEERROR, return STATUS_ERROR, 
                   print_error("%s, entry %d:  Premature end of entry.\n",
                               isfp->filename, isfp->entry_count));

        if (*s != '>') {
          while (s < end && (*s != '\n' || s[1] != '>')) s++;
          s++;
        }
        while (s < end && *s != '\n') s++;
        while (s < end && (*s != '\n' || s[1] != '>')) {
          if (s[1] == ';')
            for (s++; s < end && *s != '\n'; s++) ;
          else
            for (s++; s < end && *s != '\n'; s++)
              templen++;
        }
      }

      while (seq - isfp->seq < templen)
        *seq++ = '-';
    }
  }

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;
  else {
    *seq = '\0';
    isfp->seqlen = seq - isfp->seq;

    /*
     * Perform checks on the sequence length.
     */
    if (isfp->seqlen == 0) {
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }

    if (rawseqflag == GETSEQ_SEQUENCE) {
      isfp->entry_truelen = isfp->seqlen;
      isfp->iflag_truelen = 1;
    }
    else if (rawseqflag == GETSEQ_RAWSEQ) {
      isfp->entry_rawlen = isfp->seqlen;
      isfp->iflag_rawlen = 1;
    }
  }

  return STATUS_OK;
}



/*
 * blastout_read     (BLAST output formats)
 *
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *
 * Return: a STATUS value
 */
static int blastout_read(INTSEQFILE *isfp, int flag)
{
  int status, count;
  char ch, *s, *t, *line, *end;

  if (isfp->entry_count == 0) {
    isfp->fout_alpha1 = UNKNOWN;
    isfp->fout_progname[0] = '\0';
    isfp->fout_descr1 = isfp->fout_descr2 = NULL;

    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
      if (mystreq(line, 'Q', "QUERY=") ||
          (mystreq(line, 'B', "BLAST") &&
           ((ch = toupper(line[6])) == 'N' || ch == 'P' || ch == 'X')))
        break;
    }

    if (status == STATUS_OK && toupper(line[0]) == 'B') {
      memcpy(isfp->fout_progname, line, 6);
      isfp->fout_progname[6] = '\0';
      ch = toupper(line[5]);
      if (ch == 'N')
        isfp->fout_alpha1 = DNA;
      else if (ch == 'P' || ch == 'X')
        isfp->fout_alpha1 = PROTEIN;

      while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
        if (mystreq(line, 'Q', "QUERY="))
          break;
    }
    else if (status == STATUS_OK)
      strcpy(isfp->fout_progname, "BLASTN/BLASTP/BLASTX");

    if (status == STATUS_OK) {
      for (s=end-1; s >= line && isspace(*s); s--) ;
      if (s - 4 >= line && strncmp(s-4, "bases", 5) == 0) {
        for (s-=5; s >= line && isspace(*s); s--) ;
        for ( ; s >= line && isdigit(*s); s--) ;
        for ( ; s >= line && isspace(*s); s--) ;
        if (s >= line && *s == ',')
          end = s;
      }

      for (t=line+6; t < end && isspace(*t); t++) ;
      if (t < s)
        isfp->fout_descr1 = mystrdup2(t, s);

      status = fp_get_line(isfp, &line, &end);
    }

    if (status == STATUS_OK) {
      for (s=line; s < end && *s != '('; s++) ;
      if (s < end)
        isfp->fout_len1 = myatoi(s+1, 10, '0');

      while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
        if (line[0] == '>')
          break;
    }

    isfp->entry_seqno = isfp->entry_numseqs = 0;

    switch (status) {
    case STATUS_OK:     break;
    case STATUS_EOF:    raise_error(E_PARSEERROR, return STATUS_ERROR,
                                    print_error("%s, entry %d:  Premature EOF "
                                                "reached.\n", isfp->filename,
                                                isfp->entry_count));
    case STATUS_ERROR:  return STATUS_ERROR;
    case STATUS_FATAL:  return STATUS_FATAL;
    default:
      status_error(return STATUS_ERROR, "blastout_read");
    }

    isfp->fp_current = line;
  }

  if (!flag && isfp->entry_seqno < isfp->entry_numseqs) {
    isfp->entry_seqno++;
    if (isfp->entry_truelen > 0)
      isfp->iflag_truelen = 1;
    if (isfp->entry_rawlen > 0)
      isfp->iflag_rawlen = 1;

    return STATUS_OK;
  }

  isfp->fp_entrystart = isfp->fp_seqstart = isfp->fp_entryend = NULL;
  isfp->entry_seqlen = isfp->entry_truelen = isfp->entry_rawlen = 0;
  isfp->iflag_truelen = isfp->iflag_rawlen = 0;
  isfp->entry_count++;
  isfp->entry_seqno = isfp->entry_numseqs = 0;

  status = fp_get_line(isfp, &line, &end);
  while (status == STATUS_OK && line == end)
    status = fp_get_line(isfp, &line, &end);

  /*
   * Check for the end of the input.
   *
   * The line here must either be the beginning of a sequence's matches
   * (and so start with '>') or it must be in the middle of a sequence's
   * matches (which is signalled by the strings "  Plus Strand HSPs:",
   * "  Minus Strand HSPs:" and " Score =".
   */
  if (status == STATUS_OK) {
    if (line[0] == '>')
      ;
    else {
      for (s=line; s < end && isspace(*s); s++) ;
      if (s == end || (!mystreq(s, 'P', "PLUS") && !mystreq(s, 'M', "MINUS") &&
                       !mystreq(s, 'S', "SCORE")))
        return STATUS_EOF;
    }
  }

  /*
   * If at the first of a sequence's matches, read the header to get
   * the sequence's oneline description and length.
   */
  if (status == STATUS_OK && line[0] == '>') {
    isfp->fp_entrystart = line;
    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
      for (s=line; s < end && isspace(*s); s++) ;
      error_test(s == end, E_PARSEERROR, return STATUS_ERROR,
                 print_error("%s, entry %d:  Invalid format of BLAST-output "
                             "entry.\n", isfp->filename, isfp->entry_count));
      if (mystreq(s, 'L', "LENGTH =")) {
        isfp->fout_len2 = myatoi(s+8, 10, '0');
        break;
      }
    }

    if (status == STATUS_OK) {
      isfp->fout_descr2 = mystrdup2(isfp->fp_entrystart, line);
      for (s=t=isfp->fout_descr2; *s; ) {
        while (*s && *s != '\n') {
          if (s != t)
            *t++ = *s;
          s++;
        }

        if (*s == '\n') {
          while (*s && isspace(*s)) s++;
          if (*s)
            *t++ = ' ';
        }
      }
      if (s != t)
        *t = '\0';
      
      if ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
        status = fp_get_line(isfp, &line, &end);
    }
  }

  /*
   * Advance to the beginning of the alignment lines.
   */
  if (status == STATUS_OK) {
    isfp->fp_entrystart = line;

    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK)
      if (mystreq(line, 'Q', "QUERY:"))
        break;
  }

  /*
   * Scan to the end of the alignment lines.
   */
  if (status == STATUS_OK) {
    isfp->fp_seqstart = line;

    count = 0;
    while (status == STATUS_OK) {
      if (count == 4) {
        if (!mystreq(line, 'Q', "QUERY:"))
          break;
        count = 0;
      }

      count++;
      status = fp_get_line(isfp, &line, &end);
    }
  }

  switch (status) {
  case STATUS_OK:     break;
  case STATUS_EOF:    raise_error(E_PARSEERROR, return STATUS_ERROR,
                                  print_error("%s, entry %d:  Premature EOF "
                                              "reached.\n", isfp->filename,
                                              isfp->entry_count));
  case STATUS_ERROR:  return STATUS_ERROR;
  case STATUS_FATAL:  return STATUS_FATAL;
  default:
    status_error(return STATUS_ERROR, "blastout_read");
  }

  isfp->fp_entryend = isfp->fp_current = line;
  isfp->entry_seqno = 1;
  isfp->entry_numseqs = 2;

  return STATUS_OK;
}



static int blastout_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  char *s, *end, *seq;

  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));

  if (isfp->fp_seqstart == NULL) {
    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_rawlen = isfp->entry_truelen = 0;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;
      return STATUS_OK;
    }
    else {
      isfp->seqlen = 0;
      isfp->seq[0] = '\0';
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
  }

  s = isfp->fp_seqstart;
  end = isfp->fp_entryend;
  
  /*
   * Reallocate the sequence buffer, if necessary.
   */
  if (rawseqflag != GETSEQ_LENGTHS && (end - s + 1) / 2 >= isfp->seqsize) {
    isfp->seqsize += (end - s + 1) / 2;
    isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
    if (isfp->seq == NULL) {
      isfp->seqsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
  }
  seq = isfp->seq;

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->entry_truelen = isfp->entry_rawlen = 0;

  /*
   * Extract the sequence characters.
   */
  if (isfp->entry_seqno == 2) {
    while (s < end && *s != '\n') s++;
    for (s++; s < end && *s != '\n'; s++) ;
    s++;
  }

  while (s < end) {
    switch (rawseqflag) {
    case GETSEQ_SEQUENCE:
      for (s+=6; s < end && *s != '\n'; s++) {
        if (isalpha(*s))
          *seq++ = *s;
      }
      break;

    case GETSEQ_RAWSEQ:
      for (s+=6; s < end && *s != '\n'; s++) {
        if (!isspace(*s) && !isdigit(*s))
          *seq++ = *s;
      }
      break;

    case GETSEQ_LENGTHS:
      for (s+=6; s < end && *s != '\n'; s++) {
        if (!isspace(*s) && !isdigit(*s)) {
          isfp->entry_rawlen++;
          if (isalpha(*s))
            isfp->entry_truelen++;
        }
      }
      break;
    }

    for (s++; s < end && *s != '\n'; s++) ;
    for (s++; s < end && *s != '\n'; s++) ;
    for (s++; s < end && *s != '\n'; s++) ;
    s++;
  }

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;
  else {
    *seq = '\0';
    isfp->seqlen = seq - isfp->seq;

    /*
     * Perform checks on the sequence length.
     */
    if (isfp->seqlen == 0) {
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }

    if (rawseqflag == GETSEQ_SEQUENCE) {
      isfp->entry_truelen = isfp->seqlen;
      isfp->iflag_truelen = 1;
    }
    else if (rawseqflag == GETSEQ_RAWSEQ) {
      isfp->entry_rawlen = isfp->seqlen;
      isfp->iflag_rawlen = 1;
    }
  }

  return STATUS_OK;
}



static int phylip_read(INTSEQFILE *isfp, int flag)
{
  int count, status, numseqs, seqlen, int_seqpos, int_seqnum;
  int int_flag, int_runflag, int_lcount, seq_seqpos, seq_seqnum;
  int seq_flag, seq_runflag, init_flag, len, num_trees, multi_flag;
  int state, flagcount, aflag, cflag, fflag, mflag, uflag, wflag;
  char ch, *s, *s2, *t2, *line, *end, *int_entryend, *seq_entryend, *seqstart;

  if (isfp->entry_count == 0) {
    isfp->entry_seqno = isfp->entry_numseqs = 0;
    isfp->phylip_origfmt = isfp->format;
  }

  if (!flag && isfp->entry_seqno < isfp->entry_numseqs) {
    isfp->entry_seqno++;
    if (isfp->entry_rawlen > 0)
      isfp->iflag_rawlen = 1;
    return STATUS_OK;
  }

  isfp->fp_entrystart = isfp->fp_seqstart = isfp->fp_entryend = NULL;
  isfp->entry_seqlen = isfp->entry_truelen = isfp->entry_rawlen = 0;
  isfp->iflag_truelen = isfp->iflag_rawlen = 0;
  isfp->entry_count++;
  isfp->entry_seqno = isfp->entry_numseqs = 0;

  s = NULL;
  while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
    for (s=line; s < end && isspace(*s); s++) ;
    if (s == end)
      continue;
    else if (isdigit(*s))
      break;
    else
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("%s, entry %d:  Invalid first line of PHYLIP "
                              "entry.\n", isfp->filename, isfp->entry_count));
  }
  switch (status) {
  case STATUS_OK:     break;
  case STATUS_EOF:    error_test(isfp->entry_count == 1,
                                 E_PARSEERROR, return STATUS_ERROR,
                                 print_error("%s, entry %d:  File contains no "
                                             "entries.\n", isfp->filename,
                                             isfp->entry_count));
                      return STATUS_EOF;
  case STATUS_ERROR:  return STATUS_ERROR;
  case STATUS_FATAL:  return STATUS_FATAL;
  default:            status_error(return STATUS_ERROR, "phylip_read");
  }

  isfp->fp_entrystart = line;
  numseqs = *s - '0';
  for (s++; s < end && isdigit(*s); s++) {
    numseqs *= 10;
    numseqs += *s - '0';
  }
  while (s < end && isspace(*s)) s++;
  error_test(s == end || !isdigit(*s), E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry %d:  Invalid first line of PHYLIP "
                         "entry.\n", isfp->filename, isfp->entry_count)); 
  seqlen = *s - '0';
  for (s++; s < end && isdigit(*s); s++) {
    seqlen *= 10;
    seqlen += *s - '0';
  }

  flagcount = aflag = cflag = fflag = mflag = uflag = wflag = 0;
  for ( ; s < end; s++) {
    switch (*s) {
    case 'a':  case 'A':  aflag = 1; flagcount++; break;
    case 'c':  case 'C':  cflag = 1; flagcount++; break;
    case 'f':  case 'F':  fflag = 1; flagcount++; break;
    case 'm':  case 'M':  mflag = 1; flagcount++; break;
    case 'u':  case 'U':  uflag = 1; break;
    case 'w':  case 'W':  wflag = 1; flagcount++; break;
    }
  }

  int_seqnum = int_seqpos = int_lcount = 0;
  int_runflag = int_flag = 1;
  int_entryend = NULL;

  seq_seqnum = seq_seqpos = 0;
  seq_runflag = seq_flag = 1;
  seq_entryend = NULL;

  init_flag = 1;
  multi_flag = 0;
  seqstart = NULL;
  while ((int_runflag || seq_runflag) &&
         (status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
    for (s=line; s < end && isspace(*s); s++) ;
    if (s == end)
      continue;

    if (flagcount) {
      ch = toupper(line[0]);
      if (aflag && ch == 'A') {
        for (s2=line+1,t2="NCESTOR  "; s2 < end && *t2; s2++,t2++)
          if (*s2 != ' ' && toupper(*s2) != *t2)
            break;
        if (*t2)
          flagcount = 0;
        else
          continue;
      }
      else if (cflag && ch == 'C') {
        for (s2=line+1,t2="ATEGORIES"; s2 < end && *t2; s2++,t2++)
          if (*s2 != ' ' && toupper(*s2) != *t2)
            break;
        if (*t2)
          flagcount = 0;
        else
          continue;
      }
      else if (fflag && ch == 'F') {
        for (s2=line+1,t2="ACTORS   "; s2 < end && *t2; s2++,t2++)
          if (*s2 != ' ' && toupper(*s2) != *t2)
            break;
        if (*t2)
          flagcount = 0;
        else
          continue;
      }
      else if (mflag && ch == 'M') {
        for (s2=line+1,t2="IXTURE   "; s2 < end && *t2; s2++,t2++)
          if (*s2 != ' ' && toupper(*s2) != *t2)
            break;
        if (*t2)
          flagcount = 0;
        else
          continue;
      }
      else if (wflag && ch == 'W') {
        for (s2=line+1,t2="EIGHTS   "; s2 < end && *t2; s2++,t2++)
          if (*s2 != ' ' && toupper(*s2) != *t2)
            break;
        if (*t2)
          flagcount = 0;
        else
          continue;
      }
      else
        flagcount = 0;
    }

    len = s - line;

    if (init_flag) {
      seqstart = line;
      init_flag = 0;
    }

    if (seq_runflag) {
      if (seq_flag) {
        if (end - line < 10 || len >= 10)
          seq_runflag = 0;
        else {
          s = line + 10;
          seq_flag = 0;
        }
      }
      else {
        s = line;
        multi_flag = 0;
      }

      if (seq_runflag) {
        for ( ; s < end; s++)
          if (!(isspace(*s) || isdigit(*s)))
            seq_seqpos++;

        if (seq_seqpos > seqlen)
          seq_runflag = 0;
        else if (seq_seqpos == seqlen) {
          seq_seqnum++;
          if (seq_seqnum == numseqs) {
            seq_entryend = end + 1;
            seq_runflag = 0;
          }
          else {
            seq_seqpos = 0;
            seq_flag = 1;
          }
        }
      }
    }

    if (int_runflag) {
      if (int_flag) {
        if (end - line < 10 || len >= 10)
          int_runflag = 0;
        else
          s = line + 10;
      }
      else {
        s = line;
        multi_flag = 0;
      }

      if (int_runflag) {
        for (count=0; s < end; s++)
          if (!(isspace(*s) || isdigit(*s)))
            count++;

        if (int_seqnum == 0) {
          int_lcount = count;
          int_seqpos += count;
          if (int_seqpos > seqlen)
            int_runflag = 0;
          int_seqnum++;
        }
        else {
          if (count != int_lcount)
            int_runflag = 0;
          else if (++int_seqnum == numseqs) {
            if (int_seqpos == seqlen) {
              int_entryend = end + 1;
              int_runflag = 0;
            }
            else {
              int_seqnum = 0;
              int_lcount = 0;
              int_flag = 0;
            }
          }
        }
      }
    }
  }
  switch (status) {
  case STATUS_OK:     break;
  case STATUS_EOF:    raise_error(E_PARSEERROR, return STATUS_ERROR,
                        print_error("%s, entry %d:  Premature EOF reached.\n",
                                    isfp->filename, isfp->entry_count));
  case STATUS_ERROR:  return STATUS_ERROR;
  case STATUS_FATAL:  return STATUS_FATAL;
  default:            status_error(return STATUS_ERROR, "phylip_read");
  }

  error_test(!seq_entryend && !int_entryend, E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry %d:  Entry text does not match PHYLIP "
                         "format.\n", isfp->filename, isfp->entry_count));
  error_test(multi_flag && seq_entryend && int_entryend,
             E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry %d:  Ambiguous entry, may be interleaved "
                         "or sequential.\n", isfp->filename,
                         isfp->entry_count));

  status = STATUS_OK;
  if (seq_entryend && (multi_flag || isfp->phylip_origfmt != FORMAT_PHYINT)) {
    if (multi_flag && isfp->phylip_origfmt == FORMAT_PHYINT) {
      set_error(E_INVFORMAT);
      print_warning("Warning:  %s, entry %d:  PHYLIP Interleaved format "
                    "specified, but Sequential format found.\n",
                    isfp->filename, isfp->entry_count);
      status = STATUS_WARNING;
    }
    isfp->format = FORMAT_PHYSEQ;
    isfp->fp_entryend = seq_entryend;
  }
  else {
    if (isfp->phylip_origfmt == FORMAT_PHYSEQ) {
      set_error(E_INVFORMAT);
      print_warning("Warning:  %s, entry %d:  PHYLIP Sequential format "
                    "specified, but Interleaved format found.\n",
                    isfp->filename, isfp->entry_count);
      status = STATUS_WARNING;
    }
    isfp->format = FORMAT_PHYINT;
    isfp->fp_entryend = int_entryend;
  }

  if (uflag) {
    isfp->fp_entryend = NULL;
    state = num_trees = 0;
    while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
      if (!state) {
        for (s=line; s < end && isspace(*s); s++) ;
        if (s == end)
          continue;
        error_test(!isdigit(*s),  E_PARSEERROR, return STATUS_ERROR,
                   print_error("%s, entry %d:  Invalid user tree section of "
                               "PHYLIP entry.\n", isfp->filename,
                               isfp->entry_count)); 
        num_trees = *s - '0';
        for (s++; s < end && isdigit(*s); s++) {
          num_trees *= 10;
          num_trees += *s - '0';
        }
        state = 1;
      }
      else {
        for (s=line; s < end && *s != ';'; s++) ;
        if (s == end)
          continue;
        if (--num_trees == 0) {
          isfp->fp_entryend = end + 1;
          break;
        }
      }
    }
    switch (status) {
    case STATUS_OK:     break;
    case STATUS_EOF:    raise_error(E_PARSEERROR, return STATUS_ERROR,
                         print_error("%s, entry %d:  Premature EOF reached.\n",
                                     isfp->filename, isfp->entry_count));
    case STATUS_ERROR:  return STATUS_ERROR;
    case STATUS_FATAL:  return STATUS_FATAL;
    default:            status_error(return STATUS_ERROR, "phylip_read");
    }
  }

  isfp->fp_seqstart = seqstart;
  isfp->malign_seqno = 1;

  isfp->entry_seqno = 1;
  isfp->entry_numseqs = numseqs;
  isfp->entry_rawlen = isfp->entry_seqlen = seqlen;
  isfp->iflag_rawlen = 1;
  return status;
}


static int phyint_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  int seqpos, seqnum, flag;
  char *s, *end, *seq, *line;

  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));

  if (isfp->fp_seqstart == NULL) {
    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_rawlen = isfp->entry_truelen = 0;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;
      return STATUS_OK;
    }
    else {
      isfp->seqlen = 0;
      isfp->seq[0] = '\0';
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
  }

  while (isfp->malign_seqno < isfp->entry_seqno) {
    for (s=isfp->fp_seqstart; isspace(*s); s++) ;
    for (s++; *s != '\n'; s++) ;
    isfp->fp_seqstart = s + 1;
    isfp->malign_seqno++;
  }

  /*
   * Reallocate the sequence buffer, if necessary.
   */
  if (rawseqflag != GETSEQ_LENGTHS && isfp->entry_seqlen >= isfp->seqsize) {
    isfp->seqsize += isfp->entry_seqlen;
    isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
    if (isfp->seq == NULL) {
      isfp->seqsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
  }
  seq = isfp->seq;

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->entry_truelen = isfp->entry_rawlen = 0;

  /*
   * Extract the sequence characters.
   */
  s = isfp->fp_seqstart;
  end = isfp->fp_entryend;

  seqpos = 0;
  seqnum = isfp->malign_seqno;
  flag = 1;

  while (s < end) {
    for (line=s; s < end && isspace(*s) && *s != '\n'; s++) ;
    if (s == end)
      break;
    else if (*s == '\n') {
      s++;
      continue;
    }

    if (flag)
      s = line + 10;

    if (seqnum == isfp->malign_seqno) {
      switch (rawseqflag) {
      case GETSEQ_SEQUENCE:
        for ( ; s < end && *s != '\n'; s++) {
          if (!(isspace(*s) || isdigit(*s))) {
            seqpos++;
            if (isalpha(*s))
              *seq++ = *s;
          }
        }
        break;

      case GETSEQ_RAWSEQ:
        for ( ; s < end && *s != '\n'; s++) {
          if (!(isspace(*s) || isdigit(*s))) {
            seqpos++;
            *seq++ = *s;
          }
        }
        break;

      case GETSEQ_LENGTHS:
        for ( ; s < end && *s != '\n'; s++) {
          if (!(isspace(*s) || isdigit(*s))) {
            seqpos++;
            isfp->entry_rawlen++;
            if (isalpha(*s))
              isfp->entry_truelen++;
          }
        }
        break;
      }
      if (seqpos == isfp->entry_seqlen)
        break;
    }
    else
      for ( ; s < end && *s != '\n'; s++) ;
    
    s++;
    if (seqnum++ == isfp->entry_numseqs) {
      seqnum = 1;
      flag = 0;
    }
  }
  
  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;
  else {
    *seq = '\0';
    isfp->seqlen = seq - isfp->seq;

    /*
     * Perform checks on the sequence length.
     */
    if (isfp->seqlen == 0) {
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
    if (rawseqflag && isfp->entry_seqlen > 0 &&
        isfp->entry_seqlen != isfp->seqlen) {
      set_error(E_DIFFLENGTH);
      print_warning("Warning:  %s, entry %d:  Entry gives seq. length of %d, "
                    "but %d characters found.\n", isfp->filename,
                    isfp->entry_count, isfp->entry_seqlen, isfp->seqlen);
      return STATUS_WARNING;
    }

   if (rawseqflag == GETSEQ_SEQUENCE) {
      isfp->entry_truelen = isfp->seqlen;
      isfp->iflag_truelen = 1;
    }
    else if (rawseqflag == GETSEQ_RAWSEQ) {
      isfp->entry_rawlen = isfp->seqlen;
      isfp->iflag_rawlen = 1;
    }
  }

  return STATUS_OK;
}

static int physeq_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  int seqlen, seqpos;
  char *s, *end, *seq;

  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));

  if (isfp->fp_seqstart == NULL) {
    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_rawlen = isfp->entry_truelen = 0;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;
      return STATUS_OK;
    }
    else {
      isfp->seqlen = 0;
      isfp->seq[0] = '\0';
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
  }

  while (isfp->malign_seqno < isfp->entry_seqno) {
    s = isfp->fp_seqstart+10;
    end = isfp->fp_entryend;
    seqlen = isfp->entry_seqlen;
    for (seqpos=0; s < end && seqpos < seqlen; s++)
      if (!(isspace(*s) || isdigit(*s)))
        seqpos++;
    for ( ; *s != '\n'; s++) ;
    isfp->fp_seqstart = s + 1;
    isfp->malign_seqno++;
  }

  /*
   * Reallocate the sequence buffer, if necessary.
   */
  if (rawseqflag != GETSEQ_LENGTHS && isfp->entry_seqlen >= isfp->seqsize) {
    isfp->seqsize += isfp->entry_seqlen;
    isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
    if (isfp->seq == NULL) {
      isfp->seqsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
  }
  seq = isfp->seq;

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->entry_truelen = isfp->entry_rawlen = 0;

  /*
   * Extract the sequence characters.
   */
  s = isfp->fp_seqstart + 10;
  end = isfp->fp_entryend;

  seqlen = isfp->entry_seqlen;
  switch (rawseqflag) {
  case GETSEQ_SEQUENCE:
    for (seqpos=0; s < end && seqpos < seqlen; s++) {
      if (!(isspace(*s) || isdigit(*s))) {
        seqpos++;
        if (isalpha(*s))
          *seq++ = *s;
      }
    }
    break;

  case GETSEQ_RAWSEQ:
    for (seqpos=0; s < end && seqpos < seqlen; s++) {
      if (!(isspace(*s) || isdigit(*s))) {
        seqpos++;
        *seq++ = *s;
      }
    }
    break;

  case GETSEQ_LENGTHS:
    for (seqpos=0; s < end && seqpos < seqlen; s++) {
      if (!(isspace(*s) || isdigit(*s))) {
        seqpos++;
        isfp->entry_rawlen++;
        if (isalpha(*s))
          isfp->entry_truelen++;
      }
    }
    break;
  }

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;
  else {
    *seq = '\0';
    isfp->seqlen = seq - isfp->seq;

    /*
     * Perform checks on the sequence length.
     */
    if (isfp->seqlen == 0) {
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
    if (rawseqflag && isfp->entry_seqlen > 0 &&
        isfp->entry_seqlen != isfp->seqlen) {
      set_error(E_DIFFLENGTH);
      print_warning("Warning:  %s, entry %d:  Entry gives seq. length of %d, "
                    "but %d characters found.\n", isfp->filename,
                    isfp->entry_count, isfp->entry_seqlen, isfp->seqlen);
      return STATUS_WARNING;
    }

   if (rawseqflag == GETSEQ_SEQUENCE) {
      isfp->entry_truelen = isfp->seqlen;
      isfp->iflag_truelen = 1;
    }
    else if (rawseqflag == GETSEQ_RAWSEQ) {
      isfp->entry_rawlen = isfp->seqlen;
      isfp->iflag_rawlen = 1;
    }
  }

  return STATUS_OK;
}



static int clustal_read(INTSEQFILE *isfp, int flag)
{
  int status, count;
  char *s, *end;

  if (isfp->entry_count == 0) {
    isfp->entry_count++;
    isfp->entry_seqlen = isfp->entry_truelen = isfp->entry_rawlen = 0;
    isfp->iflag_truelen = isfp->iflag_rawlen = 0;
    isfp->entry_seqno = isfp->entry_numseqs = 0;

    isfp->fp_entrystart = isfp->fp_current;
    isfp->fp_seqstart = isfp->fp_entryend = NULL;

    /*
     * The file should contain only a single entry, so just read the
     * whole file.
     */
    status = fp_read_all(isfp);
    switch (status) {
    case STATUS_OK:    break;
    case STATUS_EOF:   raise_error(E_PARSEERROR, return STATUS_ERROR,
                         print_error("%s:  Empty file.\n", isfp->filename));
    case STATUS_ERROR: return STATUS_ERROR;
    case STATUS_FATAL: return STATUS_FATAL;
    default:           status_error(return STATUS_ERROR, "clustal_read");
    }

    isfp->fp_entryend = isfp->fp_top;

    /*
     * Find out how many sequences occur in the file.
     */
    end = isfp->fp_entryend;
    for (s=isfp->fp_entrystart; s < end && *s != '\n'; s++) ;
    for ( ; s < end && (*s != '\n' || isspace(s[1])); s++) ;
    isfp->fp_seqstart = s+1;
    for (count=0; s < end && (*s != '\n' || !isspace(s[1])); s++)
      if (*s == '\n')
        count++;

    error_test(s == end || count == 0, E_PARSEERROR, return STATUS_ERROR,
               print_error("%s, entry %d:  Invalid format of CLUSTALW "
                           "entry.\n", isfp->filename, isfp->entry_count));

    isfp->malign_seqno = isfp->entry_seqno = 1;
    isfp->entry_numseqs = count;
    return STATUS_OK;
  }

  if (!flag && isfp->entry_seqno < isfp->entry_numseqs) {
    isfp->entry_seqno++;
    if (isfp->entry_rawlen > 0)
      isfp->iflag_rawlen = 1;
    return STATUS_OK;
  }

  return STATUS_EOF;
}


static int clustal_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  int count, seqsize;
  char *s, *end, *seq;

  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));

  if (isfp->fp_seqstart == NULL) {
    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_rawlen = isfp->entry_truelen = 0;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;
      return STATUS_OK;
    }
    else {
      isfp->seqlen = 0;
      isfp->seq[0] = '\0';
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
  }

  s = isfp->fp_seqstart;
  end = isfp->fp_entryend;

  if (isfp->malign_seqno < isfp->entry_seqno) {
    while (isfp->malign_seqno < isfp->entry_seqno) {
      for ( ; s < end && *s != '\n'; s++) ;
      s++;
      isfp->malign_seqno++;
    }
    isfp->fp_seqstart = s;
  }

  /*
   * Reallocate the sequence buffer, if necessary.
   */
  if (rawseqflag != GETSEQ_LENGTHS) {
    seqsize = (isfp->fp_entryend - isfp->fp_entrystart) /
              (isfp->entry_numseqs + 1);
    if (seqsize >= isfp->seqsize) {
      isfp->seqsize += seqsize;
      isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
      if (isfp->seq == NULL) {
        isfp->seqsize = 0;
        memory_error(1, return STATUS_FATAL);
      }
    }
  }
  seq = isfp->seq;

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->entry_truelen = isfp->entry_rawlen = 0;

  /*
   * Extract the sequence characters.
   */
  while (s < end) {
    /*
     * We're at a line of the sequence, so get it, then skip to the
     * end of the block, the beginning of the next block, and then
     * count lines down to the next piece of the sequence.
     */
    switch (rawseqflag) {
    case GETSEQ_SEQUENCE:
      for (s+=15; s < end && *s != '\n'; s++)
        if (isalpha(*s))
          *seq++ = *s;
      break;

    case GETSEQ_RAWSEQ:
      for (s+=15; s < end && *s != '\n'; s++)
        if (!(isspace(*s) || isdigit(*s)))
          *seq++ = *s;
      break;

    case GETSEQ_LENGTHS:
      for (s+=15; s < end && *s != '\n'; s++) {
        if (!(isspace(*s) || isdigit(*s))) {
          isfp->entry_rawlen++;
          if (isalpha(*s))
            isfp->entry_truelen++;
        }
      }
      break;
    }

    for (s++; s < end && (*s != '\n' || !isspace(s[1])); s++) ;
    for (s++; s < end && (*s != '\n' || isspace(s[1])); s++) ;
    for (count=0; s < end && (*s != '\n' || !isspace(s[1])); s++)
      if (*s == '\n' && ++count == isfp->malign_seqno)
        break;
  }

  if (rawseqflag == GETSEQ_LENGTHS)
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;
  else {
    *seq = '\0';
    isfp->seqlen = seq - isfp->seq;

    /*
     * Perform a check on the sequence length.
     */
    if (isfp->seqlen == 0) {
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }

   if (rawseqflag == GETSEQ_SEQUENCE) {
      isfp->entry_truelen = isfp->seqlen;
      isfp->iflag_truelen = 1;
    }
    else if (rawseqflag == GETSEQ_RAWSEQ) {
      isfp->entry_rawlen = isfp->seqlen;
      isfp->iflag_rawlen = 1;
    }
  }

  return STATUS_OK;
}

  
/*
 * asn_read   (ASN.1 file format)
 *
 * This parser (which, remember, only extracts the entry) 
 * assumes the following:
 *    1) The file text consists of a hierarchy of records, where a record
 *       consists of a text string identifier and then a pair of matching
 *       braces bounding the contents of the record.  Consecutive records
 *       on the same level of the hierarchy are separated by commas.
 *    2) The file can be completely free form, with the identifiers, braces
 *       and commas occurring anywhere.  In addition, single-quoted and 
 *       double-quoted strings can occur anywhere in the record, and
 *       braces and commas in those strings are ignored (as far as the
 *       hierarchy of records is concerned).
 *    3) The entries in the file are the "seq" records.  The beginning
 *       of the record is found by first looking for the string "seq {"
 *       (where "seq" is the complete identifier) and then searching
 *       for the matching close brace.
 *    4) The "seq" records can occur anywhere in the hierarchy, except that
 *       they cannot be nested, one within another.
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *
 * Return: a STATUS value
 */
#define LOOKFOR_BIOSET 0
#define LOOKFOR_SEQSET 1
#define LOOKFOR_SEQ 2
#define INENTRY 3

static int asn_read(INTSEQFILE *isfp, int flag)
{
  int level, seq_level, status, count, state, oldpe;
  char ch, *line, *end, *s, *t, *top, *lenstr, *lenend, *seqstr;

  isfp->fp_entrystart = isfp->fp_seqstart = isfp->fp_entryend = NULL;
  isfp->entry_seqlen = isfp->entry_truelen = isfp->entry_rawlen = 0;
  isfp->iflag_truelen = isfp->iflag_rawlen = 0;
  isfp->entry_count++;
  isfp->entry_seqno = isfp->entry_numseqs = 0;

  /*
   * Use local copies of fp_current and fp_top throughout this procedure.
   */
  s = isfp->fp_current;
  top = isfp->fp_top;

  /*
   * Get the complete sequence entry.
   */
  state = (isfp->entry_count == 1 ? LOOKFOR_BIOSET : LOOKFOR_SEQ);
  seq_level = (isfp->entry_count == 1 ? 2 : 0);
  line = s;
  level = 0;
  while ((status = fp_get_line(isfp, &line, &end)) == STATUS_OK) {
    s = line;
    while (1) {
      while (s < end && isspace(*s)) s++;
      if (s == end)
        break;

      switch (*s) {
      case '"':
      case '\'':
        ch = *s++;
        while (1) {
          while (s < end && *s != ch && *(s-1) != '\\') s++;
          if (s < end)
            break;

          status = fp_get_line(isfp, &line, &end);
          switch (status) {
          case STATUS_OK:     break;
          case STATUS_EOF:    raise_error(E_PARSEERROR, return STATUS_ERROR,
                                print_error("%s, entry %d:  Premature EOF "
                                            "reached.\n", isfp->filename,
                                            isfp->entry_count));
          case STATUS_ERROR:  return STATUS_ERROR;
          case STATUS_FATAL:  return STATUS_FATAL;
          default:            status_error(return STATUS_ERROR, "asn_read");
          }
          s = line;
        }
        s++;
        break;

      case '{':
        level++;
        if (state == LOOKFOR_SEQ && level - 1 == seq_level &&
            mystreq(s-4, 'S', "SEQ {")) {
          /*
           * Set the beginning of the entry mark to either the beginning
           * of the line (if "seq {" is the first thing on the line) or
           * right at the "seq {" string.
           */
          for (t=s-5; t > line && (*t == ' ' || *t == '\t'); t--) ;
          if (t == line)
            isfp->fp_entrystart = t;
          else if (*t == '\n')
            isfp->fp_entrystart = t + 1;
          else
            isfp->fp_entrystart = s - 4;

          state = INENTRY;
        }
        s++;
        break;

      case '}':
        if (state == INENTRY && level - 1 == seq_level)
          goto ASN_LOOP_END;

        level--;
        s++;
        break;

      default:
        if (isalpha(*s)) {
          if (state == LOOKFOR_BIOSET && level == 0 &&
              mystreq(s, 'B', "BIOSEQ-SET "))
            state = LOOKFOR_SEQSET;
          else if (state == LOOKFOR_SEQSET && level == 1 &&
                   mystreq(s, 'S', "SEQ-SET "))
            state = LOOKFOR_SEQ;
        }
        while (!isspace(*s)) s++;
      }
    }
  }
ASN_LOOP_END:
  switch (status) {
  case STATUS_OK:     break;
  case STATUS_EOF:    if (state != INENTRY) {
                        error_test(isfp->entry_count == 1,
                                   E_PARSEERROR, return STATUS_ERROR,
                                   print_error("%s:  File contains no "
                                               "entries.\n", isfp->filename));
                        return STATUS_EOF;
                      }
                      else {
                        raise_error(E_PARSEERROR, return STATUS_ERROR,
                          print_error("%s, entry %d:  Premature EOF "
                                      "reached.\n", isfp->filename,
                                      isfp->entry_count));
                      }
  case STATUS_ERROR:  return STATUS_ERROR;
  case STATUS_FATAL:  return STATUS_FATAL;
  default:            status_error(return STATUS_ERROR, "asn_read");
  }

  isfp->fp_entryend = s + 1;

  /*
   * Get the sequence length, if it is there.
   */
  lenstr = seqstr = NULL;
  oldpe = pe_flag;
  pe_flag = PE_NONE;
  count = asn_parse(isfp->fp_entrystart, isfp->fp_entryend,
                    "seq.inst.length", &lenstr, &lenend,
                    "seq.inst.seq-data", &seqstr, NULL,
                    NULL);
  pe_flag = oldpe;
  error_test(count == -1, E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry %d:  Invalid format of ASN.1 entry.\n",
                         isfp->filename, isfp->entry_count));

  if (count > 0) {
    if (lenstr != NULL) {
      isfp->entry_seqlen = myatoi(lenstr + 6, 10, '0');
      isfp->entry_rawlen = isfp->entry_truelen = isfp->entry_seqlen;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;

      error_test(isfp->entry_seqlen == 0, E_PARSEERROR, return STATUS_ERROR,
                 print_error("%s, entry %d:  Cannot parse `seq.inst.length' ",
                             "sub-record.\n", isfp->filename,
                             isfp->entry_count));
    }

    if (seqstr != NULL) {
      isfp->fp_seqstart = seqstr;
      isfp->entry_seqno = isfp->entry_numseqs = 1;
    }
  }

  return STATUS_OK;
}


/*
 * asn_getseq   (ASN.1 file format)
 *
 * This parser (which extracts the sequence and its length) 
 * assumes the following:
 *    1) The file text consists of a hierarchy of records, where a record
 *       consists of a text string identifier and then a pair of matching
 *       braces bounding the contents of the record (except for simple
 *       records which contain only a string or a number).  Consecutive
 *       records on the same level of the hierarchy are separated by commas.
 *    2) The file can be completely free form, with the identifiers, braces
 *       and commas occurring anywhere.  In addition, single-quoted and 
 *       double-quoted strings can occur anywhere in the record, and
 *       braces and commas in those strings are ignored (as far as the
 *       hierarchy of records is concerned.
 *    3) Inside the entry, the sequence and its length are given in the
 *       seq.inst.seq-data and seq.inst.length (where "a.b.c" describes
 *       a portion of the record hierarchy).  The length record consists
 *       of a single number, and the seq-data record consists of a 
 *       string naming the format and then a double-quoted string giving
 *       the sequence characters (formatted appropriately).
 *    4) For the sequence to be retrieved, the format of the seq-data
 *       record must be one of the following:  iupacna, iupacaa, ncbi2na.
 *
 * Parameters:  isfp        -  an opened INTSEQFILE structure
 *              rawseqflag  -  should the actual sequence or the raw
 *                             sequence text be extracted.
 *
 * Return: a STATUS value
 */
#define IUPAC -1
#define DNA2 -2
#define DNA4 -3

static int asn_getseq(INTSEQFILE *isfp, int rawseqflag)
{
  int size, count, state, oldpe;
  char ch, *t, *seq, *seqstr, *seqend, buf[8];

  program_error(isfp->fp_entryend == NULL, return STATUS_ERROR,
                print_error("   fp_entryend not set by %s's read function\n",
                            file_table[isfp->format].ident));

  if (isfp->fp_seqstart == NULL) {
    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_rawlen = isfp->entry_truelen = 0;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;
      return STATUS_OK;
    }
    else {
      isfp->seqlen = 0;
      isfp->seq[0] = '\0';
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
  }

  seqstr = NULL;
  oldpe = pe_flag;
  pe_flag = PE_NONE;
  count = asn_parse(isfp->fp_seqstart, isfp->fp_entryend,
                    "seq-data", &seqstr, &seqend,
                    NULL);
  pe_flag = oldpe;
  error_test(count == -1, E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry %d:  Invalid format of ASN.1 entry.\n",
                         isfp->filename, isfp->entry_count));

  if (seqstr == NULL) {
    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_rawlen = isfp->entry_truelen = 0;
      isfp->iflag_rawlen = isfp->iflag_truelen = 1;
      return STATUS_OK;
    }
    else {
      isfp->seqlen = 0;
      isfp->seq[0] = '\0';
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
  }

  /*
   * Determine the format that the sequence is encoded in.
   */
  for (t=seqstr+8; t < seqend && isspace(*t); t++) ;
  error_test(t+8 >= seqend, E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry %d:  Premature end of `seq.inst.seq-data' "
                         "sub-record.\n", isfp->filename, isfp->entry_count));

  if (mystreq(t, 'I', "IUPACNA") || mystreq(t, 'I', "IUPACAA"))
    state = IUPAC;
  else if (mystreq(t, 'N', "NCBI2NA"))
    state = DNA2;
  else if (mystreq(t, 'N', "NCBI4NA"))
    state = DNA4;
  else {
    memcpy(buf, t, 7);
    buf[7] = '\0';
    raise_error(E_PARSEERROR, return STATUS_ERROR,
                print_error("%s, entry %d:  Sequence encoding (%7s) is "
                            "not supported.\n", isfp->filename,
                            isfp->entry_count, buf));
  }

  /*
   * Reallocate the sequence buffer, if necesssary.
   */
  if (isfp->entry_seqlen > 0)
    size = isfp->entry_seqlen;
  else {
    size = seqend - seqstr;
    if (state == DNA2)
      size *= 2;
  }
  if (rawseqflag != GETSEQ_LENGTHS && size + 1 >= isfp->seqsize) {
    isfp->seqsize += size + 1;
    isfp->seq = (char *) realloc(isfp->seq, isfp->seqsize);
    if (isfp->seq == NULL) {
      isfp->seqsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
  }

  /*
   * Scan through the characters of the sequence, storing them
   * into the sequence buffer.
   */
  for (t+=8; t < seqend && isspace(*t); t++) ;
  error_test(t + 7 >= seqend, E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry %d:  Premature end of `seq.inst.seq-data' "
                         "sub-record.\n", isfp->filename, isfp->entry_count));
  error_test(*t != '"' && *t != '\'', E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry %d:  Invalid format of `inst.seq-data' "
                         "sub-record.\n", isfp->filename, isfp->entry_count));

  seq = isfp->seq;
  for (ch=*t++; t < seqend && *t != ch; t++) {
    if (isspace(*t))
      continue;

    if (rawseqflag == GETSEQ_LENGTHS) {
      isfp->entry_truelen += (state == DNA2 ? 2 : 1);
      isfp->entry_rawlen += (state == DNA2 ? 2 : 1);
    }
    else {
      if (state == IUPAC)
        *seq++ = *t;
      else if (state == DNA2) {
        switch (*t) {
        case '0':  *seq++ = 'A'; *seq++ = 'A';  break;
        case '1':  *seq++ = 'A'; *seq++ = 'C';  break;
        case '2':  *seq++ = 'A'; *seq++ = 'G';  break;
        case '3':  *seq++ = 'A'; *seq++ = 'T';  break;
        case '4':  *seq++ = 'C'; *seq++ = 'A';  break;
        case '5':  *seq++ = 'C'; *seq++ = 'C';  break;
        case '6':  *seq++ = 'C'; *seq++ = 'G';  break;
        case '7':  *seq++ = 'C'; *seq++ = 'T';  break;
        case '8':  *seq++ = 'G'; *seq++ = 'A';  break;
        case '9':  *seq++ = 'G'; *seq++ = 'C';  break;
        case 'A':  *seq++ = 'G'; *seq++ = 'G';  break;
        case 'B':  *seq++ = 'G'; *seq++ = 'T';  break;
        case 'C':  *seq++ = 'T'; *seq++ = 'A';  break;
        case 'D':  *seq++ = 'T'; *seq++ = 'C';  break;
        case 'E':  *seq++ = 'T'; *seq++ = 'G';  break;
        case 'F':  *seq++ = 'T'; *seq++ = 'T';  break;
        default:
          raise_error(E_PARSEERROR, return STATUS_ERROR,
                      print_error("%s, entry %d:  Invalid character `%c' in "
                                  "sequence's ncbi2na encoding.\n",
                                  isfp->filename, isfp->entry_count, *t));
        }
      }
      else if (state == DNA4) {
        switch (*t) {
        case '1':  *seq++ = 'A';  break;
        case '2':  *seq++ = 'C';  break;
        case '3':  *seq++ = 'M';  break;
        case '4':  *seq++ = 'G';  break;
        case '5':  *seq++ = 'R';  break;
        case '6':  *seq++ = 'S';  break;
        case '7':  *seq++ = 'V';  break;
        case '8':  *seq++ = 'T';  break;
        case '9':  *seq++ = 'W';  break;
        case 'A':  case 'a':  *seq++ = 'Y';  break;
        case 'B':  case 'b':  *seq++ = 'H';  break;
        case 'C':  case 'c':  *seq++ = 'K';  break;
        case 'D':  case 'd':  *seq++ = 'D';  break;
        case 'E':  case 'e':  *seq++ = 'B';  break;
        case 'F':  case 'f':  *seq++ = 'N';  break;
        default:
          raise_error(E_PARSEERROR, return STATUS_ERROR,
                      print_error("%s, entry %d:  Invalid character `%c' in "
                                  "sequence's ncbi4na encoding.\n",
                                  isfp->filename, isfp->entry_count, *t));
        }
      }
      else {
        program_error(1, return STATUS_ERROR,
                      print_error("   Illegal state value %d in asn_getseq.\n",
                                  state));
      }
    }
  }
  error_test(t == seqend, E_PARSEERROR, return STATUS_ERROR,
             print_error("%s, entry %d:  Premature end of "
                         "`seq.inst.seq-data' sub-record.\n", isfp->filename,
                         isfp->entry_count));

  if (rawseqflag == GETSEQ_LENGTHS) {
    if (isfp->entry_seqlen > 0 && state == DNA2 &&
        (isfp->entry_seqlen == isfp->entry_truelen - 1)) {
      isfp->entry_truelen--;
      isfp->entry_rawlen--;
    }
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;
  }
  else {
    isfp->seqlen = seq - isfp->seq;
    isfp->seq[isfp->seqlen] = '\0';

    /*
     * Check if a sequence was found and the length of the sequence.
     */
    if (isfp->seqlen == 0) {
      set_error(E_NOSEQ);
      print_error("%s, entry %d:  Entry contains no sequence.\n",
                  isfp->filename, isfp->entry_count);
      return STATUS_ERROR;
    }
    
    if (isfp->entry_seqlen == 0 || 
        (state == DNA2 && isfp->entry_seqlen == isfp->seqlen - 1))
      isfp->entry_seqlen = isfp->seqlen;
    else if (isfp->entry_seqlen != isfp->seqlen) {
      set_error(E_DIFFLENGTH);
      print_warning("Warning:  %s, entry %d:  Entry gives seq. length of %d, "
                    "but %d characters found.\n", isfp->filename,
                    isfp->entry_count, isfp->entry_seqlen, isfp->seqlen);
      return STATUS_WARNING;
    }

    isfp->entry_truelen = isfp->entry_rawlen = isfp->seqlen;
    isfp->iflag_truelen = isfp->iflag_rawlen = 1;
  }

  return STATUS_OK;
}



typedef struct {
  int braceflag, matchflag, len;
  char *name;
} APSTACK;

static int testmatch(char *s, APSTACK *stack, int size)
{
  int i, j;
  char *t;

  for (i=0; i < size; i++) {
    if (stack[i].braceflag) {
      if (i == 0 || stack[i-1].braceflag) {
        if (*s != '{')
          return 0;
        s++;
      }
    }
    else {
      while (*s && *s == '{') {
        s++;
        if (*s && *s != '.')
          return 0;

        if (*s == '.')
          s++;
      }

      t = stack[i].name;
      for (j=0; j < stack[i].len && *s && *s != '.'; j++,s++,t++)
        if (toupper(*s) != toupper(*t))
          return 0;

      if (j < stack[i].len || (*s && *s != '.'))
        return 0;
    }
    if (*s == '.')
      s++;
  }

  if (*s)
    return 0;
  else
    return 1;
}

int asn_parse(char *begin, char *end, ...)
{
  static struct apstruct {
    int matched;
    char *name, **bptr, **eptr;
  } list[32];
  APSTACK stack[128];
  int i, num, pos, count, numlist;
  char qch, *s, *t, *name;
  va_list ap;

  if (!ctype_initflag)
    init_ctype();

  reset_errors();
  param_error(begin == NULL, return -1, "asn_parse", "arg 1 is NULL");
  param_error(end == NULL, return -1, "asn_parse", "arg 2 is NULL");

  va_start(ap, end);
  numlist = 0;
  while (numlist < 32 && (name = va_arg(ap, char *)) != NULL) {
    list[numlist].name = name;
    list[numlist].bptr = va_arg(ap, char **);
    list[numlist].eptr = va_arg(ap, char **);
    list[numlist].matched = 0;
    numlist++;
  }

  s = begin;
  while (s < end && isspace(*s)) s++;
  if (s == end)
    return 0;

  if (*s == '{')
    s++;

  pos = count = 0;
  while (count < numlist) {
    /*
     * Skip whitespace
     */
    while (s < end && isspace(*s)) s++;
    if (s == end)
      break;

    /*
     * Handle the next token of the string.
     */
    switch (*s) {
    case '{':
      stack[pos].name = "";
      stack[pos].len = 0;
      stack[pos].braceflag = 1;
      stack[pos].matchflag = -1;
      pos++;

      if (pos == 1 || stack[pos-2].braceflag) {
        for (i=0; i < numlist; i++) {
          if (!list[i].matched && testmatch(list[i].name, stack, pos)) {
            stack[pos-1].matchflag = i;
            list[i].matched = 1;
            if (list[i].bptr != NULL)
              *list[i].bptr = s;
            break;
          }
        }
      }

      s++;
      break;

    case ':':
      error_test(s[1] != ':' || s[2] != '=', E_PARSEERROR, return -1,
                 print_error("Error in asn_parse:  Cannot parse given "
                             "ASN.1 text.\n"));
      s += 3;
      break;

    case '}':
      while (pos > 0 && !stack[pos-1].braceflag) {
        if (stack[pos-1].matchflag != -1) {
          num = stack[pos-1].matchflag;
          if (list[num].eptr != NULL)
            *list[num].eptr = s;
          count++;
        }
        pos--;
      }
      if (pos == 0)
        goto AP_LOOP_END;

      pos--;
      s++;
      break;

    case ',':
      while (pos > 0 && !stack[pos-1].braceflag) {
        if (stack[pos-1].matchflag != -1) {
          num = stack[pos-1].matchflag;
          if (list[num].eptr != NULL)
            *list[num].eptr = s;
          count++;
        }
        pos--;
      }
      s++;
      break;

    case '\'':
    case '"':
      qch = *s++;
      while (s < end && (*s != qch || *(s-1) == '\\')) s++; 
      error_test(s == end, E_PARSEERROR, return -1,
                 print_error("Error in asn_parse:  Reached end of text while "
                             "inside quoted string.\n"));
      s++;
      break;

    default:
      for (t=s; s < end && (isalnum(*s) || *s == '-' || *s == '_'); s++) ;
      error_test(t == s, E_PARSEERROR, return -1,
                 print_error("Error in asn_parse:  Cannot parse given "
                             "ASN.1 text.\n"));
      error_test(s == end && pos > 0, E_PARSEERROR, return -1,
                 print_error("Error in asn_parse:  Given ASN.1 text contains "
                             "an incomplete record.\n"));

      stack[pos].name = t;
      stack[pos].len = s - t;
      stack[pos].braceflag = 0;
      stack[pos].matchflag = -1;
      pos++;

      for (i=0; i < numlist; i++) {
        if (!list[i].matched && testmatch(list[i].name, stack, pos)) {
          stack[pos-1].matchflag = i;
          list[i].matched = 1;
          if (list[i].bptr != NULL)
            *list[i].bptr = t;
          break;
        }
      }
    }
  }
AP_LOOP_END:
  while (count < numlist && pos > 0 && !stack[pos-1].braceflag) {
    if (stack[pos-1].matchflag != -1) {
      num = stack[pos-1].matchflag;
      if (list[num].eptr != NULL)
        *list[num].eptr = s;
      count++;
    }
    pos--;
  }
  error_test(count < numlist && pos > 0, E_PARSEERROR, return -1,
             print_error("Error in asn_parse:  Given ASN.1 text contains an "
                         "incomplete record.\n"));

  return count;
}




/*
 * 
 * Internal procedures to containing some of the grunge details
 * of the file reading.
 *
 *    fp_get_line, fp_read_more
 *
 *
 *
 */

/*
 * fp_get_line
 *
 * This function reads the next line of the file (if it exists) and
 * sets the "line_out" and "end_out" values to point to the beginning
 * and end of that next line.  It also increments the fp_current and
 * (possibly) the fp_top fields of the INTSEQFILE structure past this
 * line.
 *
 * This function will also skip the beginning of line pointer past any
 * formfeed characters (since at least one database has the habit of
 * including formfeeds in its sequence files).
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *              line_out  -  the location where the put the pointer to the
 *                           beginning of the line
 *              end_out   -  the location where the put the pointer to the
 *                           end of the line (i.e. the newline character)
 *
 * Return: a STATUS value
 */
static int fp_get_line(INTSEQFILE *isfp, char **line_out, char **end_out)
{
  register char *s, *top;
  int status;
  char *stemp, *line;

  line = s = isfp->fp_current;
  top = isfp->fp_top;
  while (1) {
    /*
     * Look for the next newline.
     */
    if (isfp->isendtagged)
      while (*s != '\n') s++;
    else {
      while (s + 10 <= top && *s != '\n' && *++s != '\n' && *++s != '\n' &&
             *++s != '\n' && *++s != '\n' && *++s != '\n' && *++s != '\n' &&
             *++s != '\n' && *++s != '\n' && *++s != '\n')
        s++;
      while (s < top && *s != '\n') s++;
    }

    /*
     * If the newline is an actual character in the file (and not
     * just the terminator signalling the end of the characters 
     * read so far), check for formfeed characters at the "beginning"
     * of the line, and set the pointers to the line.
     *
     * Otherwise, read more of the file and go back through the loop.
     */
    if (s < top) {
      while (*line == '\f') line++;

      *line_out = line;
      *end_out = s;
      isfp->fp_current = s+1;

      return STATUS_OK;
    }
    else {
      stemp = s;
      status = fp_read_more(isfp, &line, &stemp, &isfp->fp_top);
      s = stemp;
      top = isfp->fp_top;
      switch (status) {
      case STATUS_OK:     
        continue;

      case STATUS_EOF:
        while (line < s && *line == '\f') line++;

        /*
         * If the line is empty, return EOF.  Else, return the line.
         */
        if (s == line)
          return STATUS_EOF;

        *line_out = line;
        *end_out = s;
        isfp->fp_current = s;

        return STATUS_OK;

      case STATUS_ERROR:
      case STATUS_FATAL:
        isfp->fp_current = s;
        return status;

      default:
        status_error(return STATUS_ERROR, "fp_get_line");
      }
    }
  }
}


/*
 * fp_read_more
 *
 * This procedure reads in another block of the file (if it exists).
 * It performs one of three actions based on the situation.  First, if
 * there is sufficient room at the end of the "fp_buffer", then the 
 * new block is read there.  Second, if the beginning of the current
 * entry (the text of previous entries are forgotten) occurs above
 * the halfway point of the buffer, then the current entry's text is
 * shifted to the beginning of the buffer and the new block is read
 * into the end of the buffer.  Third, if the current entry's block 
 * is over half the size of the buffer, the buffer is dynamically
 * reallocated at double its size and then the new block is read into
 * the new region.
 *
 * The last three parameters are pointers into the buffer, where the
 * first two can point anywhere in the current entry and the third must
 * point to the top of the text in the buffer (i.e., it must either be
 * the "fp_top" field or a local copy of that field).  Usually, the first
 * two pointers are the beginning of the current line and the current point 
 * in the text scan (hence the names "line" and "s"), but this is not
 * required.
 *
 * The reason those pointers must be passed in is that the text may
 * be shifted in the buffer or the buffer itself might be reallocated.
 * In that event, those three pointers and the INTSEQFILE fields
 * "fp_entrystart", "fp_seqstart" and "fp_entryend" are reset appropriately.
 * NOTE:  fp_current and fp_top are NOT reset, so they must either be
 *        passed in as the parameters, or updated from the variables
 *        passed in as the parameters.
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *              line_out  -  the location of a pointer into the "fp_buffer"
 *              s_out     -  the location of a pointer into the "fp_buffer"
 *              top_out   -  the location of a pointer which points to
 *                           the top of the valid text in "fp_buffer".
 *
 * Return: a STATUS value
 */
static int fp_read_more(INTSEQFILE *isfp, char **line_out, char **s_out,
                        char **top_out)
{
  int size, shift;
  char *top, *bottom, *buffer_top, *oldbuffer;

  /*
   * Functions like seqfgcgify, seqfungcgify and others must use the
   * read operations to parse the entry they're given.  They do that
   * by setting up a dummy INTSEQFILE structure with the entry as the
   * file buffer.  They also set the filename field to the empty string
   * to signal that nothing more should be read when the read operations
   * reach the end of the entry string (since no file or pipe is open).
   * In the normal reading process, filename is always an non-empty string.
   *
   * Thus, this test to return an EOF signal when the filename is an
   * empty string.
   */
  if (isfp->filename[0] == '\0')
    return STATUS_EOF;

  /*
   * Set local copies of the bottom and top of the section of the file
   * which must be kept in memory (such as the piece of the current
   * entry read in so far), along with the top of the buffer.
   */
  top = *top_out;
  bottom = (isfp->fp_entrystart != NULL ? isfp->fp_entrystart : *line_out);
  buffer_top = isfp->fp_buffer + isfp->fp_bufsize;


  /*
   * If the file is mapped, then redo the map for the next chunk of
   * the file.  If the new mapping fails, then try to read it 
   * normally.
   */
#ifdef ISMAPABLE
  if (isfp->ismapped) {
    int newmapsize, newfilepos, offset, status;
    caddr_t addr;

    if (isfp->filepos + isfp->mapsize == isfp->filesize)
      return STATUS_EOF;

    if (top - bottom >= isfp->mapsize / 2) {
      newmapsize = isfp->mapsize * 2;
      newfilepos = isfp->filepos;
      if (newfilepos + newmapsize > isfp->filesize)
        newmapsize = isfp->filesize - newfilepos;

      offset = bottom - isfp->fp_buffer;
    }
    else {
      newfilepos = isfp->filepos + isfp->mapsize;
      newfilepos -= top - bottom;
      offset = newfilepos % MYPAGESIZE;
      newfilepos -= offset;

      newmapsize = isfp->filesize - newfilepos;
      if (newmapsize > MAXMAPSIZE)
        newmapsize = MAXMAPSIZE;
    }

    munmap(isfp->fp_buffer, isfp->mapsize);
    addr = mmap(0, newmapsize, PROT_READ, MAP_SHARED, isfp->input_fd,
                newfilepos);

    /*
     * If the new mapping works, recalculate all of the offsets, and
     * store the new map's information in "isfp" and return.
     *
     * If the new mapping fails, then turn "ismapped" off, malloc an
     * array for fp_buffer, set the file pointer to the bottommost byte
     * and read the next section.
     */
    if (addr != (caddr_t) -1) {
      *line_out = addr + offset + (*line_out - bottom);
      *s_out = addr + offset + (*s_out - bottom);

      if (isfp->fp_entrystart)
        isfp->fp_entrystart = addr + offset + (isfp->fp_entrystart - bottom);
      if (isfp->fp_seqstart)
        isfp->fp_seqstart = addr + offset + (isfp->fp_seqstart - bottom);
      if (isfp->fp_entryend)
        isfp->fp_entryend = addr + offset + (isfp->fp_seqstart - bottom);

      isfp->fp_buffer = addr;
      isfp->fp_bufsize = newmapsize;
      *top_out = isfp->fp_buffer + newmapsize;
      isfp->filepos = newfilepos;
      isfp->fp_bytepos = newfilepos;

      return STATUS_OK;
    }
    else {
      isfp->ismapped = 0;

      isfp->fp_bufsize = top - bottom + INIT_BUFSIZE;
      isfp->fp_buffer = (char *) malloc(isfp->fp_bufsize);
      if (isfp->fp_buffer == NULL) {
        isfp->fp_bufsize = 0;
        memory_error(1, return STATUS_FATAL);
      }

      status = seek_raw_file(isfp->input_fd, newfilepos + offset);
      error_test(status != STATUS_OK, E_READFAILED, return STATUS_ERROR,
                 print_error("%s:  %s\n", isfp->filename,
                             sys_errlist[errno]));

      size = read_raw_file(isfp->input_fd, isfp->fp_buffer,
                           isfp->fp_bufsize - 1);
      error_test(size <= top - bottom, E_READFAILED, return STATUS_ERROR,
                 print_error("%s:  %s\n", isfp->filename,
                             sys_errlist[errno]));

      isfp->fp_bytepos = newfilepos + offset;

      *line_out = isfp->fp_buffer + (*line_out - bottom);
      *s_out = isfp->fp_buffer + (*s_out - bottom);

      if (isfp->fp_entrystart)
        isfp->fp_entrystart = isfp->fp_buffer + (isfp->fp_entrystart - bottom);
      if (isfp->fp_seqstart)
        isfp->fp_seqstart = isfp->fp_buffer + (isfp->fp_seqstart - bottom);
      if (isfp->fp_entryend)
        isfp->fp_entryend = isfp->fp_buffer + (isfp->fp_seqstart - bottom);

      *top_out = isfp->fp_buffer + size;
      **top_out = '\n';
      isfp->isendtagged = 1;

      return STATUS_OK;
    }
  }
#endif


  /*
   * If there is enough room left in the buffer, skip to the reading
   * part.
   */
  if (buffer_top - top > CONCAT_READ_POINT)
    ;

  /*
   * If the bottom is over the halfway point, then shift the current
   * contents without reallocating.
   */
  else if (bottom - isfp->fp_buffer > isfp->fp_bufsize / 2) {
    memcpy(isfp->fp_buffer, bottom, top - bottom);
    
    shift = bottom - isfp->fp_buffer;
    isfp->fp_bytepos += shift;
    top -= shift;
    *s_out -= shift;
    *line_out -= shift;

    if (isfp->fp_entrystart)
      isfp->fp_entrystart -= shift;
    if (isfp->fp_seqstart)
      isfp->fp_seqstart -= shift;
    if (isfp->fp_entryend)
      isfp->fp_entryend -= shift;
  }

  /*
   * Otherwise, double the size of the buffer.
   */
  else {
    oldbuffer = isfp->fp_buffer;

    isfp->fp_bufsize += isfp->fp_bufsize;
    isfp->fp_buffer = (char *) realloc(isfp->fp_buffer, isfp->fp_bufsize);
    if (isfp->fp_buffer == NULL) {
      isfp->fp_bufsize = 0;
      memory_error(1, return STATUS_FATAL);
    }
    buffer_top = isfp->fp_buffer + isfp->fp_bufsize;

    top = &isfp->fp_buffer[top - oldbuffer];
    *s_out = &isfp->fp_buffer[*s_out - oldbuffer];
    *line_out = &isfp->fp_buffer[*line_out - oldbuffer];
    
    if (isfp->fp_entrystart)
      isfp->fp_entrystart = &isfp->fp_buffer[isfp->fp_entrystart - oldbuffer];
    if (isfp->fp_seqstart)
      isfp->fp_seqstart = &isfp->fp_buffer[isfp->fp_seqstart - oldbuffer];
    if (isfp->fp_entryend)
      isfp->fp_entryend = &isfp->fp_buffer[isfp->fp_seqstart - oldbuffer];
  }

  /*
   * Read in the next chunk of the file.
   */
  size = read_raw_file(isfp->input_fd, top, buffer_top - top - 1);
  if (size > 0) {
    *top_out = top + size;
    **top_out = '\n';
    isfp->isendtagged = 1;

    return STATUS_OK;
  }
  else if (size == 0) {
    *top_out = top;
    **top_out = '\n';
    isfp->isendtagged = 1;

    return STATUS_EOF;
  }
  else {
    raise_error(E_READFAILED, return STATUS_ERROR,
                print_error("%s:  %s\n", isfp->filename, sys_errlist[errno]));
  }
}


/*
 * fp_read_all
 *
 * Read the complete file into memory, using repeated calls to fp_read_more.
 *
 * Parameters:  isfp  -  an opened INTSEQFILE structure
 *
 * Return: a STATUS value
 */
static int fp_read_all(INTSEQFILE *isfp)
{
  int status;
  char *line, *s, *top;

  line = isfp->fp_current;
  top = isfp->fp_top;
  status = STATUS_OK;
  while (status == STATUS_OK) {
    s = top;
    status = fp_read_more(isfp, &line, &s, &top);
    switch (status) {
    case STATUS_OK:     break;
    case STATUS_EOF:    break;
    case STATUS_ERROR:  return STATUS_ERROR;
    case STATUS_FATAL:  return STATUS_FATAL;
    default:            status_error(return STATUS_ERROR, "fp_read_all");
    }
  }
  isfp->fp_current = line;
  isfp->fp_top = top;
  return (isfp->fp_current < isfp->fp_top ? STATUS_OK : STATUS_EOF);
}  





/*
 *
 *
 * Get Information Section
 *
 *
 *
 */

typedef struct {
  SEQINFO *info;
  int count, size, error;
} INFO;

static int is_ncbiprefix(char *);
static char *parse_ncbi_idlist(INFO *, char *, char *);


static void start_info(INFO *info, INTSEQFILE *isfp, int flag)
{
  info->info = isfp->info;
  info->size = isfp->infobufsize;
  info->count = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO
                    ? sizeof(SEQINFO) : isfp->infosize);
  info->error = 0;

  if (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO)
    memset(info->info, 0, sizeof(SEQINFO));
  else {
    if (flag == SEQINFO_DATE)  info->info->date = NULL;
    if (flag == SEQINFO_IDLIST)  info->info->idlist = NULL;
    if (flag == SEQINFO_DESCRIPTION)  info->info->description = NULL;
    if (flag == SEQINFO_COMMENT)  info->info->comment = NULL;
    if (flag == SEQINFO_ORGANISM)  info->info->organism = NULL;
    if (flag == SEQINFO_FRAGMENT)  info->info->isfragment = 0;
    if (flag == SEQINFO_CIRCULAR)  info->info->iscircular = 0;
    if (flag == SEQINFO_ALPHABET)  info->info->alphabet = UNKNOWN;
    if (flag == SEQINFO_STARTPOS)  info->info->fragstart = UNKNOWN;
  }
}

static void finish_info(INFO *info, INTSEQFILE *isfp)
{
  if (!info->error) {
    isfp->info = info->info;
    isfp->infosize = info->count;
    isfp->infobufsize = info->size;
  }
  else {
    isfp->info = NULL;
    isfp->infosize = isfp->infobufsize = 0;
    isfp->istatus = INFO_NONE;
    isfp->iflag_date = isfp->iflag_idlist = isfp->iflag_description = 0;
    isfp->iflag_comment = isfp->iflag_organism = isfp->iflag_fragment = 0;
    isfp->iflag_circular = isfp->iflag_alphabet = isfp->iflag_fragstart = 0;
  }
}


static int setup_field(INFO *info, int field, int len)
{
  int date_offset, list_offset, desc_offset, comm_offset, org_offset;
  int his_offset, fname_offset, db_offset, fmt_offset;
  char ch, *t, *a, *b, *ptr, **fieldptr, *istruct, *itop;

  if (info->error || len <= 0)
    return 0;

  istruct = (char *) info->info;
  itop = istruct + info->count;
  switch (field) {
  case SEQINFO_FILENAME:     fieldptr = &info->info->filename;  break;
  case SEQINFO_DBNAME:       fieldptr = &info->info->dbname;  break;
  case SEQINFO_FORMAT:       fieldptr = &info->info->format;  break;
  case SEQINFO_IDLIST:       fieldptr = &info->info->idlist;  break;
  case SEQINFO_DATE:         fieldptr = &info->info->date;  break;
  case SEQINFO_DESCRIPTION:  fieldptr = &info->info->description;  break;
  case SEQINFO_COMMENT:      fieldptr = &info->info->comment;  break;
  case SEQINFO_ORGANISM:     fieldptr = &info->info->organism;  break;
  case SEQINFO_HISTORY:      fieldptr = &info->info->history;  break;
  default:
    return 0;
  }
  ptr = *fieldptr;

  if (ptr != NULL && ptr >= istruct && ptr < itop) {
    for (t=ptr; *t; t++) ;

    /*
     * If the field exists and is in the middle of the strings,
     * move it to the end using the three reversals trick, i.e.,
     * from the string AB, the string rev(rev(A),rev(B)) is BA.
     */
    if (t+1 < itop) {
      for (a=ptr,b=t; a < b; a++,b--) {
        ch = *a;
        *a = *b;
        *b = ch;
      }
      for (a=t+1,b=itop-1; a < b; a++,b--) {
        ch = *a;
        *a = *b;
        *b = ch;
      }
      for (a=ptr,b=itop-1; a < b; a++,b--) {
        ch = *a;
        *a = *b;
        *b = ch;
      }

      if (info->info->filename > ptr && info->info->filename < itop)
        info->info->filename -= t - ptr + 1;
      if (info->info->dbname > ptr && info->info->dbname < itop)
        info->info->dbname -= t - ptr + 1;
      if (info->info->format > ptr && info->info->format < itop)
        info->info->format -= t - ptr + 1;
      if (info->info->date > ptr && info->info->date < itop)
        info->info->date -= t - ptr + 1;
      if (info->info->idlist > ptr && info->info->idlist < itop)
        info->info->idlist -= t - ptr + 1;
      if (info->info->description > ptr && info->info->description < itop)
        info->info->description -= t - ptr + 1;
      if (info->info->comment > ptr && info->info->comment < itop)
        info->info->comment -= t - ptr + 1;
      if (info->info->organism > ptr && info->info->organism < itop)
        info->info->organism -= t - ptr + 1;
      if (info->info->history > ptr && info->info->history < itop)
        info->info->history -= t - ptr + 1;
      *fieldptr = itop - (t - ptr + 1);
    }
  }

  /*
   * Now realloc the information if there's not enough space.
   */
  if (info->count + len + 1 >= info->size) {
    date_offset = list_offset = desc_offset = -1;
    comm_offset = org_offset = his_offset = -1;
    fname_offset = db_offset = fmt_offset = -1;

    if (info->info->filename >= istruct && info->info->filename < itop)
      fname_offset = info->info->filename - istruct;
    if (info->info->dbname >= istruct && info->info->dbname < itop)
      db_offset = info->info->dbname - istruct;
    if (info->info->format >= istruct && info->info->format < itop)
      fmt_offset = info->info->format - istruct;
    if (info->info->date >= istruct && info->info->date < itop)
      date_offset = info->info->date - istruct;
    if (info->info->idlist >= istruct && info->info->idlist < itop)
      list_offset = info->info->idlist - istruct;
    if (info->info->description >= istruct && info->info->description < itop)
      desc_offset = info->info->description - istruct;
    if (info->info->comment >= istruct && info->info->comment < itop)
      comm_offset = info->info->comment - istruct;
    if (info->info->organism >= istruct && info->info->organism < itop)
      org_offset = info->info->organism - istruct;
    if (info->info->history >= istruct && info->info->history < itop)
      his_offset = info->info->history - istruct;

    info->size += info->size + len + 1;
    info->info = (SEQINFO *) realloc(info->info, info->size);
    if (info->info == NULL) {
      info->error = 1;
      return 0;
    }
    else {
      istruct = (char *) info->info;
      if (fname_offset != -1) info->info->filename = istruct + fname_offset;
      if (db_offset != -1)    info->info->dbname = istruct + db_offset;
      if (fmt_offset != -1)   info->info->format = istruct + fmt_offset;
      if (date_offset != -1)  info->info->date = istruct + date_offset;
      if (list_offset != -1)  info->info->idlist = istruct + list_offset;
      if (desc_offset != -1)  info->info->description = istruct + desc_offset;
      if (comm_offset != -1)  info->info->comment = istruct + comm_offset;
      if (org_offset != -1)   info->info->organism = istruct + org_offset;
      if (his_offset != -1)   info->info->history = istruct + his_offset;
    }
  }

  return 1;
}


static int is_idprefix(char *s)
{
  return (isalnum(s[0]) && isalnum(s[1]) &&
          (s[2] == ':' || 
           (isalnum(s[2]) && (s[3] == ':' || 
                              (isalnum(s[3]) && s[4] == ':')))));
}


static void add_id(INFO *info, char *type, char *s, char *end)
{
  int len;
  char *t, *s2, *t2, buffer[72];

  if (info->error)
    return;

  while (s < end && isspace(*s)) s++;
  while (s < end && isspace(*(end-1))) end--;
  if (s < end && (*s == '\'' || *s == '"') &&
      (*(end-1) == '\'' || *(end-1) == '"')) {
    s++;
    end--;
    while (s < end && isspace(*s)) s++;
    while (s < end && isspace(*(end-1))) end--;
  }
  if (s >= end)
    return;

  if ((end - s == 7 && mystreq(s, '(', "(BELOW)")) ||
      (end - s == 7 && mystreq(s, 'U', "UNKNOWN")) ||
      (end - s == 6 && mystreq(s, 'U', "UNKNWN")))
    return;

  if (is_ncbiprefix(s) && parse_ncbi_idlist(info, s, end) != s)
    return;

  s2 = buffer;
  len = 0;
  if (is_idprefix(s)) {
    for (t2=s; *t2 != ':'; s2++,t2++,len++)
      *s2 = tolower(*t2);
    for ( ; t2 < end; s2++,t2++,len++)
      *s2 = *t2;
  }
  else {
    for (t2=type; *t2; s2++,t2++,len++)
      *s2 = tolower(*t2);
    *s2++ = ':';
    len++;
    for (t2=s; t2 < end; s2++,t2++,len++)
      *s2 = *t2;
  }
  *s2 = '\0';

  if (!setup_field(info, SEQINFO_IDLIST, s2 - buffer + 2))
    return;

  if (info->info->idlist == NULL)
    t = info->info->idlist = ((char *) info->info) + info->count;
  else {
    for (s2=info->info->idlist; *s2; ) {
      for (t2=buffer; *t2 && *s2 && *s2 != '|'; s2++,t2++)
        if (toupper(*t2) != toupper(*s2))
          break;

      if (!*t2 && (!*s2 || *s2 == '|'))
        return;
      else {
        while (*s2 && *s2 != '|') s2++;
        if (*s2) s2++;
      }
    }
    
    t = ((char *) info->info) + info->count - 1;
    *t++ = '|';
  }

  strcpy(t, buffer);
  info->count += len + 1;
}

static void set_filename(INFO *info, char *s)
{
  char *t, *end;

  if (info->error || s == NULL)
    return;

  for (end=s; *end; end++) 
    ;

  if (!setup_field(info, SEQINFO_FILENAME, end - s + 1))
    return;

  if (info->info->filename == NULL)
    t = info->info->filename = ((char *) info->info) + info->count;
  else
    t = info->info->filename;

  while (s < end)
    *t++ = *s++;
  *t = '\0';
  info->count = t+1 - ((char *) info->info);
}

static void set_dbname(INFO *info, char *s)
{
  char *t, *end;

  if (info->error || s == NULL)
    return;

  for (end=s; *end; end++) 
    ;

  if (!setup_field(info, SEQINFO_DBNAME, end - s + 1))
    return;

  if (info->info->dbname == NULL)
    t = info->info->dbname = ((char *) info->info) + info->count;
  else
    t = info->info->dbname;

  while (s < end)
    *t++ = *s++;
  *t = '\0';
  info->count = t+1 - ((char *) info->info);
}

static void set_format(INFO *info, char *s)
{
  char *t, *end;

  if (info->error || s == NULL)
    return;

  for (end=s; *end; end++) 
    ;

  if (!setup_field(info, SEQINFO_FORMAT, end - s + 1))
    return;

  if (info->info->format == NULL)
    t = info->info->format = ((char *) info->info) + info->count;
  else
    t = info->info->format;

  while (s < end)
    *t++ = *s++;
  *t = '\0';
  info->count = t+1 - ((char *) info->info);
}

static void set_date(INFO *info, char *s, char *end)
{
  char *t;

  if (info->error)
    return;

  if (end == NULL) 
    for (end=s; *end; end++) 
      ;

  while (s < end && isspace(*s)) s++;
  while (s < end && isspace(*(end-1))) end--;
  if (s < end && (*s == '\'' || *s == '"') &&
      (*(end-1) == '\'' || *(end-1) == '"')) {
    s++;
    end--;
    while (s < end && isspace(*s)) s++;
    while (s < end && isspace(*(end-1))) end--;
  }
  if (s >= end || (end - s == 11 && mystreq(s, '0', "01-JAN-0000")))
    return;

  if (!setup_field(info, SEQINFO_DATE, end - s + 1))
    return;

  if (info->info->date == NULL)
    t = info->info->date = ((char *) info->info) + info->count;
  else
    t = info->info->date;

  while (s < end)
    *t++ = *s++;
  *t = '\0';
  info->count = t+1 - ((char *) info->info);
}

static void add_description(INFO *info, char *s, char *end)
{
  int count;
  char *t, *s2;

  if (info->error)
    return;

  if (end == NULL) 
    for (end=s; *end; end++) 
      ;

  while (s < end && (isspace(*s) || *s == '>' || *s == ';')) s++;
  while (s < end && isspace(*(end-1))) end--;
  if (s < end && (*s == '\'' || *s == '"') &&
      (*(end-1) == '\'' || *(end-1) == '"')) {
    s++;
    end--;
    while (s < end && isspace(*s)) s++;
    while (s < end && isspace(*(end-1))) end--;
  }
  if (s < end && (*(end-1) == '.' || *(end-1) == ';')) end--;
  if (s >= end)
    return;

  for (count=1,s2=s; s2 < end; s2++)
    if (*s2 == '\t')
      count += 2;

  if (!setup_field(info, SEQINFO_DESCRIPTION, end - s + count))
    return;

  if (info->info->description == NULL)
    t = info->info->description = ((char *) info->info) + info->count;
  else {
    t = ((char *) info->info) + info->count - 1;
    *t++ = ' ';
  }

  while (s < end) {
    if (*s == '\n') {
      *t++ = ' ';
      while (s < end && isspace(*s)) s++;
    }
    else if (*s == '\t') {
      *t++ = ' ';
      *t++ = ' ';
      *t++ = ' ';
      s++;
    }
    else
      *t++ = *s++;
  }
  *t = '\0';
  info->count = t+1 - ((char *) info->info);
}

static void add_organism(INFO *info, char *s, char *end)
{
  int count;
  char *t, *s2;

  if (info->error)
    return;

  if (end == NULL) 
    for (end=s; *end; end++) 
      ;

  while (s < end && (isspace(*s) || *s == '>' || *s == ';')) s++;
  while (s < end && isspace(*(end-1))) end--;
  if (s < end && (*s == '\'' || *s == '"') &&
      (*(end-1) == '\'' || *(end-1) == '"')) {
    s++;
    end--;
    while (s < end && isspace(*s)) s++;
    while (s < end && isspace(*(end-1))) end--;
  }
  if (s < end && *(end-1) == ';') end--;
  if (s >= end)
    return;

  for (count=1,s2=s; s2 < end; s2++)
    if (*s2 == '\t')
      count += 2;

  if (!setup_field(info, SEQINFO_ORGANISM, end - s + count))
    return;

  if (info->info->organism == NULL)
    t = info->info->organism = ((char *) info->info) + info->count;
  else {
    t = ((char *) info->info) + info->count - 1;
    *t++ = ' ';
  }

  while (s < end) {
    if (*s == '\n') {
      *t++ = ' ';
      while (s < end && isspace(*s)) s++;
    }
    else if (*s == '\t') {
      *t++ = ' ';
      *t++ = ' ';
      *t++ = ' ';
      s++;
    }
    else
      *t++ = *s++;
  }
  *t = '\0';
  info->count = t+1 - ((char *) info->info);
}

static void add_comment(INFO *info, char *s, char *end, int stripflag,
                        int stripsize)
{
  int i;
  char *t;

  if (info->error)
    return;

  if (end == NULL)
    for (end=s; *end; end++)
      ;

  if (stripflag) {
    while (s < end && ((isspace(*s) && stripsize) || *s == '>' || *s == ';'))
      s++;
    while (s < end && isspace(*(end-1))) end--;
    if (s < end && (*s == '\'' || *s == '"') &&
        (*(end-1) == '\'' || *(end-1) == '"')) {
      s++;
      end--;
      while (s < end && isspace(*s)) s++;
      while (s < end && isspace(*(end-1))) end--;
    }
  }
  if (s > end)
    return;

  if (!setup_field(info, SEQINFO_COMMENT, end - s + 2))
    return;

  if (info->info->comment == NULL)
    t = info->info->comment = ((char *) info->info) + info->count;
  else
    t = ((char *) info->info) + info->count - 1;

  while (s < end) {
    if (*s == '\n') {
      s++;
      if (stripsize == -1)
        while (s < end && isspace(*s) && *s != '\n') s++;
      else if (stripsize > 0)
        for (i=0; s < end && *s != '\n' && i < stripsize; i++) s++;

      /*
       * When you find a continuation line (two spaces and then a non-space),
       * merge it with previous line.
       */
      if (stripsize > 0 && s + 3 < end &&
          s[0] == ' ' && s[1] == ' ' && s[2] != ' ') {
        s += 2;
        *t++ = ' ';
      }
      else if (s < end)
        *t++ = '\n';
    }
    else
      *t++ = *s++;
  }
  *t++ = '\n';
  *t = '\0';
  info->count = t+1 - ((char *) info->info);
}

static void add_history(INFO *info, char *s, char *end, int stripsize)
{
  int i;
  char *t;

  if (info->error)
    return;

  if (end == NULL)
    for (end=s; *end; end++)
      ;

  while (s < end && ((isspace(*s) && stripsize) || *s == '>' || *s == ';'))
    s++;
  while (s < end && isspace(*(end-1))) end--;
  if (s < end && (*s == '\'' || *s == '"') &&
      (*(end-1) == '\'' || *(end-1) == '"')) {
    s++;
    end--;
    while (s < end && isspace(*s)) s++;
    while (s < end && isspace(*(end-1))) end--;
  }
  if (s < end && (*(end-1) == '.' || *(end-1) == ';')) end--;
  if (s > end)
    return;

  if (!setup_field(info, SEQINFO_HISTORY, end - s + 2))
    return;

  if (info->info->history == NULL)
    t = info->info->history = ((char *) info->info) + info->count;
  else
    t = ((char *) info->info) + info->count - 1;

  while (s < end) {
    if (*s == '\n') {
      s++;
      if (stripsize == -1)
        while (s < end && isspace(*s) && *s != '\n') s++;
      else if (stripsize > 0)
        for (i=0; s < end && *s != '\n' && i < stripsize; i++) s++;

      /*
       * When you find a continuation line (two spaces and then a non-space),
       * merge it with previous line.
       */
      if (s + 3 < end && s[0] == ' ' && s[1] == ' ' && s[2] != ' ') {
        s += 2;
        *t++ = ' ';
      }
      else
        *t++ = '\n';
    }
    else
      *t++ = *s++;
  }
  *t++ = '\n';
  *t = '\0';
  info->count = t+1 - ((char *) info->info);
}


static void add_retrieval(INFO *info, int type, char *string)
{
  char buffer[128];
  if (type == 0)
    sprintf(buffer, "SEQIO retrieval from plain file.   %s", get_today());
  else if (type == 1)
    sprintf(buffer, "SEQIO retrieval from %s-format entry.   %s",
            string, get_today());
  else if (type == 2)
    sprintf(buffer, "SEQIO retrieval from %s database entry.   %s",
            string, get_today());
  else if (type == 3)
    sprintf(buffer, "SEQIO retrieval from %s output.   %s",
            string, get_today());
  else
    return;

  add_history(info, buffer, NULL, 0);
}


/*
 * Valid alphabet names:
 *    1) A prefix of "ds-", "ss-" or "ms-" may precede the name.
 *    2) If it ends with "DNA", it's DNA.
 *    3) If it ends with "RNA", it's RNA.
 *    4) "PRT", "AA", "Amino...", "Protein...", "Peptide..." are Protein
 *       (where "..." means that extra text can follow the string).
 */
static int get_alphabet(char *s, char *end)
{
  char ch;

  if (end == NULL)
    for (end=s; *end; end++)
      ;

  while (s < end && isspace(*s)) s++;
  while (end > s && (isspace(*(end-1)) || *(end-1) == ',')) end--;

  if (end - s > 3 && s[2] == '-' && toupper(s[1]) == 'S' &&
      ((ch = toupper(s[0])) == 'D' || ch == 'S' || ch == 'M'))
    s += 3;

  if (end - s < 2)
    return UNKNOWN;

  if (end - s >= 2 && toupper(*(end-2)) == 'N' && toupper(*(end-1)) == 'A') {
    if (end - s == 2)
      return DNA;

    ch = toupper(*(end-3));
    if (ch == 'D')
      return DNA;
    else if (ch == 'R')
      return RNA;
  }

  ch = toupper(s[0]);
  if (ch == 'A') {
    if ((end - s == 2 && toupper(s[1]) == 'A') ||
        (end - s >= 5 && mystreq(s, 'A', "AMINO")))
      return PROTEIN;
  }
  else if (ch == 'P') {
    if ((end - s == 3 && mystreq(s+1, 'R', "RT")) ||
        (end - s >= 7 && mystreq(s+1, 'R', "ROTEIN")) ||
        (end - s >= 7 && mystreq(s+1, 'E', "EPTIDE")))
      return PROTEIN;
  }

  return UNKNOWN;
}

static void set_alphabet(INFO *info, int alpha)
{
  if (!info->error)
    info->info->alphabet = alpha;
}

static void set_fragment(INFO *info, int isfrag)
{
  if (!info->error)
    info->info->isfragment = isfrag;
}

static void set_circular(INFO *info, int iscirc)
{
  if (!info->error)
    info->info->iscircular = iscirc;
}

static void set_fragstart(INFO *info, int pos)
{
  if (!info->error)
    info->info->fragstart = pos;
}


static void parse_comment(INFO *info, char *start, char *end, int stripflag,
                          int stripsize, int flag)
{
  int i, allflag;
  char *s, *t, *line, *comend, *first_seqio, *init_seqio;

  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  /*
   * Find the end of the comments and the beginning of the "SEQIO" lines.
   * They may be separated by a blank line.
   */
  first_seqio = init_seqio = comend = NULL;
  line = start;
  for (s=start; s < end; ) {
    if (mystreq(s, 'S', "SEQIO") || 
        (first_seqio != NULL && *s == ' ' && s[1] == ' ' && s[2] != ' ')) {
      if (!first_seqio) {
        first_seqio = line;
        init_seqio = s;
      }
    }
    else if (*s == '\n') {
      comend = line;
      first_seqio = init_seqio = NULL;
    }
    else
      first_seqio = init_seqio = comend = NULL;

    while (s < end && *s != '\n') s++;
    if (s+1 >= end)
      break;
    line = ++s;

    if (stripsize) {
      if (stripsize == -1)
        while (s < end && isspace(*s) && *s != '\n') s++;
      else
        for (i=0; s < end && *s != '\n' && i < stripsize; i++) s++;
    }
  }

  if (!comend)
    comend = (first_seqio ? first_seqio : end);
  if (comend > start && (flag == SEQINFO_ALL || flag == SEQINFO_COMMENT)) {
    add_comment(info, start, comend, stripflag, stripsize);
    if (flag == SEQINFO_COMMENT)
      return;
  }

  if (first_seqio && (allflag || flag == SEQINFO_IDLIST)) {
    while (init_seqio < end && mystreq(init_seqio, 'S', "SEQIO REFS: ")) {
      for (s=init_seqio+11; s < end && isspace(*s) && *s != '\n'; s++) ;
      if (is_idprefix(s)) {
        while (s < end && !isspace(*s)) {
          for (t=s; s < end && !isspace(*s) && *s != '|' && *s != ','; s++) ;
          add_id(info, "", t, s);
          if (s < end && !isspace(*s)) s++;
        }

        while (s < end && *s != '\n') s++;

        if (s == end)
          init_seqio = end;
        else {
          s++;
          if (stripsize) {
            if (stripsize == -1)
              while (s < end && isspace(*s) && *s != '\n') s++;
            else
              for (i=0; s < end && *s != '\n' && i < stripsize; i++) s++;
          }
          init_seqio = s;
        }
      }
    }
 
    if (init_seqio < end && flag == SEQINFO_ALL)
      add_history(info, init_seqio, end, stripsize);
  }
}


static void parse_oneline(INFO *info, char *start, char *end, int info_flag)
{
  int i, flag, alphabet, tseqflag, num1, num2, allflag;
  char *s, *t, *alphastart, *alphaend;

  if (end == NULL)
    for (end=start; *end; end++) ;

  while (start < end && isspace(*start)) start++;
  while (end > start && isspace(*(end-1))) end--;
  if (start == end)
    return;

  allflag = (info_flag == SEQINFO_ALL || info_flag == SEQINFO_ALLINFO);

  /*
   * First, find and parse the length section, if it exists.
   * The length section is detected by looking for the following things
   * occurring at the end of the string: 1) a comma, 2) the digits of
   * the length, 3) one of "aa", "bp" or "ch", 4) optionally a "(...)"
   * string and 5) optionally a ".".  If any of 1, 2 or 3 is missing,
   * the text at the end of the string is considered part of the 
   * organism string (or possibly the description string.
   */
  alphastart = NULL;
  alphabet = 0;
  s = end - 1;

  if (*s == '.')
    for (s--; s >= start && isspace(*s); s--) ;

  if (s >= start && *s == ')') {
    for (s--; s >= start && *s != '('; s--) ;
    if (s >= start) {
      alphastart = s--;
      while (s >= start && isspace(*s)) s--;
    }
  }

  flag = 1;
  if (s - 1 >= start) {
    s--;
    if (((toupper(*s) == 'A' && toupper(s[1]) == 'A' && (i = 1)) ||
         (toupper(*s) == 'B' && toupper(s[1]) == 'P' && (i = 2)) ||
         (toupper(*s) == 'C' && toupper(s[1]) == 'H' && (i = 3)))) {
      alphabet = (i == 1 ? PROTEIN : (i == 2 ? DNA : UNKNOWN));
      for (s--; s >= start && isspace(*s); s--) ;
    }
    else
      flag = 0;
  }

  if (flag && s >= start) {
    if (isdigit(*s)) {
      while (s >= start && isdigit(*s)) s--;
      while (s >= start && isspace(*s)) s--;
    }
    else
      flag = 0;
  }

  if (flag && s >= start && *s == ',') {
    alphaend = end;
    end = s;

    if (allflag || info_flag == SEQINFO_ALPHABET ||
        info_flag == SEQINFO_CIRCULAR || info_flag == SEQINFO_FRAGMENT ||
        info_flag == SEQINFO_STARTPOS) {
      if (alphabet != UNKNOWN && (allflag || info_flag == SEQINFO_ALPHABET))
        set_alphabet(info, alphabet);

      if (alphastart != NULL) {
        s = alphastart + 1;
        while (s < alphaend && isspace(*s)) s++;
        while (s < alphaend && *s != ')') {
          for (t=s; s < alphaend && *s != ')' && !isspace(*s); s++) ;
          if ((alphabet = get_alphabet(t, s)) != UNKNOWN &&
              (allflag || info_flag == SEQINFO_ALPHABET))
            set_alphabet(info, alphabet);

          if ((allflag || info_flag == SEQINFO_FRAGMENT) &&
              (mystreq(t, 'F', "FRAGMENT") || mystreq(t, 'P', "PARTIAL")))
            set_fragment(info, 1);

          if ((allflag || info_flag == SEQINFO_FRAGMENT ||
               info_flag == SEQINFO_STARTPOS) &&
              toupper(*t) == 'F' && t[1] == '.' && isspace(t[2]) &&
              sscanf(t+2, " %d-%d", &num1, &num2) == 2) {
            if (info_flag != SEQINFO_STARTPOS)
              set_fragment(info, 1);
            if (info_flag != SEQINFO_FRAGMENT)
              set_fragstart(info, num1);
            while (s < alphaend && isspace(*s)) s++;
            while (s < alphaend && !isspace(*s)) s++;
          }

          if ((allflag || info_flag == SEQINFO_CIRCULAR) &&
              mystreq(t, 'C', "CIRCULAR"))
            set_circular(info, 1);

          while (s < alphaend && isspace(*s)) s++;
        }
      }
    }
  }
  if (info_flag == SEQINFO_ALPHABET || info_flag == SEQINFO_CIRCULAR ||
      info_flag == SEQINFO_STARTPOS)
    return;

  /*
   * Next, check for a list of identifiers at the beginning of the string.
   * The identifier list is of the following form:
   *
   *   1) It may begin with a '~', followed by 5-12 alphanumeric
   *      characters.  That string is treated as an accession number.
   *      That string (and a following '|') is skipped before checking
   *      for parts 2a or 2b.
   *   2a) If 2-4 alphanumeric characters are then followed by a ':',
   *       the initial non-whitespace segment is treated as a SEQIO
   *       identifier list.
   *   2b) If 2-3 alphanumeric characters are then followed by a '|'
   *       the initial non-whitespace segment is treated as an NCBI
   *       Search Format identifier list.
   */
  s = start;
  if (*s == '~') {
    for (t=s++; s < end && isalnum(*s); s++) ;
    if (s - t >= 6 && s - t <= 13) {
      if (allflag || info_flag == SEQINFO_IDLIST)
        add_id(info, "acc", t+1, s);
      if (*s == '|')
        s++;
    }
    else
      s = t;
  }
 
  if (is_idprefix(s)) {
    while (s < end && !isspace(*s)) {
      for (t=s; s < end && !isspace(*s) && *s != '|'; s++) ;
      if (allflag || info_flag == SEQINFO_IDLIST)
        add_id(info, "", t, s);
      if (s < end && *s == '|') {
        s++;
        if (!is_idprefix(s)) {
          while (s < end && !isspace(*s)) s++;
          break;
        }
      }
    }
  }
  else if (is_ncbiprefix(s)) {
    if (allflag || info_flag == SEQINFO_IDLIST)
      s = parse_ncbi_idlist(info, s, end);
    while (s < end && !isspace(*s)) s++;
  }

  /*
   * Check the end of the line for one of the keywords "(fragment)",
   * "(fragments)" or "(tentative sequence)" in the cases where no
   * length segment was found.  This is the case when alphastart is
   * set and is less than the end of the line.
   */
  tseqflag = 0;
  if ((allflag || info_flag == SEQINFO_FRAGMENT ||
       info_flag == SEQINFO_DESCRIPTION || info_flag == SEQINFO_ORGANISM) &&
      alphastart != NULL && alphastart < end) {
    if (mystreq(alphastart, '(', "(FRAGMENT)") ||
        mystreq(alphastart, '(', "(FRAGMENTS)")) {
      if (info_flag != SEQINFO_DESCRIPTION)
        set_fragment(info, 1);
      for (end=alphastart-1; end > start && isspace(*(end-1)); end--) ;
    }
    else if (mystreq(alphastart, '(', "(TENTATIVE SEQUENCE)")) {
      tseqflag = 1;
      for (end=alphastart-1; end > start && isspace(*(end-1)); end--) ;
    }
  }

  if (info_flag == SEQINFO_FRAGMENT)
    return;

  /*
   * Finally, find the description and organism sections in the rest of the
   * line.  The separation between description and organism occurs at
   * the string " - ", and if that doesn't appear, then the rest of the text
   * is considered the description.
   */
  for (start=s; start < end && isspace(*start); start++) ;
  s = start;
  while (s + 1 < end && !(*s == '-' && s[-1] == ' ' && s[1] == ' '))
    s++;

  if (s + 1 < end) {
    if (allflag || info_flag == SEQINFO_ORGANISM)
      add_organism(info, s+1, end);
    end = s - 1;
  }

  if (allflag || info_flag == SEQINFO_DESCRIPTION) {
    add_description(info, start, end);
    if (tseqflag)
      add_description(info, "(tentative sequence)", NULL);
  }
}


#define ncbi_idtable_size 13
static struct {
  char *prefix;
  int len;
  char *type1, *type2;
} ncbi_idtable[ncbi_idtable_size] = {
  { "gi|",  3, "gi",  NULL },
  { "bbs|", 4, "bbs", NULL },
  { "bbm|", 4, "bbm", NULL },
  { "gb|",  3, "acc", "gb" },
  { "gp|",  3, "acc", "gp" },
  { "emb|", 4, "acc", "embl" },
  { "pir|", 4, "acc", "pir" },
  { "sp|",  3, "acc", "sp" },
  { "dbj|", 4, "acc", "ddbj" },
  { "prf|", 4, "acc", "prf" },
  { "pdb|", 4, "pdb", "-" },
  { "oth|", 4, "acc", "oth" },
  { "lcl|", 4, "oth" }
};

static int is_ncbiprefix(char *s)
{
  return (isalpha(s[0]) && isalpha(s[1]) && 
          (s[2] == '|' || (isalpha(s[2]) && s[3] == '|')));
}

static char *parse_ncbi_idlist(INFO *info, char *s, char *end)
{
  int i;
  char *t, *t2, *t3, *pref1, *pref2;

  while (s < end && !isspace(*s)) {
    if (end - s <= 5)
      return s;

    for (i=0; i < ncbi_idtable_size; i++)
      if (myncasecmp(s, ncbi_idtable[i].prefix, ncbi_idtable[i].len) == 0)
        break;

    if (i < ncbi_idtable_size) {
      pref1 = ncbi_idtable[i].type1;
      pref2 = ncbi_idtable[i].type2;

      s += ncbi_idtable[i].len;
      for (t=s; s < end && !isspace(*s) && *s != '|'; s++) ;
      if (t == s || (pref2 != NULL && (s == end || isspace(*s))))
        return t;

      if (pref2 == NULL)
        add_id(info, pref1, t, s);
      else {
        for (t2=++s; s < end && !isspace(*s) && *s != '|'; s++) ;
        if (t2 == s)
          return t;

        if (pref2 != NULL && pref2[0] != '-')
          add_id(info, pref2, t2, s);
        add_id(info, pref1, t, t2-1);
      }

      if (s < end && *s == '|')
        s++;
    }
    else if (mystreq(s, 'P', "PAT|")) {
      for (t=(s+=4); s < end && !isspace(*s) && *s != '|'; s++) ;
      if (t == s || s == end || isspace(*s))
        return t;

      for (t2=s++; s < end && !isspace(*s) && *s != '|'; s++) ;
      if (t2 == s || s == end || isspace(*s))
        return t;

      for (t3=s++; s < end && !isspace(*s) && *s != '|'; s++) ;
      if (t3 == s)
        return t;

      *t2 = *t3 = '.';
      add_id(info, "pat", t, s);
      *t2 = *t3 = '|';
    }
    else if (mystreq(s, 'G', "GNL|")) {
      for (t=(s+=4); s < end && !isspace(*s) && *s != '|'; s++) ;
      if (t == s || s == end || isspace(*s))
        return t;

      for (t2=s++; s < end && !isspace(*s) && *s != '|'; s++) ;
      if (t2 == s)
        return t;

      *t2 = ':';
      add_id(info, "oth", t, s);
      *t2 = '|';
    }
    else
      return s;
  }

  return s;
}


void parse_gcg_oneline(INFO *info, char *line, int flag)
{
  int gcgflag, month;
  char *s, *t, date[16];

  for (s=line; *s != '\n' && isspace(*s); s++) ;

  gcgflag = 0;
  while (*s != '\n') {
    if (mystreq(s, 'M', "MSF: ")) {
      while (*s != '\n' && !isspace(*s)) s++;
      while (*s != '\n' && isspace(*s)) s++;
    }
    else if (mystreq(s, 'L', "LENGTH: ")) {
      while (*s != '\n' && !isspace(*s)) s++;
      while (*s != '\n' && isspace(*s)) s++;
    }
    else if (mystreq(s, 'T', "TYPE: ")) {
      while (*s != '\n' && !isspace(*s)) s++;
      while (*s != '\n' && isspace(*s)) s++;
      if (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO ||
          flag == SEQINFO_ALPHABET) {
        if (toupper(*s) == 'N')
          set_alphabet(info, DNA);
        else if (toupper(*s) == 'P')
          set_alphabet(info, PROTEIN);
      }
    }
    else if (mystreq(s, 'C', "CHECK: ")) {
      while (*s != '\n' && !isspace(*s)) s++;
      while (*s != '\n' && isspace(*s)) s++;
    }
    else if ((month = isamonth(s)) != 0) {
      while (*s != '\n' && !isspace(*s)) s++;
      while (*s != '\n' && isspace(*s)) s++;

      if ((flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO ||
           flag == SEQINFO_DATE) && !info->error && info->info->date == NULL &&
          isdigit(*s) && isdigit(s[1]) && s[2] == ',' &&
          s[3] == ' ' && isdigit(s[4]) && isdigit(s[5]) &&
          isdigit(s[6]) && isdigit(s[7]) && isspace(s[8])) {
        date[0] = *s;
        date[1] = s[1];
        date[2] = '-';
        date[3] = months[month][0];
        date[4] = months[month][1];
        date[5] = months[month][2];
        date[6] = '-';
        date[7] = s[4];
        date[8] = s[5];
        date[9] = s[6];
        date[10] = s[7];
        date[11] = '\0';
        set_date(info, date, date + 11);
      }
      for (s+=8; *s != '\n' && isspace(*s); s++) ;
    }
    else if (gcgflag == 0) {
      if (!info->error && info->info->description == NULL) {
        for (t=s; !isspace(*t); t++) ;
        add_description(info, s, t);
      }
    }

    while (*s != '\n' && !isspace(*s)) s++;
    while (*s != '\n' && isspace(*s)) s++;
    gcgflag = 1;
  }
}



static char *gistring, *giptr, *giend, *gilastline;

static void gi_startline(char *s, int len)
{
  giptr = gistring = s;
  giend = s + len;
  gilastline = NULL;
}

static int gi_getline(char **line_out, char **end_out, int flag)
{
  char *s;

  if (gilastline != NULL) {
    giptr = gilastline;
    gilastline = NULL;
  }

  if (giptr >= giend)
    return 0;
    
  s = giptr;
  while (1) {
    while (s < giend && *s != '\n') s++;
    if (!flag || s+1 >= giend || !isspace(s[1]))
      break;
    s++;
  }

  *line_out = giptr;
  *end_out = giptr = s;

  if (giptr < giend)
    giptr++;

  return 1;
}

static void gi_ungetline(char *line)
{
  gilastline = line;
}




static int raw_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int status;
  char *s, *t;
  INFO info;
  
  start_info(&info, isfp, flag);

  if (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO ||
       flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO ||
        flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO ||
        flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK || status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }

    if (flag == SEQINFO_RAWLEN || flag == SEQINFO_TRUELEN)
      goto RAW_GI_END;
  }

  /*
   * Store the name of the file (minus any path information) as the
   * description of the entry.
   */
  if (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO ||
      flag == SEQINFO_DESCRIPTION) {
    for (s=t=isfp->filename; *s; s++)
      if (*s == '/')
        t = s + 1;
    add_description(&info, t, s);
  }

  /*
   * Finish the INFO structure.
   */
  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 0, NULL);
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

RAW_GI_END:
  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int genbank_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int status, allflag;
  char *s, *t, *line, *end;
  INFO info;

  if (!mystreq(entry, 'L', "LOCUS ")) {
    if (isfp->filename && isfp->filename[0]) {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("%s, entry %d:  Invalid format of GenBank "
                              "entry.\n", isfp->filename, isfp->entry_count));
    }
    else {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("seqfparseent:  Invalid format of GenBank "
                              "entry.\n"));
    }
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }

    if (flag == SEQINFO_RAWLEN || flag == SEQINFO_TRUELEN)
      goto GENBANK_GI_END;
  }

  /*
   * The LOCUS line contains an identifier, the alphabet type, a "circular"
   * flag and the date.
   * The ACCESSION line contains the accession number(s).
   * The DEFINITION line contains the description string.
   * The ORGANISM line of the SOURCE structure contains the organism name.
   * The COMMENT line contains any or all of an identifier list, comment
   * lines and history lines.
   */
  gi_startline(entry, len);
  while ((status = gi_getline(&line, &end, 1)) &&
         (line[0] != 'O' || strncmp(line, "ORIGIN", 6) != 0)) {
    switch (line[0]) {
    case 'A':
      if (strncmp(line, "ACCESSION", 9) == 0) {
        if (allflag || flag == SEQINFO_IDLIST) {
          s = line + 12;
          while (s < end && isspace(*s)) s++;
          while (s < end) {
            for (t=s; s < end && !isspace(*s); s++) ;
            add_id(&info, "acc", t, s);
            while (s < end && isspace(*s)) s++;
          }
        }
      }
      break;

    case 'C':
      if (strncmp(line, "COMMENT", 7) == 0) {
        if (allflag || flag == SEQINFO_IDLIST || flag == SEQINFO_COMMENT) {
          parse_comment(&info, line+7, end, 1, 12, flag);
          if (flag == SEQINFO_COMMENT || flag == SEQINFO_IDLIST)
            goto GENBANK_GI_END;
        }
      }
      break;

    case 'D':
      if (strncmp(line, "DEFINITION", 10) == 0) {
        if (allflag || flag == SEQINFO_DESCRIPTION) {
          add_description(&info, line+12, end);
          if (flag == SEQINFO_DESCRIPTION)
            goto GENBANK_GI_END;
        }
      }
      break;

    case 'L':
      if (strncmp(line, "LOCUS", 5) == 0) {
        if (allflag || flag == SEQINFO_IDLIST)
          add_id(&info, (isfp->db_idprefix ? isfp->db_idprefix : "gb"),
                 line+12, line+22);

        if (allflag || flag == SEQINFO_ALPHABET) {
          set_alphabet(&info, get_alphabet(line+36, line+40));
          if (flag == SEQINFO_ALPHABET)
            goto GENBANK_GI_END;
        }

        if ((allflag || flag == SEQINFO_CIRCULAR) &&
            !strncmp(line+42, "circular", 8)) {
          set_circular(&info, 1);
          if (flag == SEQINFO_CIRCULAR)
            goto GENBANK_GI_END;
        }

        if (allflag || flag == SEQINFO_DATE) {
          set_date(&info, line+62, line+73);
          if (flag == SEQINFO_DATE)
            goto GENBANK_GI_END;
        }
      }
      break;

    case 'N':
    case 'P':
      if (toupper(line[1]) == 'I' && toupper(line[2]) == 'D' &&
          isspace(line[3]) && (allflag || flag == SEQINFO_IDLIST))
        add_id(&info, (line[0] == 'N' ? "nid" : "pid"), line+12, end);
      break;

    case 'S':
      if (strncmp(line, "SOURCE", 6) == 0) {
        if (allflag || flag == SEQINFO_ORGANISM) {
          for (s=line+6; s < end; s++) {
            if (*s == '\n' && strncmp(s, "\n  ORGANISM", 11) == 0) {
              s += 11;
              for (t=s; s < end && *s != '\n'; s++) ;
              add_organism(&info, t, s);
              break;
            }
          }
          if (flag == SEQINFO_ORGANISM)
            goto GENBANK_GI_END;
        }
      }
      break;
    }
  }

  if (status == 0) {
    if (isfp->filename && isfp->filename[0]) {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("%s, entry %d:  Invalid format of GenBank "
                              "entry.\n", isfp->filename, isfp->entry_count));
    }
    else {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("seqfparseent:  Invalid format of GenBank "
                              "entry.\n"));
    }
  }

  /*
   * Add the complete header as a comment if SEQINFO_ALLINFO is specified.
   */
  if (flag == SEQINFO_ALLINFO) {
    for (s=line; s < end && *s != '\n'; s++) ;
    add_comment(&info, entry, s+1, 0, 0);
  }

  /*
   * Check the GCG infoline for information about the date, alphabet and
   * description.
   */
  if (isfp->format == FORMAT_GCG && isfp->gcg_infoline &&
      (allflag || flag == SEQINFO_DATE || flag == SEQINFO_ALPHABET ||
       flag == SEQINFO_DESCRIPTION))
    parse_gcg_oneline(&info, isfp->gcg_infoline, flag);

GENBANK_GI_END:
  /*
   * Finish the INFO structure.
   */
  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "GenBank");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int pir_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int status, allflag;
  char *s, *t, *line, *end;
  INFO info;

  if (!mystreq(entry, 'E', "ENTRY ")) {
    if (isfp->filename && isfp->filename[0]) {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("%s, entry %d:  Invalid format of PIR "
                              "entry.\n", isfp->filename, isfp->entry_count));
    }
    else {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("seqfparseent:  Invalid format of PIR "
                              "entry.\n"));
    }
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL) {
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));
    if (flag == SEQINFO_ALPHABET)
      goto PIR_GI_END;
  }

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[FORMAT_PIR].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }

    if (flag == SEQINFO_RAWLEN || flag == SEQINFO_TRUELEN)
      goto PIR_GI_END;
  }

  /*
   * The ENTRY line contains the idnum number.  The ACCESSION line contains
   * the accession number.  The ORGANISM line contains the organism name
   * in the "#formal_name" sub-field.  The TITLE line contains the 
   * description.
   */
  gi_startline(entry, len);
  while ((status = gi_getline(&line, &end, 1)) &&
          (line[0] != 'S' || strncmp(line, "SEQUENCE", 8) != 0)) {
    switch (line[0]) {
    case 'A':
      if (strncmp(line, "ACCESSION", 9) == 0) {
        if (allflag || flag == SEQINFO_IDLIST) {
          s = line + 10;
          while (s < end && isspace(*s)) s++;
          while (s < end) {
            for (t=s; s < end && !isspace(*s) && *s != ';'; s++) ;
            add_id(&info, "acc", t, s);
            while (s < end && (isspace(*s) || *s == ';')) s++;
          }
        }
      }
      break;

    case 'C':
      if (strncmp(line, "COMMENT", 7) == 0) {
        if (allflag || flag == SEQINFO_IDLIST || flag == SEQINFO_COMMENT)
          parse_comment(&info, line+7, end, 1, 11, flag);
      }
      break;

    case 'D':
      if (strncmp(line, "DATE", 4) == 0) {
        if (allflag || flag == SEQINFO_DATE) {
          for (s=line+4; s < end && isspace(*s); s++) ;
          if (s + 11 < end) {
            for (t=s; t < end; t++) {
              if (*t == '#') {
                while (t < end && !isspace(*t)) t++;
                while (t < end && isspace(*t)) t++;
                if (t+11 <= end)
                  s = t;
              }
            }
            set_date(&info, s, s+11);
          }
          if (flag == SEQINFO_DATE)
            goto PIR_GI_END;
        }
      }
      break;

    case 'E':
      if (strncmp(line, "ENTRY", 5) == 0) {
        if (allflag || flag == SEQINFO_IDLIST || flag == SEQINFO_FRAGMENT) {
          for (s=line+5; s < end && isspace(*s); s++) ;
          for (t=s; s < end && !isspace(*s); s++) ;

          add_id(&info, (isfp->db_idprefix ? isfp->db_idprefix : "pir"), t, s);
          while (1) {
            while (s < end && *s != '#') s++;
            if (s == end)
              break;

            if (strncmp(s, "#type ", 6) == 0) {
              for (s+=6; s < end && isspace(*s); s++) ;
              if (s < end && strncmp(s, "fragment", 8) == 0)
                set_fragment(&info, 1);
              if (flag == SEQINFO_FRAGMENT)
                goto PIR_GI_END;
              break;
            }

            s++;
          }
        }
      }
      break;

    case 'O':
      if (strncmp(line, "ORGANISM", 8) == 0 &&
          (allflag || flag == SEQINFO_ORGANISM) && !info.error) {
        if (info.info->organism != NULL)
          info.info->organism = NULL;

        for (s=line+8; s < end; s++)
          if (strncmp(s, "#formal_name", 12) == 0)
            break;

        for (t=s+=12; s < end && *s != '#'; s++) ;
        add_organism(&info, t, s);

        if (flag == SEQINFO_ORGANISM)
          goto PIR_GI_END;
      }
      break;

    case 'T':
      if (strncmp(line, "TITLE", 5) == 0) {
        if (allflag || flag == SEQINFO_IDLIST || flag == SEQINFO_ORGANISM ||
            flag == SEQINFO_FRAGMENT || flag == SEQINFO_CIRCULAR ||
            flag == SEQINFO_DESCRIPTION) {
          parse_oneline(&info, line+6, end, flag);
             
          if (flag == SEQINFO_DESCRIPTION || flag == SEQINFO_FRAGMENT ||
              flag == SEQINFO_CIRCULAR)
            goto PIR_GI_END;
        }
      }
      break;
    }
  }

  /*
   * Add the complete header as a comment if SEQINFO_ALLINFO is specified.
   */
  if (flag == SEQINFO_ALLINFO) {
    for (s=line; s < end && *s != '\n'; s++) ;
    add_comment(&info, entry, s+1, 0, 0);
  }

  /*
   * Check the GCG infoline for information about the date, alphabet and
   * description.
   */
  if (isfp->format == FORMAT_GCG && isfp->gcg_infoline &&
      (allflag || flag == SEQINFO_DATE || flag == SEQINFO_ALPHABET ||
       flag == SEQINFO_DESCRIPTION))
    parse_gcg_oneline(&info, isfp->gcg_infoline, flag);

PIR_GI_END:
  /*
   * Finish the INFO structure.
   */
  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "PIR");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}

/*
 * This is commented out until I decide how to handle the reference id's.
 *

#define dr_table_size 24
static struct {
  char *name, *type1, *type2;
} dr_table[dr_table_size] = {
  { "AARHUS/GHENT-2DPAGE",  "ag2d",  NULL },
  { "AGIS",                 "acc",   NULL },
  { "CPGISLE",              "cpg",   NULL },
  { "DICTYDB",              "acc",   "ddb" },
  { "ECO2DBASE",            "e2d",   NULL },
  { "ECOGENE",              "acc",   "eco" },
  { "EMBL",                 "acc",   "embl" },
  { "EPD",                  "epd",   NULL },
  { "FLYBASE",              "fly",   NULL },
  { "GCRDB",                "gcr",   NULL },
  { "GENBANK",              "acc",   "gb" },
  { "HIV",                  "acc",   "hiv" },
  { "IMGT/LIGM",            "acc",   NULL },
  { "MAIZEDB",              "mdb",   NULL },
  { "MIM",                  "mim",   NULL },
  { "PDB",                  "pdb",   NULL },
  { "PIR",                  "acc",   "pir" },
  { "PROSITE",              "acc",   "pros" },
  { "REBASE",               "reb",   NULL },
  { "SWISS-2DPAGE",         "acc",   NULL },
  { "SWISS-PROT",           "acc",   "sp" },
  { "TRANSFAC",             "acc",   "tfd" },
  { "WORMPEP",              NULL,    "wpep" },
  { "YEPD",                 "yepd",  NULL }
};
*
*/

static int embl_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int alpha, period, count, os_flag, status, allflag;
  char *s, *t, *line, *end, *lastline, *prefix;
  INFO info;

  if (!mystreq(entry, 'I', "ID   ")) {
    if (isfp->filename && isfp->filename[0]) {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("%s, entry %d:  Invalid format of "
                              "EMBL/Swiss-Prot entry.\n", isfp->filename,
                              isfp->entry_count));
    }
    else {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("seqfparseent:  Invalid format of "
                              "EMBL/Swiss-Prot entry.\n"));
    }
  }

  prefix = NULL;
  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }

    if (flag == SEQINFO_RAWLEN || flag == SEQINFO_TRUELEN)
      goto EMBL_GI_END;
  }

  /*
   * The ID line contains an identifier and possibly the alphabet and whether
   * it is circular.
   * The AC line(s) contain accession numbers.
   * The NI and PI lines contain NID and PID identifiers.
   * The DT line(s) contain dates (we assume the last date is the latest).
   * The DE line(s) contain the description.
   * The OS line(s) contain the organism name.
   * The CC and XX lines contain comments.
   * The DR lines may contain cross-references (see dr_table above).
   */
  gi_startline(entry, len);
  os_flag = 0;
  while ((status = gi_getline(&line, &end, 0)) &&
         !(line[0] == '/' && line[1] == '/') && !mystreq(line, 'S', "SQ   ")) {
    if (!isspace(line[2]) || !isspace(line[3]) || !isspace(line[4]))
      continue;

    switch (line[0]) {
    case 'A':
      if (line[1] == 'C' && (allflag || flag == SEQINFO_IDLIST)) {
        for (s=line+5; s < end && isspace(*s); s++) ;
        while (s < end) {
          for (t=s; s < end && !isspace(*s) && *s != ';'; s++) ;
          add_id(&info, "acc", t, s);
          while (s < end && (isspace(*s) || *s == ';')) s++;
        }
      }
      break;

    case 'C':
      if (line[1] == 'C' && 
          (allflag || flag == SEQINFO_COMMENT || flag == SEQINFO_IDLIST)) {
        for (s=line+5; s < end && isspace(*s); s++) ;
        if (s < end) {
          lastline = end;
          while (gi_getline(&line, &end, 0)) {
            if (mystreq(line, 'C', "CC   "))
              lastline = end;
            else if (!mystreq(line, 'X', "XX"))
              break;
          }
          gi_ungetline(line);
          parse_comment(&info, s, lastline, 1, 5, flag);
        }
      }
      break;

    case 'D':
      if (line[1] == 'E' &&
          (allflag || flag == SEQINFO_DESCRIPTION ||
           flag == SEQINFO_FRAGMENT)) {
        add_description(&info, line+5, end);
        while (gi_getline(&line, &end, 0) && mystreq(line, 'D', "DE   "))
          add_description(&info, line+5, end);
        gi_ungetline(line);

        if (!info.error && info.info->description) {
          for (t=s=info.info->description; *s; s++) ;
          while (s > t && isspace(*(s-1))) s--;
          if (s - t >= 10 && *(s-1) == ')' &&
              (mystreq(s-10, '(', "(FRAGMENT)") ||
               mystreq(s-11, '(', "(FRAGMENTS)"))) {
            set_fragment(&info, 1);
            s -= (*(s-10) == '(' ? 10 : 11);
            while (s > t && isspace(*(s-1))) s--;
            if (s > t)
              *s = '\0';
            else
              info.info->description = NULL;
            if (flag == SEQINFO_FRAGMENT)
              goto EMBL_GI_END;
          }
        }

        if (flag == SEQINFO_DESCRIPTION)
          goto EMBL_GI_END;
      }
      else if (line[1] == 'T' &&
               (flag == SEQINFO_ALL || flag == SEQINFO_DATE)) {
        for (s=line+5; s < end && isspace(*s); s++) ;
        for (t=s; s < end && !isspace(*s) && *s != ';' && *s != '.'; s++) ;
        set_date(&info, t, s);
      }

     /*
      * Like above, this is commented out until I can figure out
      * how to handle reference id's.
      *
      else if (line[1] == 'R' && (allflag || flag == SEQINFO_IDLIST)) {
        for (s=line+5; s < end && isspace(*s); s++) ;
        if (s < end) {
          for (i=0; i < dr_table_size; i++) {
            if (!myncasecmp(s, dr_table[i].name, strlen(dr_table[i].name))) {
              while (s < end && !isspace(*s) && *s != ';') s++;
              while (s < end && (isspace(*s) || *s == ';')) s++;

              for (t=s; s < end && !isspace(*s) && *s != ';'; s++) ;
              t1 = t;
              s1 = s;

              while (s < end && (isspace(*s) || *s == ';')) s++;
              for (t=s; s < end && !isspace(*s) && *s != ';' && *s != '.'; s++)
                ;
              if (t != s && dr_table[i].type2 != NULL)
                add_id(&info, dr_table[i].type2, t, s);

              if (t1 != s1 && dr_table[i].type1 != NULL)
                add_id(&info, dr_table[i].type1, t1, s1);
              break;
            }
          }
        }
      }
      *
      */
      break;

    case 'I':
      if (line[1] == 'D' && 
          (allflag || flag == SEQINFO_IDLIST ||
           flag == SEQINFO_ALPHABET || flag == SEQINFO_CIRCULAR)) {
        if (allflag || flag == SEQINFO_IDLIST) {
          if (isfp->db_idprefix != NULL)
            prefix = isfp->db_idprefix;
          else {
            t = NULL;
            period = count = 0;
            for (s=line+5; s < end; s++) {
              if (*s == ';') {
                count++;
                t = s;
              }
              else if (*s == '.')
                period = 1;
            }

            if (count == 3 && period)
              prefix = (mystreq(t-3, 'E', "EPD;") ? "epd" : "embl");
            else if (count == 2 && period && mystreq(t-3, 'P', "PRT;"))
              prefix = "sp";
            else
              prefix = "oth";
          }
        }
        
        for (s=line+5; s < end && isspace(*s); s++) ;
        for (t=s; s < end && !isspace(*s) && *s != ';'; s++) ;

        if (t != s && (allflag || flag == SEQINFO_IDLIST))
          add_id(&info, prefix, t, s);

        if (allflag || flag == SEQINFO_ALPHABET || flag == SEQINFO_CIRCULAR) {
          for ( ; s < end && (isspace(*s) || *s == ';'); s++) ;
          while (s < end) {
            for (t=s; s < end && !isspace(*s) && *s != ';' && *s != '.'; s++);
            if (s - t == 8 && strncmp(t, "circular", 8) == 0 &&
                (allflag || flag == SEQINFO_CIRCULAR)) {
              set_circular(&info, 1);
              if (flag == SEQINFO_CIRCULAR)
                goto EMBL_GI_END;
            }
            else if (s > t && (alpha = get_alphabet(t, s)) != UNKNOWN &&
                     (allflag || flag == SEQINFO_ALPHABET)) {
              set_alphabet(&info, alpha);
              if (flag == SEQINFO_ALPHABET)
                goto EMBL_GI_END;
              else
                break;
            }
            while (s < end && (isspace(*s) || *s == ';' || *s == '.')) s++;
          }
        }
      }
      break;

    case 'N':
    case 'P':
      if (line[1] == 'I' && (allflag || flag == SEQINFO_IDLIST))
        add_id(&info, (line[0] == 'N' ? "nid" : "pid"), line+5, end);
      break;

    case 'O':
      if (line[1] == 'S' && !os_flag &&
          (allflag || flag == SEQINFO_ORGANISM)) {
        for (s=end; s > line + 5 && isspace(*(s-1)); s--) ;
        if (*(s-1) == '.') s--;
        add_organism(&info, line + 5, s);
        os_flag = 1;
        if (flag == SEQINFO_ORGANISM)
          goto EMBL_GI_END;
      }
      break;

    case 'X':
      if (line[1] == 'X' && 
          (allflag || flag == SEQINFO_COMMENT || flag == SEQINFO_IDLIST)) {
        for (s=line+5; s < end && isspace(*s); s++) ;
        if (s < end) {
          while (gi_getline(&line, &end, 0))
            if (!mystreq(line, 'X', "XX   "))
              break;
          gi_ungetline(line);
          parse_comment(&info, s, line, 1, 5, flag);
        }
      }
      break;
    }
  }


  /*
   * Add the complete header as a comment if SEQINFO_ALLINFO is specified.
   */
  if (flag == SEQINFO_ALLINFO) {
    if (status == 0)
      add_comment(&info, entry, entry + len, 0, 0);
    else
      add_comment(&info, entry, end+1, 0, 0);
  }

  /*
   * Check the GCG infoline for information about the date, alphabet and
   * description.
   */
  if (isfp->format == FORMAT_GCG && isfp->gcg_infoline &&
      (allflag || flag == SEQINFO_DATE || flag == SEQINFO_ALPHABET ||
       flag == SEQINFO_DESCRIPTION))
    parse_gcg_oneline(&info, isfp->gcg_infoline, flag);


EMBL_GI_END:
  /*
   * Finish the INFO structure.
   */
  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL) {
      if (prefix != NULL && prefix[0] == 's' && prefix [1] == 'p')
        add_retrieval(&info, 1, "Swiss-Prot");
      else
        add_retrieval(&info, 1, "EMBL");
    }
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int nbrf_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int status, allflag;
  char ch, *s, *t, *line, *start, *end;
  INFO info;

  if (*entry != '>') {
    if (isfp->filename && isfp->filename[0]) {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("%s, entry %d:  Invalid format of NBRF "
                              "entry.\n", isfp->filename, isfp->entry_count));
    }
    else {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("seqfparseent:  Invalid format of NBRF "
                              "entry.\n"));
    }
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }

    if (flag == SEQINFO_RAWLEN || flag == SEQINFO_TRUELEN)
      goto NBRF_GI_END;
  }

  /*
   * The first header line contains an identifier after the semi-colon 
   * (which appears at position 4).  The next line is the one-line
   * description.
   */
  gi_startline(entry, len);
  gi_getline(&line, &end, 0);
  ch = toupper(line[1]);
  if ((ch == 'P' || ch == 'F') && line[2] == '1') {
    if (allflag || flag == SEQINFO_ALPHABET) {
      set_alphabet(&info, PROTEIN);
      if (flag == SEQINFO_ALPHABET)
        goto NBRF_GI_END;
    }
    if (allflag || flag == SEQINFO_FRAGMENT) {
      set_fragment(&info, ch == 'F');
      if (flag == SEQINFO_FRAGMENT)
        goto NBRF_GI_END;
    }
  }
  else if (ch == 'D' || ch == 'R') {
    if (allflag || flag == SEQINFO_ALPHABET) {
      set_alphabet(&info, (ch == 'D' ? DNA : RNA));
      if (flag == SEQINFO_ALPHABET)
        goto NBRF_GI_END;
    }
    if (allflag || flag == SEQINFO_CIRCULAR) {
      set_circular(&info, toupper(line[2]) == 'C');
      if (flag == SEQINFO_CIRCULAR)
        goto NBRF_GI_END;
    }
  }

  if (allflag || flag == SEQINFO_IDLIST) {
    s = line + 4;
    while (s < end && !isspace(*s)) {
      for (t=s; s < end && !isspace(*s) && *s != '|'; s++) ;
      add_id(&info, (isfp->db_idprefix ? isfp->db_idprefix : "oth"), t, s);
      if (*s == '|') s++;
    }
  }

  if (allflag || flag == SEQINFO_IDLIST || flag == SEQINFO_DESCRIPTION ||
      flag == SEQINFO_ORGANISM || flag == SEQINFO_FRAGMENT ||
      flag == SEQINFO_CIRCULAR || flag == SEQINFO_STARTPOS) {
    gi_getline(&line, &end, 0);
    parse_oneline(&info, line, end, flag);
  }

  if (flag == SEQINFO_ALLINFO)
    add_comment(&info, entry, end+1, 0, 0);

  /*
   * The rest of the information occurs in the header lines at the end
   * of the entry.  Look for a Date line, Accession lines and Comment
   * lines.
   */
  if (allflag || flag == SEQINFO_IDLIST ||
      flag == SEQINFO_COMMENT || flag == SEQINFO_DATE) {
    start = NULL;
    if (isfp->filename && isfp->filename[0]) {
      if (isfp->nbrf_header != NULL) {
        if (isfp->format != FORMAT_GCG || !isfp->gcg_infoline)
          gi_startline(isfp->nbrf_header,
                       isfp->fp_entryend - isfp->nbrf_header);
        else
          gi_startline(isfp->nbrf_header,
                       isfp->gcg_infoline - isfp->nbrf_header);
        start = isfp->nbrf_header;
      }
    }
    else {
      while ((status = gi_getline(&line, &end, 0)))
        if (line != end && line[1] == ';')
          break;

      if (status == 1) {
        gi_startline(line, (entry + len) - line);
        start = line;
      }
    }
      
    if (start != NULL) {
      while (gi_getline(&line, &end, 0)) {
        if (!(line[0] == 'C' && line[1] == ';'))
          continue;

        if (mystreq(line+2, 'A', "ACCESSION: ")) {
          if (allflag || flag == SEQINFO_IDLIST) {
            for (s=line+13; s < end && isspace(*s); s++) ;
            while (s < end) {
              for (t=s; s < end && !isspace(*s) && *s != ';'; s++) ;
              add_id(&info, "acc", t, s);
              while (s < end && (isspace(*s) || *s == ';')) s++;
            }
          }
        }
        else if (mystreq(line+2, 'C', "COMMENT: ")) {
          if (allflag || flag == SEQINFO_IDLIST || flag == SEQINFO_COMMENT) {
            for (s=line+10; s < end && isspace(*s); s++) ;
            if (s < end) {
              while ((status = gi_getline(&line, &end, 0)))
                if (!mystreq(line, 'C', "C;COMMENT: "))
                  break;

              if (status == 0)
                parse_comment(&info, s, isfp->fp_entryend, 1, 11, flag);
              else {
                gi_ungetline(line);
                parse_comment(&info, s, line, 1, 11, flag);
              }
            }
          }
        }
        else if (mystreq(line+2, 'D', "DATE: ")) {
          if (allflag || flag == SEQINFO_DATE) {
            for (s=line+7; s < end && isspace(*s); s++) ;
            if (s + 11 <= end) {
              for (t=s; t + 11 <= end; t++) {
                if (*t == '#') {
                  while (t < end && !isspace(*t)) t++;
                  while (t < end && isspace(*t)) t++;
                  if (t+11 <= end)
                    s = t;
                }
              }
              set_date(&info, s, s+11);
            }
            if (flag == SEQINFO_DATE)
              goto NBRF_GI_END;
          }
        }
      }

      if (flag == SEQINFO_ALLINFO)
        add_comment(&info, start, end+1, 0, 0);
    }
  }

  /*
   * Check the GCG infoline for information about the date, alphabet and
   * description.
   */
  if (isfp->format == FORMAT_GCG && isfp->gcg_infoline &&
      (allflag || flag == SEQINFO_DATE || flag == SEQINFO_ALPHABET ||
       flag == SEQINFO_DESCRIPTION))
    parse_gcg_oneline(&info, isfp->gcg_infoline, flag);

NBRF_GI_END:
  /*
   * Finish the last fields in the INFO structure.
   */
  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "NBRF");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int fasta_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int allflag, status;
  char *s, *t, *end;
  INFO info;

  if (*entry != '>') {
    if (isfp->filename && isfp->filename[0]) {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("%s, entry %d:  Invalid format of FASTA "
                              "entry.\n", isfp->filename, isfp->entry_count));
    }
    else {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("seqfparseent:  Invalid format of FASTA "
                              "entry.\n"));
    }
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }

    if (flag == SEQINFO_RAWLEN || flag == SEQINFO_TRUELEN)
      goto FASTA_GI_END;
  }

  /*
   * The first header line is a oneline description.  All other
   * header lines are comments.
   */
  s = entry + 1;
  end = entry + len;

  for (t=s; s < end && *s != '\n'; s++) ;
  if (t != s && (allflag || flag == SEQINFO_IDLIST ||
                 flag == SEQINFO_DESCRIPTION || flag == SEQINFO_ORGANISM ||
                 flag == SEQINFO_FRAGMENT || flag == SEQINFO_CIRCULAR ||
                 flag == SEQINFO_STARTPOS)) {
    parse_oneline(&info, t, s, flag);
  }

  if ((s[1] == '>' || s[1] == ';') &&
      (allflag || flag == SEQINFO_COMMENT || flag == SEQINFO_IDLIST)) {
    s += 2;
    for (t=s; s < end && (*s != '\n' || s[1] == '>' || s[1] == ';'); s++) ;
    if (t + 2 < s && *t == '\n' && (t[1] == '>' || t[1] == ';'))
      t += 2;
    parse_comment(&info, t, s, 0, 1, flag);
  }

  if (flag == SEQINFO_ALLINFO)
    add_comment(&info, entry, s+1, 0, 0);

  /*
   * Check the GCG infoline for information about the date, alphabet and
   * description.
   */
  if (isfp->format == FORMAT_GCG && isfp->gcg_infoline &&
      (allflag || flag == SEQINFO_DATE || flag == SEQINFO_ALPHABET ||
       flag == SEQINFO_DESCRIPTION))
    parse_gcg_oneline(&info, isfp->gcg_infoline, flag);

FASTA_GI_END:
  /*
   * Finish the last fields in the INFO structure.
   */
  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "FASTA");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int stanford_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int allflag, status;
  char *s, *t, *end, *comstart, *comend;
  INFO info;

  if (*entry != ';') {
    if (isfp->filename && isfp->filename[0]) {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("%s, entry %d:  Invalid format of IG/Stanford "
                              "entry.\n", isfp->filename, isfp->entry_count));
    }
    else {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("seqfparseent:  Invalid format of IG/Stanford "
                              "entry.\n"));
    }
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }

    if (flag == SEQINFO_RAWLEN || flag == SEQINFO_TRUELEN)
      goto STANFORD_GI_END;
  }

  /*
   * The header lines are comments.  The next line after them is a 
   * oneline description.
   */
  s = entry;
  end = entry + len;

  for (comstart=++s; s < end && (*s != '\n' || s[1] == ';'); s++) ;
  comend = s;

  if (allflag || flag == SEQINFO_IDLIST || flag == SEQINFO_DESCRIPTION ||
      flag == SEQINFO_ORGANISM || flag == SEQINFO_FRAGMENT ||
      flag == SEQINFO_CIRCULAR || flag == SEQINFO_STARTPOS) {
    for (t=++s; s < end && *s != '\n'; s++) ;
    parse_oneline(&info, t, s, flag);
  }

  if (flag == SEQINFO_ALLINFO)
    add_comment(&info, entry, s+1, 0, 0);

  if (comstart != comend && 
      (allflag || flag == SEQINFO_COMMENT || flag == SEQINFO_IDLIST))
    parse_comment(&info, comstart, comend, 0, 1, flag);

  if (allflag || flag == SEQINFO_CIRCULAR) {
    for (s=end; s > entry && isspace(*(s-1)); s--) ;
    if (*(s-1) == '2')
      set_circular(&info, 1);
  }

  /*
   * Check the GCG infoline for information about the date, alphabet and
   * description.
   */
  if (isfp->format == FORMAT_GCG && isfp->gcg_infoline &&
      (allflag || flag == SEQINFO_DATE || flag == SEQINFO_ALPHABET ||
       flag == SEQINFO_DESCRIPTION))
    parse_gcg_oneline(&info, isfp->gcg_infoline, flag);

STANFORD_GI_END:
  /*
   * Finish the last fields in the INFO structure.
   */
  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "IG/Stanford");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int gcg_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int comlen, allflag, status;
  char *s, *end, *infoline;
  INFO info;

  if (isfp->gcg_subformat != FORMAT_UNKNOWN) {
    return 
      (*file_table[isfp->gcg_subformat].getinfo_fn)(isfp, entry, len, flag);
  }

  if (isfp->filename && isfp->filename[0]) {
    program_error(isfp->gcg_infoline == NULL, return STATUS_ERROR,
                  print_error("   gcg_infoline not set by GCG read "
                              "function\n"));

    infoline = isfp->gcg_infoline;
  }
  else {
    for (s=entry,end=entry+len; s < end; s++)
      if (*s == '.' && s[1] == '.' && s[2] == '\n')
        break;

    error_test(s >= end, E_PARSEERROR, return STATUS_ERROR,
               print_error("seqfparseent:  Invalid format of GCG entry.\n"));

    while (s > entry && *(s-1) != '\n') s--;
    infoline = s;
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }
  }

  /*
   * The beginning lines up to the gcg_infoline are comments.  The
   * gcg_infoline contains alphabet, date and description information.
   */
  if (allflag || flag == SEQINFO_COMMENT) {
    comlen = infoline - isfp->fp_entrystart;
    add_comment(&info, entry, entry + comlen, 0, 0);
  }

  /*
   * Check the GCG infoline for information about the date, alphabet and
   * description.
   */
  if (allflag || flag == SEQINFO_DATE || flag == SEQINFO_ALPHABET ||
      flag == SEQINFO_DESCRIPTION)
    parse_gcg_oneline(&info, infoline, flag);

  /*
   * Finish the last fields in the INFO structure.
   */
  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "GCG");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int msf_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int comlen, allflag, status;
  char *s, *t;
  INFO info;

  program_error(isfp->gcg_infoline == NULL, return STATUS_ERROR,
                print_error("   gcg_infoline not set by GCG read "
                            "function\n"));

  while (isfp->malign_seqno < isfp->entry_seqno) {
    for (s=isfp->fp_seqstart; *s != '\n'; s++) ;
    isfp->fp_seqstart = s + 1;
    for (t=s+1; *t != '\n' && isspace(*t); t++) ;
    if (*t != '\n')
      isfp->malign_seqno++;
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }
  }

  /*
   * The beginning lines up to the gcg_infoline are comments.  The
   * gcg_infoline contains alphabet, date and description information.
   */
  if (flag == SEQINFO_ALL || flag == SEQINFO_COMMENT) {
    comlen = isfp->gcg_infoline - isfp->fp_entrystart;
    add_comment(&info, entry, entry + comlen, 0, 0);
  }

  /*
   * Check the GCG infoline for information about the date, alphabet and
   * description.
   */
  if (allflag || flag == SEQINFO_DATE || flag == SEQINFO_ALPHABET ||
      flag == SEQINFO_DESCRIPTION)
    parse_gcg_oneline(&info, isfp->gcg_infoline, flag);

  /*
   * The sequence header line contains the identifier.
   */
  if (allflag || flag == SEQINFO_IDLIST) {
    for (s=isfp->fp_seqstart; isspace(*s); s++) ;
    while (!isspace(*s)) s++;
    while (*s != '\n' && isspace(*s)) s++;
    for (t=s; *s != '\n' && !isspace(*s); s++) ;
    add_id(&info, (isfp->db_idprefix ? isfp->db_idprefix : "oth"), t, s);

    /*
     * If SEQINFO_ALLINFO is set, find the "//" line that ends the header
     * and add all of the text before it as a comment.
     */
    if (flag == SEQINFO_ALLINFO) {
      while (*s != '\n' || s[1] != '/' || s[2] != '/') s++;
      add_comment(&info, entry, s+1, 0, 0);
    }
  }

  /*
   * Finish the last fields in the INFO structure.
   */
  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "MSF");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int fastaout_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int mode, slen, slen2, count, alpha, alpha1, alpha2;
  int allflag, status, seqlen, totallen;
  char ch, *s, *t, *s2, *t2, *line, *end, *entryend, buffer[512];
  char *id1, *idend1, *id2, *idend2, descr1[128], descr2[128];
  INFO info;

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if (((allflag || flag == SEQINFO_TRUELEN ||
        flag == SEQINFO_FRAGMENT) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }

    if (flag == SEQINFO_RAWLEN || flag == SEQINFO_TRUELEN)
      goto FOUT_GI_END;
  }

  if (allflag || flag == SEQINFO_FRAGMENT) {
    totallen = (isfp->entry_seqno == 1 ? isfp->fout_len1 : isfp->fout_len2);
    if (isfp->entry_truelen && totallen && totallen < isfp->entry_truelen)
      set_fragment(&info, 1);

    if (flag == SEQINFO_FRAGMENT)
      goto FOUT_GI_END;
  }

  /*
   * Scan through the entry, and the information collected from the
   * beginning of the file, to construct the needed info.
   */
  entryend = entry + len;
  mode = isfp->fout_mode;
  line = end = NULL;

  if (mode == FASTA_MODE)
    for (line=end=entry; *end != '\n'; end++) ;

  id1 = idend1 = id2 = idend2 = NULL;
  alpha1 = alpha2 = UNKNOWN;
  if (isfp->fout_markx == MARKX10 &&
      (allflag || flag == SEQINFO_IDLIST ||
       flag == SEQINFO_ALPHABET || flag == SEQINFO_COMMENT)) {
    for (s=entry; s < entryend && (*s != '\n' || s[1] != '>'); s++) ;
    s++;
    for (id1=++s; s < entryend && !isspace(*s); s++) ;
    if (alpha1 != UNKNOWN)
      for (idend1=s++; s < entryend && (*s != '\n' || s[1] != '>'); s++) ;
    else {
      for (idend1=s++; s < entryend && (*s != '\n' || s[1] != '>'); s++) {
        if (*s == '\n' && s[1] == ';' && !strncmp(s+2, " sq_type: ", 10)) {
          for (s+=11; s < entryend && *s != '\n' && isspace(*s); s++) ;
          if (toupper(*s) == 'P')
            alpha1 = PROTEIN;
          else if (toupper(*s) == 'D')
            alpha1 = DNA;
        }
      }
    }
    s++;
    for (id2=++s; s < entryend && !isspace(*s); s++) ;
    idend2 = s;
    if (s >= entryend)
      id1 = idend1 = id2 = idend2 = NULL;
    else {
      if (alpha2 != UNKNOWN)
        for (s++; s < entryend && (*s != '\n' || s[1] != '>'); s++) ;
      else {
        for (s++; s < entryend && (*s != '\n' || s[1] != '>'); s++) {
          if (*s == '\n' && s[1] == ';' && !strncmp(s+2, " sq_type: ", 10)) {
            for (s+=11; s < entryend && *s != '\n' && isspace(*s); s++) ;
            if (toupper(*s) == 'P')
              alpha2 = PROTEIN;
            else if (toupper(*s) == 'D')
              alpha2 = DNA;
          }
        }
      }
    }
  }

  if (isfp->entry_seqno == 1) {
    if (mode == LFASTA_MODE && isfp->fout_markx == MARKX10) {
      if (id1 && (allflag || flag == SEQINFO_IDLIST))
        add_id(&info, (isfp->db_idprefix ? isfp->db_idprefix : "oth"),
               id1, idend1);

      if (alpha1 != UNKNOWN && (allflag || flag == SEQINFO_ALPHABET)) {
        set_alphabet(&info, alpha1);
        if (flag == SEQINFO_ALPHABET)
          goto FOUT_GI_END;
      }
    }
    else {
      if (isfp->fout_id1 && (allflag || flag == SEQINFO_IDLIST)) {
        for (s=isfp->fout_id1; *s; s++) ;
        add_id(&info, (isfp->db_idprefix ? isfp->db_idprefix : "oth"),
               isfp->fout_id1, s);
      }

      if (isfp->fout_descr1 && (allflag || flag == SEQINFO_DESCRIPTION)) {
        for (s=isfp->fout_descr1; *s; s++) ;
        add_description(&info, isfp->fout_descr1, s);
        if (flag == SEQINFO_DESCRIPTION)
          goto FOUT_GI_END;
      }

      if (allflag || flag == SEQINFO_ALPHABET) {
        if (isfp->db_alpha != NULL)
          set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));
        else if (isfp->fout_alpha1 != UNKNOWN)
          set_alphabet(&info, isfp->fout_alpha1);
        else if (isfp->fout_markx == MARKX10 && alpha1 != UNKNOWN)
          set_alphabet(&info, alpha1);

        if (flag == SEQINFO_ALPHABET)
          goto FOUT_GI_END;
      }
    }
  }
  else {
    if (mode == LFASTA_MODE && isfp->fout_markx == MARKX10) {
      if (id2 && (allflag || flag == SEQINFO_IDLIST))
        add_id(&info, (isfp->db_idprefix ? isfp->db_idprefix : "oth"),
               id2, idend2);

      if (alpha2 != UNKNOWN && (allflag || flag == SEQINFO_ALPHABET)) {
        set_alphabet(&info, alpha2);
        if (flag == SEQINFO_ALPHABET)
          goto FOUT_GI_END;
      }
    }
    else {
      if (mode == FASTA_MODE) {
        s = (line[0] == '>' && line[1] == '>' ? line + 2 : line);
        for (t=s; !isspace(*s); s++) ;
        if (allflag || flag == SEQINFO_IDLIST)
          add_id(&info, (isfp->db_idprefix ? isfp->db_idprefix : "oth"), t, s);

        while (isspace(*s) && *s != '\n') s++;
        if (isfp->fout_markx != MARKX10)
          for (t=s,s=end-1; s > t && *s != ')'; s--) ;
        else {
          t = s;
          s = end;
        }

        if (s > t) {
          if (isfp->fout_markx != MARKX10 && 
              (allflag || flag == SEQINFO_ALPHABET)) {
            if (isfp->db_alpha != NULL)
              set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));
            else if (*(s-2) == 'a' && *(s-1) == 'a')
              set_alphabet(&info, PROTEIN);
            else if (*(s-2) == 'n' && *(s-1) == 't')
              set_alphabet(&info, DNA);
            
            if (flag == SEQINFO_ALPHABET)
              goto FOUT_GI_END;
          }

          if (isfp->fout_markx != MARKX10)
            for (s--; s > t && *s != '('; s--) ;
          if (s > t && (allflag || flag == SEQINFO_DESCRIPTION)) {
            add_description(&info, t, s);
          }
        }
        if (isfp->fout_markx == MARKX10 && alpha2 != UNKNOWN &&
            (allflag || flag == SEQINFO_ALPHABET)) {
          set_alphabet(&info, alpha2);
          if (flag == SEQINFO_ALPHABET)
            goto FOUT_GI_END;
        }
      }
      else {
        if (isfp->fout_id2 && (allflag || flag == SEQINFO_IDLIST)) {
          for (s=isfp->fout_id2; *s; s++) ;
          add_id(&info, (isfp->db_idprefix ? isfp->db_idprefix : "oth"),
                 isfp->fout_id2, s);
        }

        if (isfp->fout_descr2 && (allflag || flag == SEQINFO_DESCRIPTION)) {
          for (s=isfp->fout_descr2; *s; s++) ;
          add_description(&info, isfp->fout_descr2, s);
          if (flag == SEQINFO_DESCRIPTION)
            goto FOUT_GI_END;
        }

        if (allflag || flag == SEQINFO_ALPHABET) {
          if (isfp->db_alpha != NULL)
            set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));
          else if (isfp->fout_alpha2 != UNKNOWN)
            set_alphabet(&info, isfp->fout_alpha2);
          else if (isfp->fout_markx == MARKX10 && alpha2 != UNKNOWN)
            set_alphabet(&info, alpha2);

          if (flag == SEQINFO_ALPHABET)
            goto FOUT_GI_END;
        }
      }
    }
  }

  if (allflag || flag == SEQINFO_COMMENT) {
    if (mode == LFASTA_MODE && isfp->fout_markx == MARKX10) {
      descr1[0] = '>';
      if (id1) {
        slen = idend1 - id1;
        memcpy(descr1+1, id1, slen);
      }
      else
        slen = 0;
      descr1[slen+1] = '\0';
    }
    else
      sprintf(descr1, ">%s %s, %d %s", isfp->fout_id1, isfp->fout_descr1,
              isfp->fout_len1,
              (isfp->fout_alpha1 == DNA ? "nt" 
                 : (isfp->fout_alpha1 == PROTEIN ? "aa" : "ch")));

    if (mode == LFASTA_MODE && isfp->fout_markx == MARKX10) {
      descr2[0] = '>';
      if (id2) {
        slen = idend2 - id2;
        memcpy(descr2+1, id2, slen);
      }
      else
        slen = 0;
      descr2[slen+1] = '\0';
    }
    else if (mode == FASTA_MODE) {
      if (isfp->fout_markx == MARKX10) {
        descr2[0] = '>';
        slen = end - line - 2;
        memcpy(descr2+1, line+2, slen);

        for (s=end; s < entryend && (*s != '\n' || s[1] != '>'); s++) ;
        for (s++; s < entryend && (*s != '\n' || s[1] != '>'); s++) ;
        count = 0;
        seqlen = 0;
        alpha = UNKNOWN;
        for (s++; s < entryend && count < 2; s++) {
          if (*s == '\n') {
            if (s[1] != ';')
              break;
            else if (strncmp(s+1, "; sq_len: ", 10) == 0) {
              for (s+=11; s < entryend && isspace(*s); s++) ;
              if (s < entryend && isdigit(*s)) {
                seqlen = *s - '0';
                for (s++; isdigit(*s); s++) {
                  seqlen *= 10;
                  seqlen += *s - '0';
                }
              }
              s--;
              count++;
            }
            else if (strncmp(s+1, "; sq_type: ", 11) == 0) {
              for (s+=12; s < entryend && isspace(*s); s++) ;
              ch = toupper(*s);
              if (ch == 'P')
                alpha = PROTEIN;
              else if (ch == 'D')
                alpha = DNA;
              else
                alpha = UNKNOWN;
              s--;
              count++;
            }
          }
        }

        if (seqlen > 0 && alpha != UNKNOWN)
          sprintf(descr2+slen+1, ", %d %s", seqlen,
                  (alpha == PROTEIN ? "aa" : "nt"));
        else
          descr2[slen+1] = '\0';
      }
      else {
        t = (line[0] == '>' && line[1] == '>' ? line + 2 : line);
        for (s=end-1; s > t && *s != ')'; s--) ;
        for (s2=s,s--; s > t && *s != '('; s--) ;
        for (t2=s; s > t && isspace(*(s-1)); s--) ;
        if (s > t) {
          descr2[0] = '>';
          slen = s - t;
          memcpy(descr2+1, t, slen);
          descr2[slen+1] = ',';
          descr2[slen+2] = ' ';
          slen2 = s2 - t2 - 1;
          memcpy(descr2+slen+3, t2+1, slen2);
          descr2[slen+3+slen2] = '\0';
        }
        else
          descr2[0] = '\0';
      }
    }
    else {
      sprintf(descr2, ">%s %s, %d %s", isfp->fout_id2, isfp->fout_descr2,
              isfp->fout_len2, 
              (isfp->fout_alpha2 == DNA ? "nt" 
                : (isfp->fout_alpha2 == PROTEIN ? "aa" : "ch")));
    }

    sprintf(buffer, "From %s output alignment of:\n   %s\nand\n   %s\n\n",
            isfp->fout_progname, descr1, descr2);
    for (end=buffer; *end; end++) ;
    add_comment(&info, buffer, end, 0, 0);

    if (isfp->fout_markx == MARKX10) {
      for (s=entry; s < entryend && *s != '\n'; s++) ;
      s++;
      if (s < end && *s == ';') {
        for (t=s; s < entryend && (*s != '\n' || s[1] == ';'); s++) ;
        if (s < entryend)
          add_comment(&info, t, s+1, 0, 0);
      }
    }
    else {
      s = entry;
      if (!isspace(*s)) {
        for (s=entry; s < entryend && *s != '\n'; s++) ;
        s++;
      }
      if (*s != '\n') {
        for (t=s; s < entryend; s++)
          if (*s == '\n' && (s[1] == '\n' || s[1] == '>'))
            break;
        if (s < entryend)
          add_comment(&info, t, s+1, 0, 0);
      }
    }
  }

FOUT_GI_END:
  /*
   * Finish the last fields in the INFO structure.
   */
  if (allflag)
    add_retrieval(&info, 3, isfp->fout_progname);

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int blastout_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int i, allflag, status, startflag, size, pos, maxsize, fragstart, totallen;
  char *s, *s2, *t, *end, *pattern, buffer[512];
  INFO info;

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if (((allflag || flag == SEQINFO_TRUELEN ||
        flag == SEQINFO_FRAGMENT) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }

    if (flag == SEQINFO_RAWLEN || flag == SEQINFO_TRUELEN)
      goto BOUT_GI_END;
  }

  /*
   * Scan through the entry, and the information collected from the
   * beginning of the file, to construct the needed info.
   */
  if (isfp->fout_alpha1 != UNKNOWN && (allflag || flag == SEQINFO_ALPHABET)) {
    set_alphabet(&info, isfp->fout_alpha1);
    if (flag == SEQINFO_ALPHABET)
      goto BOUT_GI_END;
  }

  if (allflag || flag == SEQINFO_IDLIST ||
      flag == SEQINFO_DESCRIPTION || flag == SEQINFO_ORGANISM ||
      flag == SEQINFO_FRAGMENT || flag == SEQINFO_CIRCULAR) {
    t = (isfp->entry_seqno == 1 ? isfp->fout_descr1 : isfp->fout_descr2);
    if (t != NULL) {
      for (s=t; *s && *s != '>'; s++) ;
      parse_oneline(&info, t, s, flag);
    }
  }

  if (allflag || flag == SEQINFO_FRAGMENT || flag == SEQINFO_STARTPOS) {
    pattern = (isfp->entry_seqno == 1 ? "\nQuery:" : "\nSbjct:");
    for (s=entry; s < entry + len; s++)
      if (strncmp(s, pattern, 7) == 0)
        break;

    if (s < entry + len) {
      totallen = (isfp->entry_seqno == 1 ? isfp->fout_len1 : isfp->fout_len2);
      fragstart = myatoi(s+7, 10 ,'0');
      if ((allflag || flag == SEQINFO_FRAGMENT) && 
          (fragstart > 1 || (totallen && isfp->entry_truelen < totallen)))
        set_fragment(&info, 1);

      if ((allflag || flag == SEQINFO_STARTPOS) && 
          (fragstart > 1 || (totallen && isfp->entry_truelen < totallen)))
        set_fragstart(&info, fragstart);
    }
  }

  if (allflag || flag == SEQINFO_COMMENT) {
    sprintf(buffer, "From %s output alignment of:\n", isfp->fout_progname);
    for (end=buffer; *end; end++) ;
    add_comment(&info, buffer, end, 0, 0);

    if (isfp->fout_descr1 != NULL)
      sprintf(buffer, "   >%s\nand", isfp->fout_descr1);
    else
      strcpy(buffer, "   >Unknown sequence\nand");

    for (end=buffer; *end; end++) ;
    add_comment(&info, buffer, end, 0, 0);

    if (isfp->fout_descr2 != NULL) {
      s = isfp->fout_descr2;
      startflag = 1;
      while (*s) {
        if (*s == '>') s++;
        maxsize = (startflag ? 70 : 60);
        for (t=s++,i=1; *s && i < maxsize && *s != '>'; s++,i++) ;
        if (startflag) {
          buffer[3] = '>';
          pos = 4;
        }
        else {
          strcpy(buffer+3, "           ");
          pos = 14;
        }

        if (!*s || *s == '>') {
          size = s - t;
          startflag = 1;
        }
        else {
          for (s=t+60; s > t && !isspace(*s); s--) ;
          for (s2=s; *s2 && isspace(*s2); s2++) ;
          while (s > t && isspace(*(s-1))) s--;
          if (s == t) {
            s = t + 60;
            size = 60;
          }
          else {
            size = s - t;
            s = s2;
          }
          startflag = 0;
        }

        memcpy(buffer+pos, t, size);
        buffer[pos+size] = '\0';
        add_comment(&info, buffer, buffer + pos + size, 0, 0);
      }
    }
  
    add_comment(&info, buffer, buffer + 1, 0, 0);

    add_comment(&info, isfp->fp_entrystart, isfp->fp_seqstart, 0, 0);
  }

  /*
   * Finish the last fields in the INFO structure.
   */
BOUT_GI_END:
  if (allflag)
    add_retrieval(&info, 3, isfp->fout_progname);

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int phyint_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int allflag, status;
  char *s;
  INFO info;

  while (isfp->malign_seqno < isfp->entry_seqno) {
    for (s=isfp->fp_seqstart; isspace(*s); s++) ;
    for (s++; *s != '\n'; s++) ;
    isfp->fp_seqstart = s + 1;
    isfp->malign_seqno++;
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }
  }

  if (allflag || flag == SEQINFO_DESCRIPTION)
    add_description(&info, isfp->fp_seqstart, isfp->fp_seqstart+10);

  if (flag == SEQINFO_ALLINFO) {
    for (s=entry; *s != '\n'; s++) ;
    add_comment(&info, entry, s, 0, 0);
  }

  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "PHYLIP-Int");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int physeq_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int seqpos, seqlen, allflag, status;
  char *s, *end;
  INFO info;

  while (isfp->malign_seqno < isfp->entry_seqno) {
    s = isfp->fp_seqstart+10;
    end = isfp->fp_entryend;
    seqlen = isfp->entry_seqlen;
    for (seqpos=0; s < end && seqpos < seqlen; s++)
      if (!(isspace(*s) || isdigit(*s)))
        seqpos++;
    for ( ; *s != '\n'; s++) ;
    isfp->fp_seqstart = s + 1;
    isfp->malign_seqno++;
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }
  }

  if (allflag || flag == SEQINFO_DESCRIPTION)
    add_description(&info, isfp->fp_seqstart, isfp->fp_seqstart+10);

  if (flag == SEQINFO_ALLINFO) {
    for (s=entry; *s != '\n'; s++) ;
    add_comment(&info, entry, s, 0, 0);
  }

  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "PHYLIP-Seq");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int clustal_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int allflag, status;
  char *s, *end;
  INFO info;

  s = isfp->fp_seqstart;
  end = isfp->fp_entryend;

  if (isfp->malign_seqno < isfp->entry_seqno) {
    while (isfp->malign_seqno < isfp->entry_seqno) {
      for ( ; s < end && *s != '\n'; s++) ;
      s++;
      isfp->malign_seqno++;
    }
    isfp->fp_seqstart = s;
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }
  }

  if (allflag || flag == SEQINFO_DESCRIPTION)
    add_description(&info, isfp->fp_seqstart, isfp->fp_seqstart+15);

  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "Clustalw");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}


static int asn_getinfo(INTSEQFILE *isfp, char *entry, int len, int flag)
{
  int status, exists, count, alpha, oldpe, allflag;
  char *s, *t, datestr[12];
  char *idstr, *idend, *destr, *deend, *pirname, *pnend, *spname, *spend;
  char *gbname, *gbend, *emname, *emend, *otname, *otend, *piracc, *paend;
  char *spacc, *spaend, *gbacc, *gbaend, *emacc, *emaend, *otacc, *otaend;
  char *pdbmol, *pmend, *gistr, *gisend, *giim, *giimend, *name, *nameend;
  char *title, *titleend, *org, *orgend, *common, *commonend, *pirorg, *dend;
  char *poend, *pdbcomp, *pcend, *pdbsrc, *psend, *comment, *cmend, *date;
  char *cdate, *cdend, *udate, *udend, *gbdate, *gbdend, *gbedate, *gbeend;
  char *emcdate, *emcend, *emudate, *emuend, *pirdate, *pirend, *spcdate;
  char *spcend, *spudate, *spuend, *spadate, *pdbdate, *pdbend, *dbjacc;
  char *gibbs, *gibbsend, *gibbm, *gibbmend, *dbjname, *dbjend, *dbjaend;
  char *prfname, *prfend, *prfacc, *prfaend, *molstr, *molend, *dnastr;
  char *rnastr, *partstr, *topstr, *topend, *inststr, *instend;
  INFO info;

  for (s=entry; s < entry + len && isspace(*s); s++) ;
  if (!mystreq(s, 'S', "SEQ ")) {
    if (isfp->filename && isfp->filename[0]) {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("%s, entry %d:  Invalid format of ASN.1 "
                              "Bioseq-set.seq-set.seq entry.\n",
                              isfp->filename, isfp->entry_count));
    }
    else {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("seqfparseent:  Invalid format of ASN.1 "
                              "Bioseq-set.seq-set.seq entry.\n"));
    }
  }

  start_info(&info, isfp, flag);
  allflag = (flag == SEQINFO_ALL || flag == SEQINFO_ALLINFO);

  if (allflag) {
    set_filename(&info, isfp->filename);
    if (isfp->db_name != NULL)
      set_dbname(&info, isfp->db_name);
    set_format(&info, seqfformat(isfp, 0));

    if (!info.error) {
      info.info->entryno = isfp->entry_count;
      info.info->seqno = isfp->entry_seqno;
      info.info->numseqs = isfp->entry_numseqs;
    }
  }

  if ((allflag || flag == SEQINFO_ALPHABET) && isfp->db_alpha != NULL)
    set_alphabet(&info, get_alphabet(isfp->db_alpha, NULL));

  if (((allflag || flag == SEQINFO_TRUELEN) && !isfp->iflag_truelen) ||
      ((allflag || flag == SEQINFO_RAWLEN) && !isfp->iflag_rawlen)) {
    status = (*file_table[isfp->format].getseq_fn)(isfp, GETSEQ_LENGTHS);
    if (status != STATUS_OK && status != STATUS_WARNING) {
      finish_info(&info, isfp);
      return status;
    }
    if (!info.error) {
      info.info->truelen = isfp->entry_truelen;
      info.info->rawlen = isfp->entry_rawlen;
    }

    if (flag == SEQINFO_RAWLEN || flag == SEQINFO_TRUELEN)
      goto ASN_GI_END;
  }

  /*
   * Get the major pieces of the "seq" record, namely the "id" and
   * "descr" sub-records.
   */
  idstr = destr = inststr = NULL;
  oldpe = pe_flag;
  pe_flag = PE_NONE;
  status = asn_parse(entry, entry+len,
                     "seq.id", &idstr, &idend,
                     "seq.descr", &destr, &deend,
                     "seq.inst", &inststr, &instend,
                     NULL);
  pe_flag = oldpe;
  if (status == -1) {
    if (isfp->filename && isfp->filename[0]) {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("%s, entry %d:  Invalid format of ASN.1 "
                              "Bioseq-set.seq-set.seq entry.\n",
                              isfp->filename, isfp->entry_count));
    }
    else {
      raise_error(E_PARSEERROR, return STATUS_ERROR,
                  print_error("seqfparseent:  Invalid format of ASN.1 "
                              "Bioseq-set.seq-set.seq entry.\n"));
    }
  }

  /*
   * Parse the information in the `id' sub-record.
   */
  if (idstr != NULL && (allflag || flag == SEQINFO_IDLIST)) {
    pirname = spname = gbname = emname = otname = prfname = dbjname = NULL;
    piracc = spacc = gbacc = emacc = otacc = prfacc = dbjacc = NULL;
    pdbmol = gistr = giim = gibbs = gibbm = NULL;

    oldpe = pe_flag;
    pe_flag = PE_NONE;
    status = asn_parse(idstr, idend,
                       "id.pir.name", &pirname, &pnend,
                       "id.swissprot.name", &spname, &spend,
                       "id.genbank.name", &gbname, &gbend,
                       "id.embl.name", &emname, &emend,
                       "id.ddbj.name", &dbjname, &dbjend,
                       "id.prf.name", &prfname, &prfend,
                       "id.other.name", &otname, &otend,
                       "id.pir.accession", &piracc, &paend,
                       "id.swissprot.accession", &spacc, &spaend,
                       "id.genbank.accession", &gbacc, &gbaend,
                       "id.embl.accession", &emacc, &emaend,
                       "id.ddbj.accession", &dbjacc, &dbjaend,
                       "id.prf.accession", &prfacc, &prfaend,
                       "id.other.accession", &otacc, &otaend,
                       "id.pdb.mol", &pdbmol, &pmend,
                       "id.gi", &gistr, &gisend,
                       "id.giim.id", &giim, &giimend,
                       "id.gibbsq", &gibbs, &gibbsend,
                       "id.gibbmt", &gibbm, &gibbmend,
                       NULL);
    pe_flag = oldpe;
    if (status == -1) {
      if (isfp->filename && isfp->filename[0]) {
        raise_error(E_PARSEERROR, return STATUS_ERROR,
                    print_error("%s, entry %d:  Invalid format of ASN.1 "
                                "Bioseq-set.seq-set.seq entry.\n",
                                isfp->filename, isfp->entry_count));
      }
      else {
        raise_error(E_PARSEERROR, return STATUS_ERROR,
                    print_error("seqfparseent:  Invalid format of ASN.1 "
                                "Bioseq-set.seq-set.seq entry.\n"));
      }
    }

    /*
     * Set the id and accession values.
     */
    if (gbname != NULL)
      add_id(&info, "gb", gbname+4, gbend);
    if (gbacc != NULL)
      add_id(&info, "acc", gbacc+9, gbaend);

    if (pirname != NULL)
      add_id(&info, "pir", pirname+4, pnend);
    if (piracc != NULL)
      add_id(&info, "acc", piracc+9, paend);

    if (emname != NULL)
      add_id(&info, "embl", emname+4, emend);
    if (emacc != NULL)
      add_id(&info, "acc", emacc+9, emaend);

    if (spname != NULL)
      add_id(&info, "sp", spname+4, spend);
    if (spacc != NULL)
      add_id(&info, "acc", spacc+9, spaend);

    if (dbjname != NULL)
      add_id(&info, "ddbj", dbjname+4, dbjend);
    if (dbjacc != NULL)
      add_id(&info, "acc", dbjacc+9, dbjaend);

    if (prfname != NULL)
      add_id(&info, "prf", prfname+4, prfend);
    if (prfacc != NULL)
      add_id(&info, "acc", prfacc+9, prfaend);

    if (otname != NULL)
      add_id(&info, "oth", otname+4, otend);
    if (otacc != NULL)
      add_id(&info, "acc", otacc+9, otaend);

    if (pdbmol != NULL)
      add_id(&info, "pdb", pdbmol+3, pmend);

    if (gistr != NULL)
      add_id(&info, "gi", gistr+2, gisend);
    if (giim != NULL)
      add_id(&info, "giim", giim+2, giimend);
    if (gibbs != NULL)
      add_id(&info, "bbs", gibbs+6, gibbsend);
    if (gibbm != NULL)
      add_id(&info, "bbm", gibbm+6, gibbmend);
  }

  if (destr != NULL && (allflag || flag == SEQINFO_DATE)) {
    /*
     * Look for just the date.
     */
    cdate = udate = gbdate = gbedate = emcdate = emudate = NULL;
    pirdate = spcdate = spudate = spadate = pdbdate = NULL;

    oldpe = pe_flag;
    pe_flag = PE_NONE;
    status = asn_parse(destr, deend,
                       "descr.create-date", &cdate, &cdend,
                       "descr.update-date", &udate, &udend,
                       "descr.genbank.date", &gbdate, &gbdend,
                       "descr.genbank.entry-date", &gbedate, &gbeend,
                       "descr.embl.creation-date", &emcdate, &emcend,
                       "descr.embl.update-date", &emudate, &emuend,
                       "descr.pir.date", &pirdate, &pirend,
                       "descr.sp.created", &spcdate, &spcend,
                       "descr.sp.sequpd", &spudate, &spuend,
                       "descr.sp.annotupd", &spadate, &spaend,
                       "descr.pdb.deposition", &pdbdate, &pdbend,
                       NULL);
    pe_flag = oldpe;
    if (status == -1) {
      if (isfp->filename && isfp->filename[0]) {
        raise_error(E_PARSEERROR, return STATUS_ERROR,
                    print_error("%s, entry %d:  Invalid format of ASN.1 "
                                "Bioseq-set.seq-set.seq entry.\n",
                                isfp->filename, isfp->entry_count));
      }
      else {
        raise_error(E_PARSEERROR, return STATUS_ERROR,
                    print_error("seqfparseent:  Invalid format of ASN.1 "
                                "Bioseq-set.seq-set.seq entry.\n"));
      }
    }

    exists = 1;
    date = dend = NULL;
    if (cdate != NULL)  { date = cdate; dend = cdend; }
    else if (udate != NULL)  { date = udate; dend = udend; }
    else if (gbedate != NULL)  { date = gbedate; dend = gbeend; }
    else if (emudate != NULL)  { date = emudate; dend = emuend; }
    else if (emcdate != NULL)  { date = emcdate; dend = emcend; }
    else if (spadate != NULL)  { date = spadate; dend = spaend; }
    else if (spudate != NULL)  { date = spudate; dend = spuend; }
    else if (spcdate != NULL)  { date = spcdate; dend = spcend; }
    else if (pdbdate != NULL)  { date = pdbdate; dend = pdbend; }
    else {
      exists = 0;
      if (pirdate != NULL) {
        for (s=pirdate+4; s < pirend && *s != '"'; s++) ;
        for (t=++s; t < pirend && *t != '"'; t++) {
          if (*t == '#') {
            while (t < pirend && !isspace(*t)) t++;
            while (t < pirend && isspace(*t)) t++;
            if (t+11 < pirend)
              s = t;
          }
        }
        set_date(&info, s, s+11);
      }
      else if (gbdate != NULL)
        set_date(&info, gbdate+4, gbdend);
    }

    if (exists) {
      for (s=date; s < dend && !isspace(*s); s++) ;
      for ( ; s < dend && isspace(*s); s++) ;
      if (s + 3 < dend && *s == 's' && s[1] == 't') {
        if (s[2] == 'r') {
          for (s+=3; s < dend && *s != '"'; s++) ;
          for (t=++s; t < dend && *t != '"'; t++) ;
          if (t < dend)
            set_date(&info, s, t);
        }
        else if (s[2] == 'd') {
          s += 3;
          count = 0;
          while (s < dend && count < 3) {
            while (s < dend && !isalpha(*s)) s++;
            if (s == dend)
              break;

            if (*s == 'y' && strncmp(s, "year ", 5) == 0) {
              for (s+=5; s < dend && isspace(*s); s++) ;
              if (s + 4 < dend && isdigit(*s) && isdigit(s[1]) &&
                  isdigit(s[2]) && isdigit(s[3])) {
                datestr[7] = s[0];
                datestr[8] = s[1];
                datestr[9] = s[2];
                datestr[10] = s[3];
                count++;
              }
            }
            else if (*s == 'm' && strncmp(s, "month ", 6) == 0) {
              for (s+=6; s < dend && isspace(*s); s++) ;
              if (s + 2 < dend && isdigit(*s) &&
                  (isspace(s[1]) || isdigit(s[1]))) {
                t = months[(isspace(s[1]) ? s[0] - '0' : 10 + (s[1] - '0'))];
                datestr[3] = t[0];
                datestr[4] = t[1];
                datestr[5] = t[2];
                count++;
              }
            }
            else if (*s == 'd' && strncmp(s, "day ", 4) == 0) {
              for (s+=4; s < dend && isspace(*s); s++) ;
              if (s + 2 < dend && isdigit(*s) &&
                  (isspace(s[1]) || isdigit(s[1]))) {
                if (isspace(s[1])) {
                  datestr[0] = '0';
                  datestr[1] = s[0];
                }
                else {
                  datestr[0] = s[0];
                  datestr[1] = s[1];
                }
                count++;
              }
            }
          }
          if (count == 3) {
            datestr[2] = datestr[6] = '-';
            datestr[11] = '\0';
            set_date(&info, datestr, datestr+11);
          }
        }
      }
    }
  }

  /*
   * Get the description, organism and comment information.
   */
  if (destr != NULL && (allflag || flag == SEQINFO_DESCRIPTION ||
                        flag == SEQINFO_COMMENT || flag == SEQINFO_IDLIST ||
                        flag == SEQINFO_ORGANISM || flag == SEQINFO_FRAGMENT ||
                        flag == SEQINFO_ALPHABET)) {
    name = title = org = comment = common = pirorg = pdbcomp = NULL;
    pdbsrc = molstr = dnastr = rnastr = partstr = NULL;

    oldpe = pe_flag;
    pe_flag = PE_NONE;
    status = asn_parse(destr, deend,
                       "descr.name", &name, &nameend,
                       "descr.comment", &comment, &cmend,
                       "descr.title", &title, &titleend,
                       "descr.org.taxname", &org, &orgend,
                       "descr.org.common", &common, &commonend,
                       "descr.pir.source", &pirorg, &poend,
                       "descr.pdb.compound", &pdbcomp, &pcend,
                       "descr.pdb.source", &pdbsrc, &psend,
                       "descr.mol-type", &molstr, &molend,
                       "descr.modif.dna", &dnastr, NULL,
                       "descr.modif.rna", &rnastr, NULL,
                       "descr.modif.partial", &partstr, NULL,
                       NULL);
    pe_flag = oldpe;
    if (status == -1) {
      if (isfp->filename && isfp->filename[0]) {
        raise_error(E_PARSEERROR, return STATUS_ERROR,
                    print_error("%s, entry %d:  Invalid format of ASN.1 "
                                "Bioseq-set.seq-set.seq entry.\n",
                                isfp->filename, isfp->entry_count));
      }
      else {
        raise_error(E_PARSEERROR, return STATUS_ERROR,
                    print_error("seqfparseent:  Invalid format of ASN.1 "
                                "Bioseq-set.seq-set.seq entry.\n"));
      }
    }

    /*
     * Check the alphabet and isfragment information.
     */
    if (partstr != NULL && (allflag || flag == SEQINFO_FRAGMENT))
      set_fragment(&info, 1);

    if (allflag || flag == SEQINFO_ALPHABET) {
      if (dnastr != NULL)
        set_alphabet(&info, DNA);
      else if (rnastr != NULL)
        set_alphabet(&info, RNA);
      else if (molstr != NULL) {
        for (s=molstr+8; s < molend && isspace(*s); s++) ;
        for (t=s; s < molend && !isspace(*s) && *s != ','; s++) ;
        if ((alpha = get_alphabet(t, s)) != UNKNOWN)
          set_alphabet(&info, alpha);
      }
    }

    /*
     * Add the description, if it's there.
     */
    if (allflag || flag == SEQINFO_DESCRIPTION) {
      if (title != NULL)
        add_description(&info, title+5, titleend);
      else if (pdbcomp != NULL) {
        for (s=pdbcomp+8; s < pcend && *s != '"'; s++) ;
        if (s < pcend) {
          for (t=++s; s < pcend && *s != '"'; s++) ;
          if (s < pcend)
            add_description(&info, t, s);
        }
      }
      else if (name != NULL)
        add_description(&info, name+4, nameend);
    }

    /*
     * Add the organism name, if it's there.
     */
    if (allflag || flag == SEQINFO_ORGANISM) {
      if (org != NULL)
        add_organism(&info, org+7, orgend);
      else if (common != NULL)
        add_organism(&info, common+6, commonend);
      else if (pirorg != NULL)
        add_organism(&info, pirorg+6, poend);
      else if (pdbsrc != NULL) {
        for (s=pdbsrc+8; s < psend && *s != '"'; s++) ;
        if (s < psend) {
          for (t=++s; s < psend && *s != '"'; s++) ;
          if (s < psend)
            add_organism(&info, t, s);
        }
      }
    }

    if (allflag || flag == SEQINFO_COMMENT || flag == SEQINFO_IDLIST) {
      if (comment != NULL) {
        for (s=comment+7; s < cmend && isspace(*s); s++) ;
        if (*s == '"') s++;
        for (t=cmend; t > s && isspace(*(t-1)); t--) ;
        if (*(t-1) == '"') t--;

        parse_comment(&info, s, t, 1, 1, flag);

        while (asn_parse(cmend+1, deend,
                         "comment", &comment, &cmend, NULL) == 1) {
          for (s=comment+7; s < cmend && isspace(*s); s++) ;
          if (*s == '"') s++;
          for (t=cmend; t > s && isspace(*(t-1)); t--) ;
          if (*(t-1) == '"') t--;

          parse_comment(&info, s, t, 1, 1, flag);
        }
      }
    }
  }

  /*
   * Check the information in the "inst" record.
   */
  if (inststr != NULL && 
      (allflag || flag == SEQINFO_ALPHABET || flag == SEQINFO_CIRCULAR)) {
    molstr = topstr = NULL;

    oldpe = pe_flag;
    pe_flag = PE_NONE;
    status = asn_parse(inststr, instend,
                       "inst.mol", &molstr, &molend,
                       "inst.topology", &topstr, &topend,
                       NULL);
    pe_flag = oldpe;
    if (status == -1) {
      if (isfp->filename && isfp->filename[0]) {
        raise_error(E_PARSEERROR, return STATUS_ERROR,
                    print_error("%s, entry %d:  Invalid format of ASN.1 "
                                "Bioseq-set.seq-set.seq entry.\n",
                                isfp->filename, isfp->entry_count));
      }
      else {
        raise_error(E_PARSEERROR, return STATUS_ERROR,
                    print_error("seqfparseent:  Invalid format of ASN.1 "
                                "Bioseq-set.seq-set.seq entry.\n"));
      }
    }

    if (molstr != NULL && (allflag || flag == SEQINFO_ALPHABET)) {
      for (s=molstr+3; s < molend && isspace(*s); s++) ;
      for (t=s; s < molend && !isspace(*s) && *s != ','; s++) ;
      if ((alpha = get_alphabet(t, s)) != UNKNOWN)
        set_alphabet(&info, alpha);
    }

    if (topstr != NULL && (allflag || flag == SEQINFO_CIRCULAR)) {
      for (s=topstr+8; s < topend && isspace(*s); s++) ;
      if (mystreq(s, 'C', "CIRCULAR"))
        set_circular(&info, 1);
    }
  }

ASN_GI_END:
  /*
   * Finish the last fields in the INFO structure.
   */
  if (flag == SEQINFO_ALL) {
    if (isfp->db_name == NULL)
      add_retrieval(&info, 1, "ASN.1");
    else
      add_retrieval(&info, 2, isfp->db_name);
  }

  finish_info(&info, isfp);
  memory_error(info.error, return STATUS_FATAL);
  return STATUS_OK;
}




/*
 *
 *
 * Sequence Output Section
 *
 *
 *
 */

#define PLAIN -1
#define RAW -2

static int guessalpha(char *seq, int seqlen, int *align_out, int *truelen_out)
{
  int i, dna, u, t, others, newlines, gaps, truelen;
  char ch;

  dna = u = t = gaps = others = newlines = truelen = 0;
  for (i=0; i < seqlen; i++) {
    if (isalpha(seq[i]))
      truelen++;

    ch = toupper(seq[i]);
    if (ch == 'A' || ch == 'G' || ch == 'C' || ch == 'T' || ch == 'U') {
      dna++;
      if (ch == 'U')
        u++;
      else if (ch == 'T')
        t++;
    }
    else if (ch == '-' || ch == '.' || ch == '(' || ch == ')' || ch == ',')
      gaps++;
    else if (ch == '\n')
      newlines++;
    else if (!isalpha(ch) && !isspace(ch))
      others++;
  }

  if (truelen_out)
    *truelen_out = truelen;

  if (newlines == 0 && ((float) dna / (float) seqlen) >= 0.85) {
    if (align_out)
      *align_out = (others + gaps > 0);
    return (t == 0 && u != 0 ? RNA : DNA);
  }
  else if (others + newlines == 0) {
    if (align_out)
      *align_out = (gaps > 0);
    return PROTEIN;
  }
  else
    return (newlines ? PLAIN : RAW);
}

static int putline(FILE *fp, int flag, int pos, char *string, int width,
                   char *filler, int addcontflag)
{
  int len, flen, contflag;
  char *s, *t;

  if (width == 0) {
    if (flag)
      fputc(' ', fp);

    for (s=string; *s; s++) {
      for (t=s; *s && *s != '\n'; s++) ;
      if (t != s)
        fwrite(t, 1, s - t, fp);

      if (!*s || !s[1])
        break;

      fputc('\n', fp);
      fputs(filler, fp);
    }

    return 0;
  }
  else {
    flen = strlen(filler);
    if (flag) {
      if (pos+1 >= width) {
        fputc('\n', fp);
        fputs(filler, fp);
        pos = flen;
        if (addcontflag) {
          fputs("  ", fp);
          pos += 2;
        }
      }
      else {
        fputc(' ', fp);
        pos++;
      }
    }

    s = string;
    while (*s) {
      for (t=s,len=pos; *s && *s != '\n'; s++, len++)
        if (len == width)
          break;

      if (*s && *s != '\n') {
        if (isspace(*s))
          while (s > t && isspace(*(s-1))) s--;
        else {
          while (s > t && !isspace(*(s-1))) s--;
          if (s == t && !flag)
            while (*s && !isspace(*s)) s++;
          else
            while (s > t && isspace(*(s-1))) s--;
        }
      }
      
      if (t != s) {
        fwrite(t, 1, s - t, fp);
        pos += s - t;
      }

      contflag = (addcontflag ? flag : 0);
      while (*s && isspace(*s) && *s != '\n') s++;
      if (!*s)
        break;
      else if (*s == '\n') {
        if (!s[1])
          break;
        s++;
      }
      else
        contflag = addcontflag;

      fputc('\n', fp);
      fputs(filler, fp);
      pos = flen;
      flag = 0;
      if (contflag) {
        fputs("  ", fp);
        pos += 2;
      }
    }

    return pos;
  }
}


static void put_oneline(FILE *fp, SEQINFO *info, int truelen, int alpha,
                        char *idlist)
{
  int flag;
  char *s;

  flag = 0;
  if (idlist && idlist[0]) {
    for (s=idlist; *s; s++)
      fputc(*s, fp);
    flag = 1;
  }

  if (info->description && info->description[0]) {
    if (flag)
      fputc(' ', fp);
    fputs(info->description, fp);
  }
  if (info->organism && info->organism[0]) {
    fputs(" - ", fp);
    fputs(info->organism, fp);
  }

  if (truelen > 0) {
    fprintf(fp, ", %d ", truelen);
    fputs((alpha == DNA || alpha == RNA 
              ? "bp" : (alpha == PROTEIN ? "aa" : "ch")), fp);

    if (alpha == RNA || info->iscircular || info->isfragment) {
      fputs(" (", fp);
      flag = 0;
      if (info->iscircular) {
        fputs("circular", fp);
        flag = 1;
      }
      if (alpha == RNA) {
        if (flag)
          fputc(' ', fp);
        fputs("RNA", fp);
        flag = 1;
      }
      if (info->isfragment) {
        if (info->fragstart > 0) {
          if (flag)
            fputs(", ", fp);
          fprintf(fp, "f. %d-%d", info->fragstart,
                  info->fragstart + truelen - 1);
        }
        else {
          if (flag)
            fputc(' ', fp);
          fputs("fragment", fp);
        }
      }
      fputc(')', fp);
    }

    fputc('.', fp);
  }
}

static void simple_putseq(FILE *fp, char *seq, int seqlen, int alpha,
                          int align_flag, int prettyflag)
{
  int i, j;

  if (seqlen == 0)
    return;

  if (!align_flag && (alpha == DNA || alpha == RNA || alpha == PROTEIN)) {
    i = j = 0;
    if (prettyflag != 1)
      fputs("  ", fp);
    while (1) {
      fputc(seq[i++], fp);
      if (i == seqlen)
        break;

      if (++j == 60) {
        fputc('\n', fp);
        if (prettyflag != 1)
          fputs("  ", fp);
        j = 0;
      }
      else if (prettyflag != 1 && j % 10 == 0)
        fputc(' ', fp);
    }
  }
  else if (align_flag || alpha == RAW) {
    i = j = 0;
    while (1) {
      fputc(seq[i++], fp);
      if (i == seqlen)
        break;

      if (++j == 60) {
        fputc('\n', fp);
        j = 0;
      }
    }
  }
  else if (alpha == PLAIN) {
    fwrite(seq, 1, seqlen, fp);
  }
}


static int gcg_checksum(char *seq, int seqlen, int seed)
{
  int i, checksum;
  char ch;

  checksum = seed;
  for (i=0; i < seqlen; i++) {
    ch = toupper((seq[i] == '-' ? '.' : seq[i]));
    checksum += (i % 57 + 1) * ch;
  }

  return (checksum % 10000);
}


static int gcg_checksum2(char *seq, int seqlen, int maxlen, int seed)
{
  int i, checksum;
  char ch;

  checksum = seed;
  for (i=0; i < seqlen; i++) {
    ch = toupper((seq[i] == '-' ? '.' : seq[i]));
    checksum += (i % 57 + 1) * ch;
  }
  for ( ; i < maxlen; i++)
    checksum += (i % 57 + 1) * '.';

  return (checksum % 10000);
}


static void putgcgseq(FILE *fp, char *seq, int seqlen, SEQINFO *info)
{
  int i, j, k, len, alpha, align_flag, count;
  char *date, *s, buffer[32];

  fputc('\n', fp);

  /*
   * Print the infoline.
   */
  len = seqfoneline(info, buffer, 20, 1);
  fputs("  ", fp);
  fputs(buffer, fp);
  fputs("  ", fp);
  fprintf(fp, "Length: %d  ", seqlen);

  date = get_today();
  for (i=1; i <= 12; i++)
    if (myncasecmp(date+3, months[i], 3) == 0)
      break;
  if (i <= 12)
    fprintf(fp, "%s %c%c, %s %s  ", gcg_full_months[i], date[0], date[1],
            date+7, date+12);

  if (info->alphabet != UNKNOWN)
    alpha = info->alphabet;
  else
    alpha = guessalpha(seq, seqlen, &align_flag, NULL);

  if (alpha == RNA || alpha == DNA)
    fputs("Type: N  ", fp);
  else if (alpha == PROTEIN)
    fputs("Type: P  ", fp);

  fprintf(fp, "Check: %d  ..\n\n", gcg_checksum(seq, seqlen, 0));

  /*
   * Print the gcg sequence lines.
   */
  for (i=0,count=1; i < seqlen; count+=50) {
    sprintf(buffer, "%8d  ", count);
    for (j=0,s=buffer+9; i < seqlen && j < 5; j++) {
      for (k=0; i < seqlen && k < 10; k++,i++)
        *s++ = (seq[i] == '-' ? '.' : seq[i]);
      *s++ = ' ';
    }
    *s++ = '\n';
    *s++ = '\n';
    *s = '\0';
    fputs(buffer, fp);
  }
}



static int raw_putseq(INTSEQFILE *isfp, char *seq, int seqlen, SEQINFO *info)
{
  fputs(seq, isfp->output_fp);
  return STATUS_OK;
}

static int plain_putseq(INTSEQFILE *isfp, char *seq, int seqlen, SEQINFO *info)
{
  simple_putseq(isfp->output_fp, seq, seqlen, PLAIN, 0, isfp->prettyflag);
  if (seq[seqlen-1] != '\n')
    fputc('\n', isfp->output_fp);
  return STATUS_OK;
}

static int genbank_putseq(INTSEQFILE *isfp, char *seq, int seqlen,
                          SEQINFO *info)
{
  int i, j, k, a, c, g, t2, u, o, count, alpha, flag, flag2;
  char ch, *s, *t, *id, *idend, buffer[128];
  char *gbid, *gbend, *shortid, *shend, *nid_start, *nid_end;
  FILE *fp = isfp->output_fp;

  /*
   * Look for either a "gb" identifier or a short, non-accession identifier.
   * If found, use that as the entry ID.  Otherwise, fill in string
   * "(below)" or "Unknown" whether there are any non-accession identifiers.
   *
   * If an ID was used, remember it so that it won't be duplicated when 
   * the "Cross-Refs:" line is printed.
   */
  id = NULL;
  if (info->idlist && info->idlist[0]) {
    gbid = gbend = shortid = shend = NULL;
    flag = 0;
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (!gbid && mystreq(t, 'G', "GB:")) {
        gbid = t;
        gbend = s;
        break;
      }
      if (!mystreq(t, 'A', "ACC:") && !mystreq(t, 'N', "NID:")) {
        flag = 1;
        if (!shortid && s - t <= 10) {
          shortid = t;
          shend = s;
        }
      }
      if (*s) s++;
    }
    if (gbid) {
      id = gbid + 3;
      idend = gbend;
    }
    else if (shortid) {
      id = shortid;
      idend = shend;
    }
    else if (flag) {
      id = "(below)";
      idend = id + 7;
    }
    else {
      id = "Unknown";
      idend = id + 7;
    }
  }
  else {
    id = "Unknown";
    idend = id + 7;
  }
  ch = *idend; *idend = '\0';
  fprintf(fp, "LOCUS       %-10s", id);
  *idend = ch;

  /*
   * Print the rest of the LOCUS line.
   */
  alpha = (info->alphabet != UNKNOWN 
             ? info->alphabet : guessalpha(seq, seqlen, NULL, NULL));

  fprintf(fp, "%7d %s %3s%-4s  %-10s%3s       %-11s\n",
          seqlen,                                         /* seq. length */
          (alpha == PROTEIN ? "aa" : "bp"),               /* char. type */
          "",                                             /* strand type */
          (alpha == RNA ? "RNA" :                         /* seq. type */
            (alpha == DNA ? "DNA" : 
              (alpha == PROTEIN ? "PRT" : ""))),
          (info->iscircular ? "circular" : ""),           /* linear/circular */
          "UNC",                                          /* category */
          (info->date && info->date[0] ? info->date 
                                       : "01-JAN-0000")); /* date */

  /*
   * Print the DEFINITION, ACCESSION, NID and ORGANISM lines.
   */
  if (info->description && info->description[0]) {
    fputs("DEFINITION  ", fp);
    putline(fp, 0, 12, info->description, 80, "            ", 0);
    fputc('.', fp);
    fputc('\n', fp);
  }

  flag = 0;
  nid_start = nid_end = NULL;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (mystreq(t, 'A', "ACC:")) {
        if (!flag) {
          fputs("ACCESSION   ", fp);
          flag = 12;
        }
        else if (flag + (s - t - 4 + 1) > 80) {
          fputs("\n            ", fp);
          flag = 12;
        }
        else {
          fputc(' ', fp);
          flag++;
        }

        fwrite(t+4, 1, s - t - 4, fp);
        flag += s - t - 4;
      }
      else if (mystreq(t, 'N', "NID:")) {
        nid_start = t;
        nid_end = s;
      }
      if (*s) s++;
    }
  }
  if (flag)
    fputc('\n', fp);

  if (nid_start != NULL) {
    fputs("NID         ", fp);
    fwrite(nid_start + 4, 1, nid_end - nid_start - 4, fp);
    fputc('\n', fp);
  }

  if (info->organism && info->organism[0]) {
    fputs("SOURCE      .\n  ORGANISM  ", fp);
    putline(fp, 0, 12, info->organism, 80, "            ", 0);
    fputc('\n', fp);
  }

  /*
   * Print the comment section:  actual comments, Cross-References, history.
   */
  flag = 0;
  if (info->comment && info->comment[0]) {
    fputs("COMMENT     ", fp);
    putline(fp, 0, 12, info->comment, 80, "            ", 1);
    fputc('\n', fp);
    flag = 1;
  }

  flag2 = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (!(id && (t == id || t+3 == id)) && 
          !mystreq(t, 'A', "ACC:") && !mystreq(t, 'N', "NID:")) {
        if (!flag2) {
          if (!flag) {
            fputs("COMMENT     SEQIO Refs: ", fp);
            flag = 1;
          }
          else
            fputs("            \n            SEQIO Refs: ", fp);
          flag2 = 24;
        }
        else if (flag2 + (s - t) + 1 >= 80) {
          fputs("\n            SEQIO Refs: ", fp);
          flag2 = 24;
        }
        else {
          fputc('|', fp);
          flag2++;
        }

        fwrite(t, 1, s - t, fp);
        flag2 += s - t;
      }
      if (*s) s++;
    }
  }
  if (flag2)
    fputc('\n', fp);

  if (info->history && info->history[0]) {
    if (!flag) {
      fputs("COMMENT     ", fp);
      putline(fp, 0, 12, info->history, 80, "            ", 1);
      fputc('\n', fp);
      flag = 1;
    }
    else {
      if (!flag2)
        fputs("            \n", fp);
      fputs("            ", fp);
      putline(fp, 0, 12, info->history, 80, "            ", 1);
      fputc('\n', fp);
    }
  }

  /*
   * Count the bases, and possibly print a base count.
   */
  if (alpha == DNA || alpha == RNA) {
    a = c = g = t2 = u = o = 0;
    for (i=0; i < seqlen; i++) {
      switch (seq[i]) {
      case 'a': case 'A':  a++; break;
      case 'c': case 'C':  c++; break;
      case 'g': case 'G':  g++; break;
      case 't': case 'T':  t2++; break;
      case 'u': case 'U':  u++; break;
      default:             o++;
      }
    }
    if (alpha == RNA && !(u == 0 && t2 != 0))
      fprintf(fp, "BASE COUNT  %7d a%7d c%7d g%7d u", a, c, g, u);
    else
      fprintf(fp, "BASE COUNT  %7d a%7d c%7d g%7d t", a, c, g, t2);
    if (o)
      fprintf(fp, "%7d others", o);
    fputc('\n', fp);
  }

  /*
   * Print the sequence.
   */
  fputs("ORIGIN\n", fp);

  if (isfp->format == FORMAT_GCG)
    putgcgseq(fp, seq, seqlen, info);
  else {
    for (i=0,count=1; i < seqlen; count+=60) {
      sprintf(buffer, "   %6d", count);
      for (j=0,s=buffer+9; i < seqlen && j < 6; j++) {
        *s++ = ' ';
        for (k=0; i < seqlen && k < 10; k++)
          *s++ = seq[i++];
      }
      *s++ = '\n';
      *s = '\0';
      fputs(buffer, fp);
    }
    fputs("//\n", fp);
  }

  return STATUS_OK;
}


static int pir_putseq(INTSEQFILE *isfp, char *seq, int seqlen, SEQINFO *info)
{
  int i, j, count, pos, flag, flag2;
  char ch, *s, *t, *id, *idend;
  FILE *fp = isfp->output_fp;

  /*
   * Print a non-accession identifier (a "pir" ID if possible) and
   * whether the sequence is complete or a fragment.
   */
  id = idend = NULL;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (mystreq(t, 'P', "PIR:")) {
        id = t+4;
        idend = s;
        break;
      }
      else if (!id && !mystreq(t, 'A', "ACC:")) {
        id = t;
        idend = s;
      }
      if (*s) s++;
    }

    if (!id) {
      id = "UNKNWN";
      idend = id + 6;
    }
  }
  else {
    id = "UNKNWN";
    idend = id + 6;
  }
  ch = *idend; *idend = '\0';
  fprintf(fp, "ENTRY            %s       #type %s\n", id,
          (info->isfragment ? "fragment" : "complete"));
  *idend = ch;

  if (info->description && info->description[0]) {
    fputs("TITLE            ", fp);
    pos = putline(fp, 0, 17, info->description, 80, "                 ", 1);
    if (info->organism && info->organism[0]) {
      pos = putline(fp, 1, pos, "-", 80, "                 ", 1);
      pos = putline(fp, 1, pos, info->organism, 80, "                 ", 1);
    }
    if (info->isfragment)
      pos = putline(fp, 1, pos, "(fragment)", 80, "                 ", 1);
    fputc('\n', fp);
  }

  if (info->organism && info->organism[0]) {
    fputs("ORGANISM         #formal_name ", fp);
    putline(fp, 0, 30, info->organism, 80, "                 ", 1);
    fputc('\n', fp);
  }

  if (info->date && info->date[0])
    fprintf(fp, "DATE             %s\n", info->date);

  flag = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (mystreq(t, 'A', "ACC:")) {
        if (!flag) {
          fputs("ACCESSIONS       ", fp);
          flag = 17;
        }
        else if (flag + (s - t - 4 + 3) > 80) {
          fputs(";\n                   ", fp);
          flag = 19;
        }
        else {
          fputs("; ", fp);
          flag += 2;
        }

        fwrite(t+4, 1, s - t - 4, fp);
        flag += s - t - 4;
      }
      if (*s) s++;
    }
  }
  if (flag)
    fputc('\n', fp);

  /*
   * Print the comment section:  actual comments, Cross-References, history.
   */
  flag = 0;
  if (info->comment && info->comment[0]) {
    fputs("COMMENT    ", fp);
    putline(fp, 0, 11, info->comment, 80, "           ", 1);
    fputc('\n', fp);
    flag = 1;
  }

  flag2 = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (!(id && (t == id || t+4 == id)) && !mystreq(t, 'A', "ACC:")) {
        if (!flag2) {
          if (!flag) {
            fputs("COMMENT    SEQIO Refs: ", fp);
            flag = 1;
          }
          else
            fputs("           \n           SEQIO Refs: ", fp);
          flag2 = 23;
        }
        else if (flag2 + (s - t) + 1 >= 80) {
          fputs("\n           SEQIO Refs: ", fp);
          flag2 = 23;
        }
        else {
          fputc('|', fp);
          flag2 = 23;
        }

        fwrite(t, 1, s - t, fp);
        flag2 += s - t;
      }
      if (*s) s++;
    }
  }
  if (flag2)
    fputc('\n', fp);

  if (info->history && info->history[0]) {
    if (!flag) {
      fputs("COMMENT    ", fp);
      putline(fp, 0, 11, info->history, 80, "           ", 1);
      fputc('\n', fp);
      flag = 1;
    }
    else {
      if (!flag2)
        fputs("           \n", fp);
      fputs("           ", fp);
      putline(fp, 0, 11, info->history, 80, "           ", 1);
      fputc('\n', fp);
    }
  }

  fprintf(fp, "SUMMARY          #length %d\n", seqlen);
  fputs("SEQUENCE\n", fp);

  if (isfp->format == FORMAT_GCG)
    putgcgseq(fp, seq, seqlen, info);
  else {
    fputs("                5        10        15"
          "        20        25        30\n", fp);

    for (i=0,count=1; i < seqlen; count+=30) {
      fprintf(fp, "%7d", count);
      for (j=0; i < seqlen && j < 30; j++) {
        fputc(' ', fp);
        fputc(seq[i++], fp);
      }
      fputc('\n', fp);
    }
    fputs("///\n", fp);
  }
           
  return STATUS_OK;
}

static int embl_putseq(INTSEQFILE *isfp, char *seq, int seqlen, SEQINFO *info)
{
  int i, j, k, a, c, g, t2, u, o, alpha, count, flag, flag2, pos, epd_flag;
  char ch, *s, *t, *id, *idend, *nid_start, *nid_end;
  FILE *fp = isfp->output_fp;

  /*
   * Print a non-accession identifier (an "embl" ID if possible) and
   * whether the sequence is complete or a fragment.
   */
  epd_flag = 0;
  id = idend = NULL;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (mystreq(t, 'E', "EMBL:")) {
        id = t+5;
        idend = s;
        break;
      }
      else if (mystreq(t, 'E', "EPD:")) {
        epd_flag = 1;
        id = t+4;
        idend = s;
        break;
      }
      else if (!id && !mystreq(t, 'A', "ACC:") && !mystreq(t, 'N', "NID:")) {
        id = t;
        idend = s;
      }
      if (*s) s++;
    }
    if (!id) {
      id = "Unknown";
      idend = id + 7;
    }
  }
  else {
    id = "Unknown";
    idend = id + 7;
  }
  ch = *idend; *idend = '\0';
  fprintf(fp, "ID   %-10s", id);
  *idend = ch;

  alpha = (info->alphabet != UNKNOWN 
             ? info->alphabet : guessalpha(seq, seqlen, NULL, NULL));

  fputs(" converted; ", fp);
  if (info->iscircular)
    fputs("circular ", fp);
  fputs((alpha == RNA ? "RNA" 
                      : (alpha == DNA ? "DNA" 
                                      : (alpha == PROTEIN ? "PRT" : "UNK"))),
        fp);
  fprintf(fp, "; %s; %d %s.\n", (!epd_flag ? "UNC" : "EPD"), seqlen,
          (alpha == DNA || alpha == RNA ? "BP" 
                                        : (alpha == PROTEIN ? "AA" : "CH")));
  fputs("XX\n", fp);

  flag = 0;
  nid_start = nid_end = NULL;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (mystreq(t, 'A', "ACC:")) {
        if (!flag) {
          fputs("AC   ", fp);
          flag = 5;
        }
        else if (flag + (s - t - 4 + 2) > 80) {
          fputs("\nAC   ", fp);
          flag = 5;
        }
        else {
          fputc(' ', fp);
          flag++;
        }

        fwrite(t+4, 1, s - t - 4, fp);
        fputc(';', fp);
        flag += (s - t - 4) + 1;
      }
      else if (mystreq(t, 'N', "NID:")) {
        nid_start = t;
        nid_end = s;
      }
      if (*s) s++;
    }
  }
  if (flag)
    fputs("\nXX\n", fp);

  if (nid_start != NULL) {
    fputs("NI   ", fp);
    fwrite(nid_start + 4, 1, nid_end - nid_start - 4, fp);
    fputs("\nXX\n", fp);
  }

  if (info->date && info->date[0])
    fprintf(fp, "DT   %s\nXX\n", info->date);

  if (info->description && info->description[0]) {
    fputs("DE   ", fp);
    pos = putline(fp, 0, 5, info->description, 80, "DE   ", 0);
    if (info->isfragment)
      putline(fp, 1, pos, "(fragment)", 80, "DE   ", 0);
    fputs("\nXX\n", fp);
  }

  if (info->organism && info->organism[0]) {
    fputs("OS   ", fp);
    putline(fp, 0, 5, info->organism, 80, "OS   ", 0);
    fputs("\nXX\n", fp);
  }

  /*
   * Print the comment section:  actual comments, Cross-References, history.
   */
  flag = 0;
  if (info->comment && info->comment[0]) {
    fputs("CC   ", fp);
    putline(fp, 0, 5, info->comment, 80, "CC   ", 1);
    fputc('\n', fp);
    flag = 1;
  }

  flag2 = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (!(id && (t == id || t+4 == id || t+5 == id)) &&
          !mystreq(t, 'A', "ACC:") && !mystreq(t, 'N', "NID:")) {
        if (!flag2) {
          if (!flag) {
            fputs("CC   SEQIO Refs: ", fp);
            flag = 1;
          }
          else
            fputs("CC   \nCC   SEQIO Refs: ", fp);
          flag2 = 17;
        }
        else if (flag2 + (s - t) + 1 >= 80) {
          fputs("\nCC   SEQIO Refs: ", fp);
          flag2 = 17;
        }
        else {
          fputc('|', fp);
          flag2++;
        }

        fwrite(t, 1, s - t, fp);
        flag2 += s - t;
      }
      if (*s) s++;
    }
  }
  if (flag2)
    fputc('\n', fp);

  if (info->history && info->history[0]) {
    if (!flag) {
      fputs("CC   ", fp);
      putline(fp, 0, 5, info->history, 80, "CC   ", 1);
      fputc('\n', fp);
      flag = 1;
    }
    else {
      if (!flag2)
        fputs("CC   \n", fp);
      fputs("CC   ", fp);
      putline(fp, 0, 5, info->history, 80, "CC   ", 1);
      fputc('\n', fp);
    }
  }
  if (flag)
    fputs("XX\n", fp);

  fprintf(fp, "SQ   Sequence %d %s;", seqlen,  
          (alpha == DNA || alpha == RNA ? "BP" 
             : (alpha == PROTEIN ? "AA" : "CH")));
  if (alpha != DNA && alpha != RNA)
    fputc('\n', fp);
  else {
    a = c = g = t2 = u = o = 0;
    for (i=0; i < seqlen; i++) {
      switch (seq[i]) {
      case 'a': case 'A':  a++; break;
      case 'c': case 'C':  c++; break;
      case 'g': case 'G':  g++; break;
      case 't': case 'T':  t2++; break;
      case 'u': case 'U':  u++; break;
      default:             o++;
      }
    }

    if (alpha == RNA && !(u == 0 && t2 != 0))
      fprintf(fp, " %d A; %d C; %d G; %d U; %d other;\n", a, c, g, u, o);
    else
      fprintf(fp, " %d A; %d C; %d G; %d T; %d other;\n", a, c, g, t2, o);
  }

  if (isfp->format == FORMAT_GCG)
    putgcgseq(fp, seq, seqlen, info);
  else {
    for (i=0,count=60; i < seqlen; count+=60) {
      fputs("    ", fp);
      for (j=0; j < 6; j++) {
        fputc(' ', fp);
        for (k=0; k < 10; k++)
          fputc((i < seqlen ? seq[i++] : ' '), fp);
      }
    
      fprintf(fp, "%10d\n", (i < seqlen ? count : seqlen));
    }
    fputs("//\n", fp);
  }

  return STATUS_OK;
}


static int sprot_putseq(INTSEQFILE *isfp, char *seq, int seqlen, SEQINFO *info)
{
  int i, j, k, alpha, count, flag, flag2, pos;
  char ch, *s, *t, *id, *idend;
  FILE *fp = isfp->output_fp;

  /*
   * Print a non-accession identifier (an "sp" ID if possible) and
   * whether the sequence is complete or a fragment.
   */
  id = idend = NULL;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (mystreq(t, 'S', "SP:")) {
        id = t+3;
        idend = s;
        break;
      }
      else if (!id && !mystreq(t, 'A', "ACC:")) {
        id = t;
        idend = s;
      }
      if (*s) s++;
    }
    if (!id) {
      id = "Unknown";
      idend = id + 7;
    }
  }
  else {
    id = "Unknown";
    idend = id + 7;
  }
  ch = *idend; *idend = '\0';
  fprintf(fp, "ID   %-10s", id);
  *idend = ch;

  alpha = (info->alphabet != UNKNOWN ? info->alphabet 
                                     : guessalpha(seq, seqlen, NULL, NULL));

  fputs("  CONVERTED;  ", fp);
  if (info->iscircular)
    fputs("circular ", fp);
  else
    fputs("    ", fp);
          
  fputs((alpha == RNA ? "RNA" 
                      : (alpha == DNA ? "DNA" 
                                      : (alpha == PROTEIN ? "PRT" : "UNK"))),
        fp);
  fprintf(fp, "; %5d %s.\n", seqlen, (alpha == DNA || alpha == RNA ? "BP" 
                                        : (alpha == PROTEIN ? "AA" : "CH")));

  flag = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (mystreq(t, 'A', "ACC:")) {
        if (!flag) {
          fputs("AC   ", fp);
          flag = 5;
        }
        else if (flag + (s - t - 4) + 2 > 80) {
          fputs("\nAC   ", fp);
          flag = 5;
        }
        else {
          fputc(' ', fp);
          flag++;
        }

        fwrite(t+4, 1, s - t - 4, fp);
        fputc(';', fp);
        flag += (s - t - 4) + 1;
      }
      if (*s) s++;
    }
  }
  if (flag)
    fputc('\n', fp);

  if (info->date && info->date[0])
    fprintf(fp, "DT   %s\n", info->date);

  if (info->description && info->description[0]) {
    fputs("DE   ", fp);
    pos = putline(fp, 0, 5, info->description, 75, "DE   ", 0);
    if (info->isfragment)
      putline(fp, 1, pos, "(FRAGMENT)", 75, "DE   ", 0);
    fputc('.', fp);
    fputc('\n', fp);
  }

  if (info->organism && info->organism[0]) {
    fputs("OS   ", fp);
    putline(fp, 0, 5, info->organism, 75, "OS   ", 0);
    fputc('.', fp);
    fputc('\n', fp);
  }

  /*
   * Print the comment section:  Cross-References, actual comments, history.
   */
  flag = 0;
  if (info->comment && info->comment[0]) {
    fputs("CC   ", fp);
    putline(fp, 0, 5, info->comment, 75, "CC   ", 1);
    fputc('\n', fp);
    flag = 1;
  }

  flag2 = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (!(id && (t == id || t+3 == id)) && !mystreq(t, 'A', "ACC:")) {
        if (!flag2) {
          if (!flag) {
            fputs("CC   SEQIO Refs: ", fp);
            flag = 1;
          }
          else
            fputs("CC   \nCC   SEQIO Refs: ", fp);
          flag2 = 17;
        }
        else if (flag2 + (s - t) + 1 >= 80) {
          fputs("\nCC   SEQIO Refs: ", fp);
          flag2 = 17;
        }
        else {
          fputc('|', fp);
          flag2++;
        }

        fwrite(t, 1, s - t, fp);
        flag2 += s - t;
      }
      if (*s) s++;
    }
  }
  if (flag2)
    fputc('\n', fp);

  if (info->history && info->history[0]) {
    if (!flag) {
      fputs("CC   ", fp);
      putline(fp, 0, 5, info->history, 75, "CC   ", 1);
      fputc('\n', fp);
      flag = 1;
    }
    else {
      if (!flag2)
        fputs("CC   \n", fp);
      fputs("CC   ", fp);
      putline(fp, 0, 5, info->history, 75, "CC   ", 1);
      fputc('\n', fp);
    }
  }

  fprintf(fp, "SQ   SEQUENCE   %d %s;\n", seqlen, 
          (alpha == DNA || alpha == RNA ? "BP" 
             : (alpha == PROTEIN ? "AA" : "CH")));

  if (isfp->format == FORMAT_GCG)
    putgcgseq(fp, seq, seqlen, info);
  else {
    for (i=0,count=60; i < seqlen; count+=60) {
      fputs("    ", fp);
      for (j=0; i < seqlen && j < 6; j++) {
        fputc(' ', fp);
        for (k=0; i < seqlen && k < 10; k++)
          fputc(seq[i++], fp);
      }
      fputc('\n', fp);
    }
    fputs("//\n", fp);
  }

  return STATUS_OK;
}

static int nbrf_putseq(INTSEQFILE *isfp, char *seq, int seqlen, SEQINFO *info)
{
  int alpha, align_flag, flag, truelen;
  char *s, *t;
  FILE *fp = isfp->output_fp;

  alpha = guessalpha(seq, seqlen, &align_flag, &truelen);
  if (info->alphabet != UNKNOWN)
    alpha = info->alphabet;

  fputc('>', fp);
  switch (alpha) {
  case PROTEIN:
    fputs((info->isfragment ? "F1" : "P1"), fp);
    break;

  case DNA:
    fputs((info->iscircular ? "DC" : "DL"), fp);
    break;

  case RNA:
    fputs((info->iscircular ? "RC" : "RL"), fp);
    break;

  default:
    fputs("XX", fp);
  }
  fputc(';', fp);

  /*
   * Output the list of identifiers.
   */
  flag = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (!mystreq(t, 'A', "ACC:")) {
        if (flag)
          fputc('|', fp);

        fwrite(t, 1, s - t, fp);
        flag = 1;
      }
      if (*s) s++;
    }
  }
  if (!flag)
    fputs("Unknown", fp);
  fputc('\n', fp);

  /*
   * Then put the oneline description and the actual sequence.
   */
  put_oneline(fp, info, truelen, alpha, NULL);
  fputc('\n', fp);

  if (isfp->format != FORMAT_GCG) {
    simple_putseq(fp, seq, seqlen, alpha, align_flag, isfp->prettyflag);
    fputc('*', fp);
    fputc('\n', fp);
  }

  /*
   * Output the date, accession numbers, comments and history info.
   */
  if (info->date && info->date[0])
    fprintf(fp, "C;Date: %s\n", info->date);

  flag = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (mystreq(t, 'A', "ACC:")) {
        if (!flag) {
          fputs("C;Accession: ", fp);
          flag = 13;
        }
        else if (flag + (s - t - 4) + 2 >= 80) {
          fputs("\nC;Accession: ", fp);
          flag = 13;
        }
        else {
          fputs("; ", fp);
          flag += 2;
        }

        fwrite(t+4, 1, s - t - 4, fp);
        flag += s - t - 4;
      }
      if (*s) s++;
    }
  }
  if (flag)
    fputc('\n', fp);

  flag = 0;
  if (info->comment && info->comment[0]) {
    fputs("C;Comment: ", fp);
    putline(fp, 0, 11, info->comment, 80, "C;Comment: ", 1);
    fputc('\n', fp);
    flag = 1;
  }

  if (info->history && info->history[0]) {
    if (flag)
      fputs("C;Comment: \n", fp);
    fputs("C;Comment: ", fp);
    putline(fp, 0, 11, info->history, 80, "C;Comment: ", 1);
    fputc('\n', fp);
  }

  if (isfp->format == FORMAT_GCG)
    putgcgseq(fp, seq, seqlen, info);

  return STATUS_OK;
}

static int nbrfold_putseq(INTSEQFILE *isfp, char *seq, int seqlen,
                          SEQINFO *info)
{
  int alpha, align_flag, truelen;
  char *s, *t;
  FILE *fp = isfp->output_fp;

  alpha = guessalpha(seq, seqlen, &align_flag, &truelen);
  if (info->alphabet != UNKNOWN)
    alpha = info->alphabet;

  fputc('>', fp);
  switch (alpha) {
  case PROTEIN:
    fputs((info->isfragment ? "F1" : "P1"), fp);
    break;

  case DNA:
    fputs((info->iscircular ? "DC" : "DL"), fp);
    break;

  case RNA:
    fputs((info->iscircular ? "RC" : "RL"), fp);
    break;

  default:
    fputs("XX", fp);
  }
  fputc(';', fp);

  if (!(info->idlist && info->idlist[0]))
    fputs("Unknown\n", fp);
  else {
    for (t=s=info->idlist; *s && *s != '|'; s++) ;
    fwrite(t, 1, s - t, fp);
    fputc('\n', fp);
    if (*s && mystreq(s+1, 'A', "ACC:")) {
      for (t=++s; *s && *s != '|'; s++) ;
      fputc('~', fp);
      fwrite(t+4, 1, s - t - 4, fp);
      fputc(' ', fp);
    }
  }

  put_oneline(fp, info, truelen, alpha, NULL);
  fputc('\n', fp);

  if (isfp->format == FORMAT_GCG)
    putgcgseq(fp, seq, seqlen, info);
  else {
    simple_putseq(fp, seq, seqlen, alpha, align_flag, isfp->prettyflag);
    fputc('*', fp);
    fputc('\n', fp);
  }
    
  return STATUS_OK;
}

static int fasta_putseq(INTSEQFILE *isfp, char *seq, int seqlen, SEQINFO *info)
{
  int flag, alpha, align_flag, truelen;
  char *s, *t, *idlist;
  FILE *fp = isfp->output_fp;

  alpha = guessalpha(seq, seqlen, &align_flag, &truelen);
  if (info->alphabet != UNKNOWN)
    alpha = info->alphabet;
  
  fputc('>', fp);

  flag = 0;
  idlist = NULL;
  if (info->idlist && info->idlist[0]) {
    for (t=s=info->idlist; *s && *s != '|'; s++) ;
    if (*s && mystreq(s+1, 'A', "ACC:"))
      for (s++; *s && *s != '|'; s++) ;
    fwrite(t, 1, s - t, fp);
    flag = 1;
    if (*s)
      idlist = s+1;
  }
  if (flag)
    fputc(' ', fp);

  put_oneline(fp, info, truelen, alpha, NULL);
  fputc('\n', fp);

  if (info->comment && info->comment[0]) {
    fputs(";\n", fp);
    fputc(';', fp);
    putline(fp, 0, 0, info->comment, 0, ";", 0);
    fputc('\n', fp);
  }

  flag = 0;
  if (idlist) {
    for (s=idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;

      if (!flag) {
        fputs(";SEQIO Refs: ", fp);
        flag = 13;
      }
      else if (flag + (s - t) + 1 >= 80) {
        fputs("\n;SEQIO Refs: ", fp);
        flag = 13;
      }
      else {
        fputc('|', fp);
        flag++;
      }

      fwrite(t, 1, s - t, fp);
      flag += s - t;

      if (*s) s++;
    }
  }
  if (flag)
    fputc('\n', fp);

  if (info->history && info->history[0]) {
    if (!flag)
      fputs(";\n", fp);
    fputc(';', fp);
    putline(fp, 0, 0, info->history, 0, ";", 0);
    fputc('\n', fp);
  }

  if (isfp->format == FORMAT_GCG)
    putgcgseq(fp, seq, seqlen, info);
  else {
    simple_putseq(fp, seq, seqlen, alpha, align_flag, isfp->prettyflag);
    fputc('\n', fp);
  }
    
  return STATUS_OK;
}

static int fastaold_putseq(INTSEQFILE *isfp, char *seq, int seqlen,
                           SEQINFO *info)
{
  int alpha, align_flag, truelen;
  char *s, *t;
  FILE *fp = isfp->output_fp;

  alpha = guessalpha(seq, seqlen, &align_flag, &truelen);
  if (info->alphabet != UNKNOWN)
    alpha = info->alphabet;

  fputc('>', fp);

  if (info->idlist && info->idlist[0]) {
    for (t=s=info->idlist; *s && *s != '|'; s++) ;
    if (*s && mystreq(s+1, 'A', "ACC:"))
      for (s++; *s && *s != '|'; s++) ;
    fwrite(t, 1, s - t, fp);
    fputc(' ', fp);
  }

  put_oneline(fp, info, truelen, alpha, NULL);
  fputc('\n', fp);

  if (isfp->format == FORMAT_GCG)
    putgcgseq(fp, seq, seqlen, info);
  else {
    simple_putseq(fp, seq, seqlen, alpha, align_flag, isfp->prettyflag);
    fputc('\n', fp);
  }
    
  return STATUS_OK;
}

static int stanford_putseq(INTSEQFILE *isfp, char *seq, int seqlen,
                           SEQINFO *info)
{
  int flag, flag2, alpha, align_flag, truelen;
  char *s, *t;
  FILE *fp = isfp->output_fp;

  alpha = guessalpha(seq, seqlen, &align_flag, &truelen);
  if (info->alphabet != UNKNOWN)
    alpha = info->alphabet;

  flag = 0;
  if (info->comment && info->comment[0]) {
    fputc(';', fp);
    putline(fp, 0, 0, info->comment, 0, ";", 0);
    fputc('\n', fp);
    flag = 1;
  }

  flag2 = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s && *s != '|'; s++) ;
    if (*s && mystreq(s+1, 'A', "ACC:"))
      for (s++; *s && *s != '|'; s++) ;

    if (*s) {
      if (flag)
        fputs(";\n", fp);
        
      while (*s) {
        for (t=s; *s && *s != '|'; s++) ;

        if (!flag2) {
          fputs(";SEQIO Refs: ", fp);
          flag2 = 13;
        }
        else if (flag2 + (s - t) + 1 >= 80) {
          fputs("\n;SEQIO Refs: ", fp);
          flag2 = 13;
        }
        else {
          fputc('|', fp);
          flag2++;
        }

        fwrite(t, 1, s - t, fp);
        flag2 += s - t;

        if (*s) s++;
      }

      if (flag2)
        fputc('\n', fp);
      flag = 1;
    }
  }
  if (info->history && info->history[0]) {
    if (flag && !flag2)
      fputs(";\n", fp);
    fputc(';', fp);
    putline(fp, 0, 0, info->history, 0, ";", 0);
    fputc('\n', fp);
    flag = 1;
  }
  if (!flag)
    fputs(";\n", fp);

  if (info->idlist && info->idlist[0]) {
    for (t=s=info->idlist; *s && *s != '|'; s++) ;
    if (*s && mystreq(s+1, 'A', "ACC:"))
      for (s++; *s && *s != '|'; s++) ;
    fwrite(t, 1, s - t, fp);
    fputc(' ', fp);
    flag = 1;
  }

  put_oneline(fp, info, truelen, alpha, NULL);
  fputc('\n', fp);

  if (isfp->format == FORMAT_GCG)
    putgcgseq(fp, seq, seqlen, info);
  else {
    simple_putseq(fp, seq, seqlen, alpha, align_flag, isfp->prettyflag);
    fputc((info->iscircular ? '2' : '1'), fp);
    fputc('\n', fp);
  }
    
  return STATUS_OK;
}

static int stanfordold_putseq(INTSEQFILE *isfp, char *seq, int seqlen,
                              SEQINFO *info)
{
  int alpha, flag, align_flag, truelen;
  char *s, *t;
  FILE *fp = isfp->output_fp;

  alpha = guessalpha(seq, seqlen, &align_flag, &truelen);
  if (info->alphabet != UNKNOWN)
    alpha = info->alphabet;

  flag = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s && *s != '|'; s++) ;
    if (*s && mystreq(s+1, 'A', "ACC:"))
      for (s++; *s && *s != '|'; s++) ;

    if (*s) {
      fprintf(fp, ";SEQIO Refs: %s\n", s + 1);
      flag = 1;
    }
  }

  if (!flag && info->comment && info->comment[0]) {
    for (t=s=info->comment; *s && *s != '\n'; s++) ;
    fputc(';', fp);
    fwrite(t, 1, s - t, fp);
    fputc('\n', fp);
  }
  else if (!flag)
    fputs(";\n", fp);

  if (info->idlist && info->idlist[0]) {
    for (t=s=info->idlist; *s && *s != '|'; s++) ;
    if (*s && mystreq(s+1, 'A', "ACC:"))
      for (s++; *s && *s != '|'; s++) ;
    fwrite(t, 1, s - t, fp);
    fputc(' ', fp);
    flag = 1;
  }

  put_oneline(fp, info, seqlen, alpha, NULL);
  fputc('\n', fp);

  if (isfp->format == FORMAT_GCG)
    putgcgseq(fp, seq, seqlen, info);
  else {
    simple_putseq(fp, seq, seqlen, alpha, align_flag, isfp->prettyflag);
    fputc((info->iscircular ? '2' : '1'), fp);
    fputc('\n', fp);
  }
    
  return STATUS_OK;
}

static int gcg_putseq(INTSEQFILE *isfp, char *seq, int seqlen, SEQINFO *info)
{
  int flag, flag2;
  char *s, *t;
  FILE *fp = isfp->output_fp;

  if (++isfp->entry_count != 1) {
    set_error(E_EOF);
    return STATUS_EOF;
  }

  if (isfp->gcg_subformat != FORMAT_UNKNOWN) {
    return
      (*file_table[isfp->gcg_subformat].putseq_fn)(isfp, seq, seqlen, info);
  }

  /*
   * Print the identifiers.
   */
  flag = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;

      if (!flag) {
        fputs("Identifiers:\n    ", fp);
        flag = 4;
      }
      else if (flag + (s - t) + 1 >= 80) {
        fputs("\n    ", fp);
        flag = 4;
      }
      else {
        fputc('|', fp);
        flag++;
      }

      fwrite(t, 1, s - t, fp);
      flag += s - t;

      if (*s) s++;
    }
  }
  if (flag)
    fputc('\n', fp);

  if (info->description && info->description[0]) {
    if (flag)
      fputc('\n', fp);
    fputs("Description:\n    ", fp);
    putline(fp, 0, 4, info->description, 80, "    ", 0);
    fputc('\n', fp);
    flag = 1;
  }

  if (info->organism && info->organism[0]) {
    if (flag)
      fputc('\n', fp);
    fputs("Organism:\n    ", fp);
    putline(fp, 0, 4, info->organism, 80, "    ", 0);
    fputc('\n', fp);
    flag = 1;
  }

  if ((info->comment && info->comment[0]) ||
      (info->history && info->history[0])) {
    flag2 = 0;
    if (info->comment && info->comment[0]) {
      if (flag)
        fputc('\n', fp);
      fputs("Comments:\n    ", fp);
      putline(fp, 0, 0, info->comment, 0, "    ", 0);
      fputc('\n', fp);
      flag2 = 1;
    }

    if (info->history && info->history[0]) {
      if (flag2)
        fputc('\n', fp);
      fputs("    ", fp);
      putline(fp, 0, 0, info->history, 0, "    ", 0);
      fputc('\n', fp);
    }
    flag = 1;
  }

  putgcgseq(fp, seq, seqlen, info);

  return STATUS_OK;
}

static int msf_putseq(INTSEQFILE *isfp, char *seq, int seqlen, SEQINFO *info)
{
  int i, count, len, size;
  char buffer[32];

  if (isfp->malign_count == isfp->malign_size) {
    if (isfp->malign_size == 0) {
      size = 32;
      isfp->malign_seqs = (char **) malloc(size * sizeof(char *));
      isfp->malign_ids = (char **) malloc(size * sizeof(char *));
      isfp->malign_seqlens = (int *) malloc(size * sizeof(int));
      memory_error(isfp->malign_seqs == NULL || isfp->malign_ids == NULL ||
                   isfp->malign_seqlens == NULL,
                   return STATUS_FATAL);
      isfp->malign_size = size;
    }
    else {
      size = (isfp->malign_size += isfp->malign_size);
      isfp->malign_seqs = (char **) realloc(isfp->malign_seqs,
                                            size * sizeof(char *));
      isfp->malign_ids = (char **) realloc(isfp->malign_ids,
                                           size * sizeof(char *));
      isfp->malign_seqlens = (int *) realloc(isfp->malign_seqlens,
                                             size * sizeof(int));
      if (isfp->malign_seqs == NULL || isfp->malign_ids == NULL ||
          isfp->malign_seqlens == NULL) {
        if (isfp->malign_seqs != NULL) {
          for (i=0; i < isfp->malign_count; i++)
            free(isfp->malign_seqs[i]);
          free(isfp->malign_seqs);
          isfp->malign_seqs = NULL;
        }
        if (isfp->malign_ids != NULL) {
          for (i=0; i < isfp->malign_count; i++)
            free(isfp->malign_ids[i]);
          free(isfp->malign_ids);
          isfp->malign_ids = NULL;
        }
        if (isfp->malign_seqlens != NULL) {
          free(isfp->malign_seqlens);
          isfp->malign_seqlens = NULL;
        }
        isfp->malign_count = isfp->malign_size = 0;
        memory_error(1, return STATUS_FATAL);
      }
    }
  }

  count = isfp->malign_count;
  isfp->malign_seqs[count] = mystrdup2(seq, seq+seqlen);
  isfp->malign_seqlens[count] = seqlen;
  len = seqfoneline(info, buffer, 10, 1);
  isfp->malign_ids[count] = (len ? mystrdup(buffer) : mystrdup("Unknown"));

  if (isfp->malign_seqs[count] == NULL || isfp->malign_ids[count] == NULL) {
    if (isfp->malign_seqs != NULL) {
      for (i=0; i < isfp->malign_count; i++)
        free(isfp->malign_seqs[i]);
      free(isfp->malign_seqs);
      isfp->malign_seqs = NULL;
    }
    if (isfp->malign_ids != NULL) {
      for (i=0; i < isfp->malign_count; i++)
        free(isfp->malign_ids[i]);
      free(isfp->malign_ids);
      isfp->malign_ids = NULL;
    }
    if (isfp->malign_seqlens != NULL) {
      free(isfp->malign_seqlens);
      isfp->malign_seqlens = NULL;
    }
    isfp->malign_count = isfp->malign_size = 0;
    memory_error(1, return STATUS_FATAL);
  }

  isfp->malign_count++;
  return STATUS_OK;
}

static int msf_putseqend(INTSEQFILE *isfp)
{
  int i, j, k, len, len2, count, maxlen, numseqs, check, alpha;
  int align_flag, all_idprefix, any_idprefix, stripflag;
  char *seq, *s, *t, *date, buffer[32];
  FILE *fp;

  if (isfp->malign_count == 0)
    return STATUS_OK;

  fp = isfp->output_fp;
  numseqs = isfp->malign_count;
  all_idprefix = 1;
  any_idprefix = 0;
  for (i=0,maxlen=0; i < numseqs; i++) {
    if (maxlen < isfp->malign_seqlens[i])
      maxlen = isfp->malign_seqlens[i];

    if (is_idprefix(isfp->malign_ids[i]))
      any_idprefix = 1;
    else
      all_idprefix = 0;
  }
  stripflag = (any_idprefix && !all_idprefix);

  fputs("PileUp\n\n\n", fp);

  /*
   * Print the infoline.
   */
  for (t=s=isfp->filename; *s; s++)
    if (*s == dirch)
      t = s;
  fputc(' ', fp);
  fputs((t == isfp->filename ? t : t+1), fp);
  fputs("  ", fp);
  fprintf(fp, "MSF: %d  ", maxlen);

  alpha = guessalpha(isfp->malign_seqs[0], isfp->malign_seqlens[0],
                     &align_flag, NULL);

  if (alpha == RNA || alpha == DNA)
    fputs("Type: N  ", fp);
  else if (alpha == PROTEIN)
    fputs("Type: P  ", fp);

  date = get_today();
  for (i=1; i <= 12; i++)
    if (myncasecmp(date+3, months[i], 3) == 0)
      break;
  if (i <= 12)
    fprintf(fp, "%s %c%c, %s %s  ", gcg_full_months[i], date[0], date[1],
            date+7, date+12);

  for (i=0,check=0; i < numseqs; i++)
    check = gcg_checksum2(isfp->malign_seqs[i], isfp->malign_seqlens[i],
                          maxlen, check);
  fprintf(fp, "Check: %d  ..\n", check);

  /*
   * Print the header lines.
   */
  fputc('\n', fp);
  for (i=0; i < numseqs; i++) {
    s = isfp->malign_ids[i];
    if (stripflag && is_idprefix(s))
      while (*s++ != ':') ;

    fprintf(fp, " Name: %-15s  Len: %5d  Check: %4d  Weight:  1.00\n",
            s, maxlen, gcg_checksum2(isfp->malign_seqs[i],
                                     isfp->malign_seqlens[i], maxlen, 0));
  }
  fputs("\n//\n\n", fp);

  /*
   * Print the sequence lines.
   */
  for (j=0; j < maxlen; j+=50) {
    fputs("            ", fp);
    sprintf(buffer, "%d", j+1);
    len = strlen(buffer);
    fputs(buffer, fp);

    count = (j + 50 <= maxlen ? 50 : maxlen - j);
    sprintf(buffer, "%d", (j + count));
    len2 = strlen(buffer);

    count += count / 10 + (count % 10 ? 0 : -1);
    if (count >= len + len2) {
      for (i=0; i < count - len - len2; i++)
        fputc(' ', fp);
      fputs(buffer, fp);
    }
    fputc('\n', fp);

    for (i=0; i < numseqs; i++) {
      s = isfp->malign_ids[i];
      if (is_idprefix(s) && stripflag)
        while (*s++ != ':') ;
      fprintf(fp, "%-10s  ", s);

      len = isfp->malign_seqlens[i] - j;
      seq = isfp->malign_seqs[i] + (len > 0 ? j : 0);
      for (count=0,k=0; count < 50 && j + count < maxlen; count++) {
        if (k++ == 10) {
          fputc(' ', fp);
          k = 1;
        }
        if (count < len) {
          fputc((*seq == '-' ? '.' : *seq), fp);
          seq++;
        }
        else
          fputc('.', fp);
      }
      fputc('\n', fp);
    }
    fputc('\n', fp);
  }

  return STATUS_OK;
}


static int phylip_putseq(INTSEQFILE *isfp, char *seq, int seqlen,
                         SEQINFO *info)
{
  int i, count, len, size;
  char buffer[32];

  if (isfp->malign_count == isfp->malign_size) {
    if (isfp->malign_size == 0) {
      size = 32;
      isfp->malign_seqs = (char **) malloc(size * sizeof(char *));
      isfp->malign_ids = (char **) malloc(size * sizeof(char *));
      isfp->malign_seqlens = (int *) malloc(size * sizeof(int));
      memory_error(isfp->malign_seqs == NULL || isfp->malign_ids == NULL ||
                   isfp->malign_seqlens == NULL,
                   return STATUS_FATAL);
      isfp->malign_size = size;
    }
    else {
      size = (isfp->malign_size += isfp->malign_size);
      isfp->malign_seqs = (char **) realloc(isfp->malign_seqs,
                                            size * sizeof(char *));
      isfp->malign_ids = (char **) realloc(isfp->malign_ids,
                                           size * sizeof(char *));
      isfp->malign_seqlens = (int *) realloc(isfp->malign_seqlens,
                                             size * sizeof(int));
      if (isfp->malign_seqs == NULL || isfp->malign_ids == NULL ||
          isfp->malign_seqlens == NULL) {
        if (isfp->malign_seqs != NULL) {
          for (i=0; i < isfp->malign_count; i++)
            free(isfp->malign_seqs[i]);
          free(isfp->malign_seqs);
          isfp->malign_seqs = NULL;
        }
        if (isfp->malign_ids != NULL) {
          for (i=0; i < isfp->malign_count; i++)
            free(isfp->malign_ids[i]);
          free(isfp->malign_ids);
          isfp->malign_ids = NULL;
        }
        if (isfp->malign_seqlens != NULL) {
          free(isfp->malign_seqlens);
          isfp->malign_seqlens = NULL;
        }
        isfp->malign_count = isfp->malign_size = 0;
        memory_error(1, return STATUS_FATAL);
      }
    }
  }

  count = isfp->malign_count;
  isfp->malign_seqs[count] = mystrdup2(seq, seq+seqlen);
  isfp->malign_seqlens[count] = seqlen;
  len = seqfoneline(info, buffer, 10, 1);
  isfp->malign_ids[count] = (len ? mystrdup(buffer) : mystrdup("Unknown"));

  if (isfp->malign_seqs[count] == NULL || isfp->malign_ids[count] == NULL) {
    if (isfp->malign_seqs != NULL) {
      for (i=0; i < isfp->malign_count; i++)
        free(isfp->malign_seqs[i]);
      free(isfp->malign_seqs);
      isfp->malign_seqs = NULL;
    }
    if (isfp->malign_ids != NULL) {
      for (i=0; i < isfp->malign_count; i++)
        free(isfp->malign_ids[i]);
      free(isfp->malign_ids);
      isfp->malign_ids = NULL;
    }
    if (isfp->malign_seqlens != NULL) {
      free(isfp->malign_seqlens);
      isfp->malign_seqlens = NULL;
    }
    isfp->malign_count = isfp->malign_size = 0;
    memory_error(1, return STATUS_FATAL);
  }

  isfp->malign_count++;
  return STATUS_OK;
}

static int phyint_putseqend(INTSEQFILE *isfp)
{
  int i, j, k, len, count, maxlen, numseqs, flag;
  int all_idprefix, any_idprefix, stripflag;
  char *seq, *s;
  FILE *fp;

  if (isfp->malign_count == 0)
    return STATUS_OK;

  fp = isfp->output_fp;
  numseqs = isfp->malign_count;
  all_idprefix = 1;
  any_idprefix = 0;
  for (i=0,maxlen=0; i < numseqs; i++) {
    if (maxlen < isfp->malign_seqlens[i])
      maxlen = isfp->malign_seqlens[i];

    if (is_idprefix(isfp->malign_ids[i]))
      any_idprefix = 1;
    else
      all_idprefix = 0;
  }
  stripflag = (any_idprefix && !all_idprefix);

  fprintf(fp, "     %d    %d  I\n", numseqs, maxlen);

  for (flag=1,j=0; j < maxlen; j+=50,flag=0) {
    for (i=0; i < numseqs; i++) {
      if (flag) {
        s = isfp->malign_ids[i];
        if (stripflag && is_idprefix(s))
          while (*s++ != ':') ;

        fprintf(fp, "%-10s", s);
      }
      else
        fputs("          ", fp);

      len = isfp->malign_seqlens[i] - j;
      seq = isfp->malign_seqs[i] + (len > 0 ? j : 0);
      for (count=0,k=10; count < 50 && j + count < maxlen; count++,k++) {
        if (k == 10) {
          fputc(' ', fp);
          k = 0;
        }
        if (count < len)
          fputc(*seq++, fp);
        else
          fputc('-', fp);
      }
      fputc(' ', fp);
      fputc('\n', fp);
    }
    if (j + 50 < maxlen)
      fputc('\n', fp);
  }

  return STATUS_OK;
}


static int physeq_putseqend(INTSEQFILE *isfp)
{
  int i, j, k, len, count, maxlen, numseqs, flag;
  int all_idprefix, any_idprefix, stripflag;
  char *seq, *s;
  FILE *fp;

  if (isfp->malign_count == 0)
    return STATUS_OK;

  fp = isfp->output_fp;
  numseqs = isfp->malign_count;
  all_idprefix = 1;
  any_idprefix = 0;
  for (i=0,maxlen=0; i < numseqs; i++) {
    if (maxlen < isfp->malign_seqlens[i])
      maxlen = isfp->malign_seqlens[i];

    if (is_idprefix(isfp->malign_ids[i]))
      any_idprefix = 1;
    else
      all_idprefix = 0;
  }
  stripflag = (any_idprefix && !all_idprefix);

  fprintf(fp, "     %d    %d\n", numseqs, maxlen);

  for (i=0; i < numseqs; i++) {
    for (flag=1,j=0; j < maxlen; j+=50,flag=0) {
      if (flag) {
        s = isfp->malign_ids[i];
        if (stripflag && is_idprefix(s))
          while (*s++ != ':') ;

        fprintf(fp, "%-10s", s);
      }
      else
        fputs("          ", fp);

      len = isfp->malign_seqlens[i] - j;
      seq = isfp->malign_seqs[i] + (len > 0 ? j : 0);
      for (count=0,k=10; count < 50 && j + count < maxlen; count++,k++) {
        if (k == 10) {
          fputc(' ', fp);
          k = 0;
        }
        if (count < len)
          fputc(*seq++, fp);
        else
          fputc('-', fp);
      }
      fputc(' ', fp);
      fputc('\n', fp);
    }
  }

  return STATUS_OK;
}


static int clustal_putseq(INTSEQFILE *isfp, char *seq, int seqlen,
                          SEQINFO *info)
{
  int i, count, len, size;
  char buffer[32];

  if (isfp->malign_count == isfp->malign_size) {
    if (isfp->malign_size == 0) {
      size = 32;
      isfp->malign_seqs = (char **) malloc(size * sizeof(char *));
      isfp->malign_ids = (char **) malloc(size * sizeof(char *));
      isfp->malign_seqlens = (int *) malloc(size * sizeof(int));
      memory_error(isfp->malign_seqs == NULL || isfp->malign_ids == NULL ||
                   isfp->malign_seqlens == NULL,
                   return STATUS_FATAL);
      isfp->malign_size = size;
    }
    else {
      size = (isfp->malign_size += isfp->malign_size);
      isfp->malign_seqs = (char **) realloc(isfp->malign_seqs,
                                            size * sizeof(char *));
      isfp->malign_ids = (char **) realloc(isfp->malign_ids,
                                           size * sizeof(char *));
      isfp->malign_seqlens = (int *) realloc(isfp->malign_seqlens,
                                             size * sizeof(int));
      if (isfp->malign_seqs == NULL || isfp->malign_ids == NULL ||
          isfp->malign_seqlens == NULL) {
        if (isfp->malign_seqs != NULL) {
          for (i=0; i < isfp->malign_count; i++)
            free(isfp->malign_seqs[i]);
          free(isfp->malign_seqs);
          isfp->malign_seqs = NULL;
        }
        if (isfp->malign_ids != NULL) {
          for (i=0; i < isfp->malign_count; i++)
            free(isfp->malign_ids[i]);
          free(isfp->malign_ids);
          isfp->malign_ids = NULL;
        }
        if (isfp->malign_seqlens != NULL) {
          free(isfp->malign_seqlens);
          isfp->malign_seqlens = NULL;
        }
        isfp->malign_count = isfp->malign_size = 0;
        memory_error(1, return STATUS_FATAL);
      }
    }
  }

  count = isfp->malign_count;
  isfp->malign_seqs[count] = mystrdup2(seq, seq+seqlen);
  isfp->malign_seqlens[count] = seqlen;
  len = seqfoneline(info, buffer, 15, 1);
  isfp->malign_ids[count] = (len ? mystrdup(buffer) : mystrdup("Unknown"));

  if (isfp->malign_seqs[count] == NULL || isfp->malign_ids[count] == NULL) {
    if (isfp->malign_seqs != NULL) {
      for (i=0; i < isfp->malign_count; i++)
        free(isfp->malign_seqs[i]);
      free(isfp->malign_seqs);
      isfp->malign_seqs = NULL;
    }
    if (isfp->malign_ids != NULL) {
      for (i=0; i < isfp->malign_count; i++)
        free(isfp->malign_ids[i]);
      free(isfp->malign_ids);
      isfp->malign_ids = NULL;
    }
    if (isfp->malign_seqlens != NULL) {
      free(isfp->malign_seqlens);
      isfp->malign_seqlens = NULL;
    }
    isfp->malign_count = isfp->malign_size = 0;
    memory_error(1, return STATUS_FATAL);
  }

  isfp->malign_count++;
  return STATUS_OK;
}

static int clustal_putseqend(INTSEQFILE *isfp)
{
  int i, j, len, count, maxlen, numseqs;
  int all_idprefix, any_idprefix, stripflag;
  char *seq, *s;
  FILE *fp;

  if (isfp->malign_count == 0)
    return STATUS_OK;

  fp = isfp->output_fp;
  numseqs = isfp->malign_count;
  all_idprefix = 1;
  any_idprefix = 0;
  for (i=0,maxlen=0; i < numseqs; i++) {
    if (maxlen < isfp->malign_seqlens[i])
      maxlen = isfp->malign_seqlens[i];

    if (is_idprefix(isfp->malign_ids[i]))
      any_idprefix = 1;
    else
      all_idprefix = 0;
  }
  stripflag = (any_idprefix && !all_idprefix);

  fputs("CLUSTAL W(*.**) multiple sequence alignment\n\n\n", fp);

  for (j=0; j < maxlen; j+=60) {
    fputc('\n', fp);
    for (i=0; i < numseqs; i++) {
      s = isfp->malign_ids[i];
      if (stripflag && is_idprefix(s))
        while (*s++ != ':') ;

      fprintf(fp, "%-15s", s);

      len = isfp->malign_seqlens[i] - j;
      seq = isfp->malign_seqs[i] + (len > 0 ? j : 0);
      for (count=0; count < 60 && j + count < maxlen; count++) {
        if (count < len)
          fputc(*seq++, fp);
        else
          fputc('-', fp);
      }
      fputc('\n', fp);
    }
    for (count=0; count < 15; count++)
      fputc(' ', fp);
    for (count=0; count < 60 && j + count < maxlen; count++)
      fputc(' ', fp);
    fputc('\n', fp);
  }

  return STATUS_OK;
}



static int asn_putseq(INTSEQFILE *isfp, char *seq, int seqlen, SEQINFO *info)
{
  int i, col, alpha, flag, flag2, idlen, pos;
  char ch, *s, *t, *accstr, *accend, *gistr, *gisend, *giimstr, *giimend;
  char *bbsstr, *bbsend, *bbmstr, *bbmend, *idstr, *idend, *idpref;
  FILE *fp = isfp->output_fp;

  /*
   * Print the header (either of the whole file or the link between `seq'
   * records).
   */
  if (isfp->entry_count == 0)
    fputs("Bioseq-set ::= {\n  seq-set {\n", fp);
  else
    fputs(" ,\n", fp);
  isfp->entry_count++;

  fputs("    seq {\n", fp);

  /*
   * Do the `id' record.
   */
  accstr = gistr = giimstr = bbsstr = bbmstr = idstr = NULL;
  accend = gisend = giimend = bbsend = bbmend = idend = NULL;
  idpref = NULL;
  idlen = 0;

  fputs("      id {\n", fp);
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;

      ch = toupper(*t);
      if (!accstr && mystreq(t, 'A', "ACC:")) {
        accstr = t;
        accend = s;
      }
      else if (ch == 'B') {
        if (!bbsstr && mystreq(t, 'B', "BBS:")) {
          bbsstr = t;
          bbsend = s;
        }
        else if (!bbmstr && mystreq(t, 'B', "BBM:")) {
          bbmstr = t;
          bbmend = s;
        }
      }
      else if (ch == 'G' && toupper(t[1]) == 'I') {
        if (!gistr && t[2] == ':') {    /* GI: */
          gistr = t;
          gisend = s;
        }
        else if (!giimstr && mystreq(t+2, 'I', "IM:")) {   /* GIIM: */
          giimstr = t;
          giimend = s;
        }
      }
      else if (!idstr && (((i = 1) && mystreq(t, 'G', "GB:")) ||
                          ((i = 2) && mystreq(t, 'P', "PIR:")) ||
                          ((i = 3) && mystreq(t, 'E', "EMBL:")) ||
                          ((i = 4) && mystreq(t, 'S', "SP:")) ||
                          ((i = 5) && mystreq(t, 'P', "PDB:")) ||
                          ((i = 6) && mystreq(t, 'D', "DBJ:")) ||
                          ((i = 7) && mystreq(t, 'P', "PRF:")) ||
                          ((i = 8) && mystreq(t, 'O', "OTH:")))) {
        idstr = t;
        idend = s;
        switch (i) {
        case 1:  idpref = "genbank"; idlen = 3; break;
        case 2:  idpref = "pir"; idlen = 4; break;
        case 3:  idpref = "embl"; idlen = 5; break;
        case 4:  idpref = "swissprot"; idlen = 3; break;
        case 5:  idpref = "pdb"; idlen = 4; break;
        case 6:  idpref = "ddbj"; idlen = 4; break;
        case 7:  idpref = "prf"; idlen = 4; break;
        case 8:  idpref = "other"; idlen = 4; break;
        }
      }

      if (*s) s++;
    }

    flag = 0;
    if (idstr) {
      if (!strcmp(idpref, "pdb")) {
        fputs("        pdb {\n          mol \"", fp);
        fwrite(idstr+4, 1, idend - idstr - 4, fp);
        fputs("\" }", fp);
        accstr = accend = NULL;
      }
      else {
        fprintf(fp, "        %s {\n          name \"", idpref);
        fwrite(idstr+idlen, 1, idend - idstr - idlen, fp);
        fputc('"', fp);
        if (accstr) {
          fputs(" ,\n          accession \"", fp);
          fwrite(accstr+4, 1, accend - accstr - 4, fp);
          fputc('"', fp);
        }
        fputs(" }", fp);
      }
      flag = 1;
    }
    else
      accstr = NULL;

    if (gistr) {
      if (flag)
        fputs(" ,\n", fp);
      fputs("        gi ", fp);
      fwrite(gistr+3, 1, gisend - gistr - 3, fp);
      flag = 1;
    }
    if (bbsstr) {
      if (flag)
        fputs(" ,\n", fp);
      fputs("        gibbsq ", fp);
      fwrite(bbsstr+4, 1, bbsend - bbsstr - 4, fp);
      flag = 1;
    }
    if (bbmstr) {
      if (flag)
        fputs(" ,\n", fp);
      fputs("        gibbmt ", fp);
      fwrite(bbmstr+4, 1, bbmend - bbmstr - 4, fp);
      flag = 1;
    }
    if (giimstr) {
      if (flag)
        fputs(" ,\n", fp);
      fputs("        giim {\n          id ", fp);
      fwrite(giimstr+5, 1, giimend - giimstr - 5, fp);
      fputs(" }", fp);
      flag = 1;
    }

    if (!flag)
      fputs("        other {\n          name \"(below)\" }", fp);
  }
  else {
    fputs("        other {\n          name \"Unknown\" }", fp);
  }
  fputs(" } ,\n", fp);


  /*
   * Do the `descr' record.
   */
  flag = 0;
  if (info->isfragment) {
    if (!flag) {
      fputs("      descr {\n", fp);
      flag = 1;
    }
    else
      fputs(" ,\n", fp);

    fputs("        modif {\n          partial }", fp);
  }

  if (info->description && info->description[0]) {
    if (!flag) {
      fputs("      descr {\n", fp);
      flag = 1;
    }
    else
      fputs(" ,\n", fp);

    fputs("        title \"", fp);
    putline(fp, 0, 15, info->description, 78, " ", 0);
    fputc('"', fp);
  }

  if (info->organism && info->organism[0]) {
    if (!flag) {
      fputs("      descr {\n", fp);
      flag = 1;
    }
    else
      fputs(" ,\n", fp);

    fputs("        org {\n          taxname \"", fp);
    putline(fp, 0, 19, info->organism, 78, " ", 0);
    fputs("\" }", fp);
  }

  if (info->date && info->date[0]) {
    if (!flag) {
      fputs("      descr {\n", fp);
      flag = 1;
    }
    else
      fputs(" ,\n", fp);

    fputs("        update-date\n", fp);
    fprintf(fp, "          str \"%s\"", info->date);
  }

  if (info->comment && info->comment[0]) {
    if (!flag) {
      fputs("      descr {\n", fp);
      flag = 1;
    }
    else
      fputs(" ,\n", fp);

    fputs("        comment \"", fp);
    putline(fp, 0, 17, info->comment, 78, " ", 1);
    fputc('"', fp);
  }

  flag2 = 0;
  if (info->idlist && info->idlist[0]) {
    for (s=info->idlist; *s; ) {
      for (t=s; *s && *s != '|'; s++) ;
      if (t != idstr && t != accstr && t != gistr && 
          t != giimstr && t != bbsstr && t != bbmstr) {
        if (!flag2) {
          if (!flag) {
            fputs("      descr {\n", fp);
            flag = 1;
          }
          else {
            fputs(" ,\n", fp);
          }
            
          fputs("        comment \"SEQIO Refs: ", fp);
          flag2 = 29;
        }
        else if (flag2 + (s - t) + 3 >= 78) {
          fputs("\" ,\n        comment \"SEQIO Refs: ", fp);
          flag2 = 29;
        }
        else {
          fputc('|', fp);
          flag2++;
        }

        fwrite(t, 1, s - t, fp);
        flag2 += s - t;
      }

      if (*s) s++;
    }
  }

  if (info->history && info->history[0]) {
    if (!flag2) {
      if (!flag) {
        fputs("      descr {\n", fp);
        flag = 1;
      }
      else
        fputs(" ,\n", fp);

      fputs("        comment \"", fp);
      pos = 17;
      flag2 = 1;
    }
    else {
      fputs("\n ", fp);
      pos = 1;
    }

    putline(fp, 0, pos, info->history, 78, " ", 1);
  }
  
  if (flag2)
    fputc('"', fp);

  if (flag)
    fputs(" } ,\n", fp);

  /*
   * Do the `inst' record.
   */
  fputs("      inst {\n", fp);
  fputs("        repr raw ,\n", fp);

  alpha = (info->alphabet != UNKNOWN ? info->alphabet 
                                     : guessalpha(seq, seqlen, NULL, NULL));
  if (alpha == DNA || alpha == RNA || alpha == PROTEIN)
    fprintf(fp, "        mol %s ,\n",
            (alpha == DNA ? "dna" : (alpha == RNA ? "rna" : "aa")));

  fprintf(fp, "        length %d ,\n", seqlen);
  if (info->iscircular)
    fputs("        topology circular ,\n", fp);
    
  fputs("        seq-data\n", fp);
  fprintf(fp, "          %s \"",
          (alpha == DNA || alpha == RNA ? "iupacna" 
             : (alpha == PROTEIN ? "iupacaa" : "ascii")));

  for (i=0,col=19; i < seqlen; i++) {
    if (++col == 79) {
      fputc('\n', fp);
      col = 1;
    }
    fputc(seq[i], fp);
  }
  fputs("\" } }", fp);

  return STATUS_OK;
}


static int asn_putseqend(INTSEQFILE *isfp)
{
  if (isfp->optype == OP_WRITE)
    fputs(" } }\n", isfp->output_fp);
  return STATUS_OK;
}





/*
 *
 *
 * Section containing the annotate functions.
 *
 *
 *
 */


static int genbank_annotate(FILE *fp, char *entry, int entrylen,
                            char *newcomment, int flag)
{
  int status, count, hcount, ncount, pos;
  char *s, *line, *end, *history, *comment, *comend, *lastline;

  error_test(!mystreq(entry, 'L', "LOCUS       "),
             E_PARSEERROR, return STATUS_ERROR, 
             print_error("seqfannotate:  Entry not in GenBank format.\n"));

  for (ncount=0,s=newcomment; *s; s++)
    if (*s == '\n')
      ncount++;
  if (*(s-1) != '\n')
    ncount++;

  /*
   * Skip past and output the text before the comment.
   */
  gi_startline(entry, entrylen);
  
  lastline = NULL;
  while (1) {
    status = gi_getline(&line, &end, 0);
    error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
               print_error("seqfannotate:  Premature end of entry.\n"));
  
    if (mystreq(line, 'C', "COMMENT") || mystreq(line, 'F', "FEATURES") ||
        mystreq(line, 'B', "BASE COUNT") || mystreq(line, 'O', "ORIGIN"))
      break;
  }
  fwrite(entry, 1, line - entry, fp);

  /*
   * Output the old and new comments and the history, depending on what
   * appears in the entry and the value of `flag'.
   */
  fputs("COMMENT     ", fp);

  history = NULL;
  count = hcount = 0;
  if (toupper(*line) == 'C') {
    for (s=line+7; s < line + 12 && s < end && isspace(*s); s++) ;
    comment = s;
    comend = NULL;
    while (isspace(line[0]) || count == 0) {
      for (s=line; s < end && isspace(*s); s++) ;
      if (s < end &&
          (mystreq(s, 'S', "SEQIO") || (history != NULL && s == line + 14))) {
        if (history == NULL) {
          history = s;
          if (comend == NULL)
            comend = line;
        }
        hcount++;
      }
      else {
        history = NULL;
        hcount = 0;
        comend = (line + 12 == end ? line : NULL);
      }

      status = gi_getline(&line, &end, 0);
      error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
                 print_error("seqfannotate:  Premature end of entry.\n"));
      count++;
    }
    if (comend == NULL)
      comend = line;

    if (flag && comment < comend) {
      fwrite(comment, 1, comend - comment, fp);
      fputs("            \n            ", fp);
      count++;
    }
  }

  /*
   * Output the new comment, the history lines if they exist, and 
   * a history line noting the annotation.
   */
  putline(fp, 0, 12, newcomment, 80, "            ", 1);
  fputs("\n            \n            ", fp);

  if (history != NULL) {
    fwrite(history, 1, line - history, fp);
    fputs("            ", fp);
  }

  pos = (flag ? count - hcount + 1 : 1);
  fprintf(fp, "SEQIO annotation, lines %d-%d.   %s\n", pos, pos + ncount - 1,
          get_today());

  /*
   * Output the rest of the entry.
   */
  end = entry + entrylen;
  fwrite(line, 1, end - line, fp);

  return STATUS_OK;
}


static int pir_annotate(FILE *fp, char *entry, int entrylen, char *newcomment,
                        int flag)
{
  int status, count, hcount, ncount, pos;
  char *s, *line, *end, *history, *comment, *comend, *lastline;

  error_test(!mystreq(entry, 'E', "ENTRY"), E_PARSEERROR, return STATUS_ERROR, 
             print_error("seqfannotate:  Entry not in PIR format.\n"));

  for (ncount=0,s=newcomment; *s; s++)
    if (*s == '\n')
      ncount++;
  if (*(s-1) != '\n')
    ncount++;

  /*
   * Skip past and output the text before the comment.
   */
  gi_startline(entry, entrylen);
  
  lastline = NULL;
  while (1) {
    status = gi_getline(&line, &end, 0);
    error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
               print_error("seqfannotate:  Premature end of entry.\n"));
  
    if (!isspace(line[0]) &&
        (mystreq(line, 'C', "COMMENT") || mystreq(line, 'G', "GENETIC") ||
         mystreq(line, 'C', "CLASSIFICATION") ||
         mystreq(line, 'K', "KEYWORDS") || mystreq(line, 'F', "FEATURE") ||
         mystreq(line, 'S', "SUMMARY") || mystreq(line, 'S', "SEQUENCE")))
      break;
  }
  fwrite(entry, 1, line - entry, fp);

  /*
   * Output the old and new comments and the history, depending on what
   * appears in the entry and the value of `flag'.
   */
  fputs("COMMENT    ", fp);

  history = NULL;
  count = hcount = 0;
  if (toupper(*line) == 'C' && toupper(line[1]) == 'O') {
    for (s=line+7; s < line + 11 && s < end && isspace(*s); s++) ;
    comment = s;
    comend = NULL;
    while (isspace(line[0]) || count == 0) {
      for (s=line; s < end && isspace(*s); s++) ;
      if (s < end &&
          (mystreq(s, 'S', "SEQIO") || (history != NULL && s == line + 13))) {
        if (history == NULL) {
          history = s;
          if (comend == NULL)
            comend = line;
        }
        hcount++;
      }
      else {
        history = NULL;
        hcount = 0;
        comend = (line + 11 == end ? line : NULL);
      }

      status = gi_getline(&line, &end, 0);
      error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
                 print_error("seqfannotate:  Premature end of entry.\n"));
      count++;
    }
    if (comend == NULL)
      comend = line;

    if (flag && comment < comend) {
      fwrite(comment, 1, comend - comment, fp);
      fputs("           \n           ", fp);
      count++;
    }
  }

  /*
   * Output the new comment, the history lines if they exist, and 
   * a history line noting the annotation.
   */
  putline(fp, 0, 11, newcomment, 80, "           ", 1);
  fputs("\n           \n           ", fp);

  if (history != NULL) {
    fwrite(history, 1, line - history, fp);
    fputs("           ", fp);
  }

  pos = (flag ? count - hcount + 1 : 1);
  fprintf(fp, "SEQIO annotation, lines %d-%d.   %s\n", pos, pos + ncount - 1,
          get_today());

  /*
   * Output the rest of the entry.
   */
  end = entry + entrylen;
  fwrite(line, 1, end - line, fp);

  return STATUS_OK;
}

static int embl_annotate(FILE *fp, char *entry, int entrylen, char *newcomment,
                         int flag)
{
  int status, count, hcount, ncount, pos, ccflag, xxflag;
  char *s, *line, *end, *history, *comment, *comend, *lastline;
  char *trailing_line;

  error_test(!mystreq(entry, 'I', "ID   "), E_PARSEERROR, return STATUS_ERROR, 
             print_error("seqfannotate:  Entry not in EMBL format.\n"));

  for (ncount=0,s=newcomment; *s; s++)
    if (*s == '\n')
      ncount++;
  if (*(s-1) != '\n')
    ncount++;

  /*
   * Skip past and output the text before the comment.
   */
  gi_startline(entry, entrylen);
  
  lastline = NULL;
  while (1) {
    status = gi_getline(&line, &end, 0);
    error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
               print_error("seqfannotate:  Premature end of entry.\n"));
  
    if (mystreq(line, 'C', "CC   ") || mystreq(line, 'X', "XX   ") ||
        mystreq(line, 'S', "SQ   ") || !strncmp(line, "     ", 5))
      break;
    
    if (!lastline &&
        (mystreq(line, 'D', "DR   ") || mystreq(line, 'P', "PR   ") ||
         mystreq(line, 'F', "FT   ") || mystreq(line, 'F', "FH   ")))
      lastline = line;
  }
  ccflag = (toupper(*line) == 'C');
  xxflag = (toupper(*line) == 'X');

  if (!ccflag && !xxflag && lastline != NULL) {
    gi_ungetline(lastline);
    gi_getline(&line, &end, 0);
  }

  fwrite(entry, 1, line - entry, fp);

  /*
   * Output the old and new comments and the history, depending on what
   * appears in the entry and the value of `flag'.
   */
  fputs((xxflag ? "XX   " : "CC   "), fp);

  history = NULL;
  count = hcount = 0;
  if (ccflag || xxflag) {
    comment = line+5;
    comend = NULL;
    trailing_line = NULL;
    while ((xxflag && mystreq(line, 'X', "XX   ")) ||
           (ccflag &&
            (mystreq(line, 'C', "CC   ") || mystreq(line, 'X', "XX")))) {
      if (trailing_line != NULL) {
        history = NULL;
        hcount = 0;
        comend = trailing_line;
        trailing_line = NULL;
      }

      if (end - line > 5 && (mystreq(line+5, 'S', "SEQIO") ||
                             (history != NULL && line[5] == ' ' &&
                              line[6] == ' ' && line[7] != ' '))) {
        if (history == NULL) {
          history = line+5;
          if (comend == NULL)
            comend = line;
        }
        hcount++;
      }
      else if (end - line < 5)
        trailing_line = line;
      else {
        history = NULL;
        hcount = 0;
        comend = (line+5 == end ? line : NULL);
      }

      status = gi_getline(&line, &end, 0);
      error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
                 print_error("seqfannotate:  Premature end of entry.\n"));
      count++;
    }
    if (trailing_line != NULL)
      line = trailing_line;

    if (comend == NULL)
      comend = line;

    if (flag && comment < comend) {
      fwrite(comment, 1, comend - comment, fp);
      fputs((xxflag ? "XX   \nXX   " : "CC   \nCC   "), fp);
      count++;
    }
  }

  /*
   * Output the new comment, the history lines if they exist, and 
   * a history line noting the annotation.
   */
  putline(fp, 0, 5, newcomment, 80, (xxflag ? "XX   " : "CC   "), 1);
  fputs((xxflag ? "\nXX   \nXX   " : "\nCC   \nCC   "), fp);

  if (history != NULL) {
    fwrite(history, 1, line - history, fp);
    fputs((xxflag ? "XX   " : "CC   "), fp);
  }

  pos = (flag ? count - hcount + 1 : 1);
  fprintf(fp, "SEQIO annotation, lines %d-%d.   %s\n", pos, pos + ncount - 1,
          get_today());

  if (!ccflag && !xxflag)
    fputs("XX\n", fp);

  /*
   * Output the rest of the entry.
   */
  end = entry + entrylen;
  fwrite(line, 1, end - line, fp);

  return STATUS_OK;
}


static int sprot_annotate(FILE *fp, char *entry, int entrylen,
                          char *newcomment, int flag)
{
  int status, count, hcount, ncount, pos, ccflag;
  char *s, *line, *end, *history, *comment, *comend, *lastline;

  error_test(!mystreq(entry, 'I', "ID   "), E_PARSEERROR, return STATUS_ERROR, 
             print_error("seqfannotate:  Entry not in Swiss-Prot format.\n"));

  for (ncount=0,s=newcomment; *s; s++)
    if (*s == '\n')
      ncount++;
  if (*(s-1) != '\n')
    ncount++;

  /*
   * Skip past and output the text before the comment.
   */
  gi_startline(entry, entrylen);
  
  lastline = NULL;
  while (1) {
    status = gi_getline(&line, &end, 0);
    error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
               print_error("seqfannotate:  Premature end of entry.\n"));
  
    if (mystreq(line, 'C', "CC   ") || mystreq(line, 'S', "SQ   ") ||
        !strncmp(line, "     ", 5))
      break;
    
    if (!lastline &&
        (mystreq(line, 'D', "DR   ") || mystreq(line, 'K', "KW   ") ||
         mystreq(line, 'F', "FT   ")))
      lastline = line;
  }
  ccflag = (toupper(*line) == 'C');

  if (!ccflag && lastline != NULL) {
    gi_ungetline(lastline);
    gi_getline(&line, &end, 0);
  }

  fwrite(entry, 1, line - entry, fp);

  /*
   * Output the old and new comments and the history, depending on what
   * appears in the entry and the value of `flag'.
   */
  fputs("CC   ", fp);

  history = NULL;
  count = hcount = 0;
  if (ccflag) {
    comment = line+5;
    comend = NULL;
    while (mystreq(line, 'C', "CC   ")) {
      if (mystreq(line+5, 'S', "SEQIO") ||
          (history != NULL && line[5] == ' ' &&
           line[6] == ' ' && line[7] != ' ')) {
        if (history == NULL) {
          history = line+5;
          if (comend == NULL)
            comend = line;
        }
        hcount++;
      }
      else {
        history = NULL;
        hcount = 0;
        comend = (line+5 == end ? line : NULL);
      }

      status = gi_getline(&line, &end, 0);
      error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
                 print_error("seqfannotate:  Premature end of entry.\n"));
      count++;
    }
    if (comend == NULL)
      comend = line;

    if (flag && comment < comend) {
      fwrite(comment, 1, comend - comment, fp);
      fputs("CC   \nCC   ", fp);
      count++;
    }
  }

  /*
   * Output the new comment, the history lines if they exist, and 
   * a history line noting the annotation.
   */
  putline(fp, 0, 5, newcomment, 80, "CC   ", 1);
  fputs("\nCC   \nCC   ", fp);

  if (history != NULL) {
    fwrite(history, 1, line - history, fp);
    fputs("CC   ", fp);
  }

  pos = (flag ? count - hcount + 1 : 1);
  fprintf(fp, "SEQIO annotation, lines %d-%d.   %s\n", pos, pos + ncount - 1,
          get_today());

  /*
   * Output the rest of the entry.
   */
  end = entry + entrylen;
  fwrite(line, 1, end - line, fp);

  return STATUS_OK;
}


static int fasta_annotate(FILE *fp, char *entry, int entrylen,
                          char *newcomment, int flag)
{
  int status, count, hcount, ncount, pos;
  char *s, *line, *end, *history, *comment, *comend;

  error_test(*entry != '>', E_PARSEERROR, return STATUS_ERROR, 
             print_error("seqfannotate:  Entry not in FASTA format.\n"));

  for (ncount=0,s=newcomment; *s; s++)
    if (*s == '\n')
      ncount++;
  if (*(s-1) != '\n')
    ncount++;

  /*
   * Skip past and output the oneline description before the comment.
   */
  gi_startline(entry, entrylen);
  if (gi_getline(&line, &end, 0) == 0 || gi_getline(&line, &end, 0) == 0)
    raise_error(E_PARSEERROR, return STATUS_ERROR, 
                print_error("seqfannotate:  Premature end of entry.\n"));

  fwrite(entry, 1, line - entry, fp);

  /*
   * Output the old and new comments and the history, depending on what
   * appears in the entry and the value of `flag'.
   */
  fputc('>', fp);

  history = NULL;
  count = hcount = 0;
  if (*line == '>') {
    comment = line+1;
    comend = NULL;
    while (*line == '>') {
      if (mystreq(line+1, 'S', "SEQIO") ||
          (history != NULL && line[1] == ' ' &&
           line[2] == ' ' && line[3] != ' ')) {
        if (history == NULL) {
          history = line+1;
          if (comend == NULL)
            comend = line;
        }
        hcount++;
      }
      else {
        history = NULL;
        hcount = 0;
        comend = (line+1 == end ? line : NULL);
      }

      status = gi_getline(&line, &end, 0);
      error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
                 print_error("seqfannotate:  Premature end of entry.\n"));
      count++;
    }
    if (comend == NULL)
      comend = line;

    if (flag && comment < comend) {
      fwrite(comment, 1, comend - comment, fp);
      fputc('\n', fp);
    }
  }

  /*
   * Output the new comment, the history lines if they exist, and 
   * a history line noting the annotation.
   */
  fputs("\n>", fp);
  count++;
  putline(fp, 0, 1, newcomment, 80, ">", 1);
  fputs("\n>\n>", fp);

  if (history != NULL) {
    fwrite(history, 1, line - history, fp);
    fputc('>', fp);
  }

  pos = (flag ? count - hcount + 1 : 1);
  fprintf(fp, "SEQIO annotation, lines %d-%d.   %s\n", pos, pos + ncount - 1,
          get_today());

  /*
   * Output the rest of the entry.
   */
  end = entry + entrylen;
  fwrite(line, 1, end - line, fp);

  return STATUS_OK;
}


static int nbrf_annotate(FILE *fp, char *entry, int entrylen,
                         char *newcomment, int flag)
{
  int status, count, hcount, ncount, pos, ccflag;
  char *s, *line, *end, *history, *comment, *comend, *lastline;

  error_test(*entry != '>', E_PARSEERROR, return STATUS_ERROR, 
             print_error("seqfannotate:  Entry not in NBRF format.\n"));

  for (ncount=0,s=newcomment; *s; s++)
    if (*s == '\n')
      ncount++;
  if (*(s-1) != '\n')
    ncount++;

  /*
   * Skip past and output the oneline description before the comment.
   */
  gi_startline(entry, entrylen);
  
  lastline = NULL;
  ccflag = 0;
  while ((status = gi_getline(&line, &end, 0))) {
    if (mystreq(line, 'C', "C;COMMENT:")) {
      ccflag = 1;
      break;
    }
    
    if (!lastline &&
        (mystreq(line, 'F', "F;") ||
         (toupper(line[0]) == 'C' &&
          (mystreq(line, 'C', "C;GENETICS:") ||
           mystreq(line, 'C', "C;COMPLEX:") ||
           mystreq(line, 'C', "C;FUNCTION:") ||
           mystreq(line, 'C', "C;SUPERFAMILY:") ||
           mystreq(line, 'C', "C;KEYWORDS:")))))
      lastline = line;
  }
  if (status == 0)
    fwrite(entry, 1, entrylen, fp);
  else {
    if (!ccflag && lastline != NULL) {
      gi_ungetline(lastline);
      gi_getline(&line, &end, 0);
    }

    fwrite(entry, 1, line - entry, fp);
  }

  /*
   * Output the old and new comments and the history, depending on what
   * appears in the entry and the value of `flag'.
   */
  fputs("C;Comment: ", fp);

  history = NULL;
  count = hcount = 0;
  if (ccflag) {
    comment = line+11;
    comend = NULL;
    while (status && mystreq(line, 'C', "C;COMMENT: ")) {
      if (mystreq(line+11, 'S', "SEQIO") ||
          (history != NULL && line[11] == ' ' &&
           line[12] == ' ' && line[13] != ' ')) {
        if (history == NULL) {
          history = line+11;
          if (comend == NULL)
            comend = line;
        }
        hcount++;
      }
      else {
        history = NULL;
        hcount = 0;
        comend = (line+11 == end ? line : NULL);
      }

      status = gi_getline(&line, &end, 0);
      count++;
    }
    if (status == 0)
      comend = entry + entrylen;
    else if (comend == NULL)
      comend = line;

    if (flag && comment < comend) {
      fwrite(comment, 1, comend - comment, fp);
      fputs("C;Comment: \nC;Comment: ", fp);
      count++;
    }
  }

  /*
   * Output the new comment, the history lines if they exist, and 
   * a history line noting the annotation.
   */
  putline(fp, 0, 11, newcomment, 80, "C;Comment: ", 1);
  fputs("\nC;Comment: \nC;Comment: ", fp);

  if (history != NULL) {
    fwrite(history, 1, line - history, fp);
    fputs("C;Comment: ", fp);
  }

  pos = (flag ? count - hcount + 1 : 1);
  fprintf(fp, "SEQIO annotation, lines %d-%d.   %s\n", pos, pos + ncount - 1,
          get_today());

  /*
   * Output the rest of the entry.
   */
  if (status != 0) {
    end = entry + entrylen;
    fwrite(line, 1, end - line, fp);
  }

  return STATUS_OK;
}


static int stanford_annotate(FILE *fp, char *entry, int entrylen,
                             char *newcomment, int flag)
{
  int status, count, hcount, ncount, pos;
  char *s, *line, *end, *history, *comment, *comend;

  error_test(*entry != ';', E_PARSEERROR, return STATUS_ERROR, 
             print_error("seqfannotate:  Entry not in IG/Stanford format.\n"));

  for (ncount=0,s=newcomment; *s; s++)
    if (*s == '\n')
      ncount++;
  if (*(s-1) != '\n')
    ncount++;

  /*
   * Setup the gi_getline info to start at the first comment line (which
   * is also the first entry line).
   */
  gi_startline(entry, entrylen);
  status = gi_getline(&line, &end, 0);
  error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
             print_error("seqfannotate:  Premature end of entry.\n"));
  
  /*
   * Output the old and new comments and the history, depending on what
   * appears in the entry and the value of `flag'.
   */
  fputc(';', fp);

  history = NULL;
  count = hcount = 0;
  if (*line == ';') {
    comment = line+1;
    comend = NULL;
    while (*line == ';') {
      if (mystreq(line+1, 'S', "SEQIO") ||
          (history != NULL && line[1] == ' ' &&
           line[2] == ' ' && line[3] != ' ')) {
        if (history == NULL) {
          history = line+1;
          if (comend == NULL)
            comend = line;
        }
        hcount++;
      }
      else {
        history = NULL;
        hcount = 0;
        comend = (line+1 == end ? line : NULL);
      }

      status = gi_getline(&line, &end, 0);
      error_test(status == 0, E_PARSEERROR, return STATUS_ERROR, 
                 print_error("seqfannotate:  Premature end of entry.\n"));
      count++;
    }
    if (comend == NULL)
      comend = line;

    if (flag && comment < comend) {
      fwrite(comment, 1, comend - comment, fp);
      fputs(";\n;", fp);
      count++;
    }
  }

  /*
   * Output the new comment, the history lines if they exist, and 
   * a history line noting the annotation.
   */
  putline(fp, 0, 1, newcomment, 80, ";", 1);
  fputs("\n;\n;", fp);

  if (history != NULL) {
    fwrite(history, 1, line - history, fp);
    fputc(';', fp);
  }

  pos = (flag ? count - hcount + 1 : 1);
  fprintf(fp, "SEQIO annotation, lines %d-%d.   %s\n", pos, pos + ncount - 1,
          get_today());

  /*
   * Output the rest of the entry.
   */
  end = entry + entrylen;
  fwrite(line, 1, end - line, fp);

  return STATUS_OK;
}

static int asn_annotate(FILE *fp, char *entry, int entrylen, char *newcomment,
                        int flag)
{
  int ccflag, status, count, hcount, ncount, pos, oldpe;
  int tempc, temph, inhistory, blank;
  char qch, *s, *start, *end, *entryend, *head, *tail, *comment, *comend;
  char *destr, *deend, *str, *strend, *lastcomment, *lastend;

  entryend = entry + entrylen;

  for (s=entry; s < entryend && isspace(*s); s++) ;
  error_test(s + 4 >= entryend || !mystreq(s, 'S', "SEQ "),
             E_PARSEERROR, return STATUS_ERROR,
             print_error("seqfannotate:  Entry not an ASN.1 "
                         "`Bioseq-set.seq-set.seq' record.\n"));

  for (ncount=0,s=newcomment; *s; s++)
    if (*s == '\n')
      ncount++;

  /*
   * Look for the "seq.descr.comment" sub-record.  If no such record is found,
   * look for, in order, a "seq.descr" (comment goes at the end of that
   * record), "seq.id" (descr.comment goes after that record), or
   * "seq.inst" (descr.comment goes before that record).  If none of those
   * are found, put a descr.comment record at the end of the entry.
   */
  ccflag = 0;
  head = tail = start = end = NULL;
  comment = comend = NULL;

  destr = NULL;
  oldpe = pe_flag;
  pe_flag = PE_NONE;
  status = asn_parse(entry, entryend, "seq.descr", &destr, &deend, NULL);
  pe_flag = oldpe;
  error_test(status == -1, E_PARSEERROR, return STATUS_ERROR,
             print_error("seqfannotate:  Invalid format for ASN.1 entry.\n"));

  if (destr != NULL) {
    oldpe = pe_flag;
    pe_flag = PE_NONE;
    status = asn_parse(destr, deend, "descr.comment", &comment, &comend, NULL);
    pe_flag = oldpe;
    error_test(status == -1, E_PARSEERROR, return STATUS_ERROR,
               print_error("seqfannotate:  Invalid format for ASN.1 "
                           "entry.\n"));

    if (comment != NULL) {
      ccflag = 1;
      start = comment;
      end = deend;
    }
    else {
      for (start=deend; start > destr && isspace(*(start-1)); start--) ;
      error_test(*(start-1) != '}', E_PARSEERROR, return STATUS_ERROR,
               print_error("seqfannotate:  Invalid format for ASN.1 "
                           "entry.\n"));
      end = --start;
      head = ",\n        comment \"";
      tail = "\" ";
    }
  }
  else {
    str = NULL;
    oldpe = pe_flag;
    pe_flag = PE_NONE;
    status = asn_parse(entry, entryend, "seq.id", &str, &strend, NULL);
    pe_flag = oldpe;
    error_test(status == -1, E_PARSEERROR, return STATUS_ERROR,
               print_error("seqfannotate:  Invalid format for ASN.1 "
                           "entry.\n"));

    if (str != NULL) {
      start = end = strend;
      head = ",\n      descr {\n        comment \"";
      tail = "\" } ";
    }
    else {
      oldpe = pe_flag;
      pe_flag = PE_NONE;
      status = asn_parse(entry, entryend, "seq.inst", &str, &strend, NULL);
      pe_flag = oldpe;
      error_test(status == -1, E_PARSEERROR, return STATUS_ERROR,
                 print_error("seqfannotate:  Invalid format for ASN.1 "
                             "entry.\n"));

      if (str != NULL) {
        start = end = str;
        head = "descr {\n        comment \"";
        tail = "\" } ,\n      ";
      }
      else {
        for (start=entryend; start > entry && isspace(*(start-1)); start--) ;
        error_test(*(start-1) != '}', E_PARSEERROR, return STATUS_ERROR,
                   print_error("seqfannotate:  Invalid format for ASN.1 "
                               "entry.\n"));
        end = --start;
        head = ",\n      descr {\n        comment \"";
        tail = "\" } ";
      }
    }
  }

  fwrite(entry, 1, start - entry, fp);

  count = hcount = 0;
  if (!ccflag) {
    fputs(head, fp);
    pos = 17;
  }
  else {
    lastcomment = s = start;
    while (1) {
      for (s+=7; s < end && isspace(*s); s++) ;
      error_test(s == end || (*s != '\'' && *s != '"'),
                 E_PARSEERROR, return STATUS_ERROR,
                 print_error("seqfannotate:  Invalid format for ASN.1 "
                             "entry.\n"));

      tempc = temph = inhistory = blank = 0;
      inhistory = mystreq(s+1, 'S', "SEQIO");
      if (inhistory)
        temph = 1;
      else
        tempc = 1;

      qch = *s;
      for (s++; s < end && *s != qch && *(s-1) != '\\'; s++) {
        if (*s == '\n') {
          if (mystreq(s+1, ' ', " SEQIO") ||
              (inhistory && s[1] == ' ' && s[2] == ' ' &&
               s[3] == ' ' && s[4] != ' ')) {
            inhistory = 1;
            temph++;
          }
          else if (s[1] == ' ' && s[2] == '\n') {
            inhistory = temph = 0;
            blank = 1;
          }
          else
            blank = inhistory = temph = 0;

          tempc++;
        }
      }
      error_test(s == end, E_PARSEERROR, return STATUS_ERROR,
                 print_error("seqfannotate:  Invalid format for ASN.1 "
                             "entry.\n"));

      hcount += temph;
      count += (tempc - temph - blank);
      lastend = s++;

      while (s < end && isspace(*s)) s++;
      error_test(s == end || (*s != ',' && *s != '}'),
                 E_PARSEERROR, return STATUS_ERROR,
                 print_error("seqfannotate:  Invalid format for ASN.1 "
                             "entry.\n"));

      if (*s != ',') {
        if (flag) {
          fwrite(lastcomment, 1, s - lastcomment, fp);
          fputs(",\n        ", fp);
        }
        end = s;
        fputs("comment \"", fp);
        tail = "\" ";
        break;
      }

      for (s++; s < end && isspace(*s); s++) ;
      error_test(s == end, E_PARSEERROR, return STATUS_ERROR,
                 print_error("seqfannotate:  Invalid format for ASN.1 "
                             "entry.\n"));

      if (!mystreq(s, 'C', "COMMENT ")) {
        comment = NULL;
        oldpe = pe_flag;
        pe_flag = PE_NONE;
        status = asn_parse(s, end, "comment", &comment, &deend, NULL);
        pe_flag = oldpe;
        error_test(status == -1, E_PARSEERROR, return STATUS_ERROR,
                   print_error("seqfannotate:  Invalid format for ASN.1 "
                               "entry.\n"));

        if (comment != NULL) {
          if (flag)
            fwrite(lastcomment, 1, s - lastcomment, fp);
          fwrite(s, 1, comment - s, fp);
          lastcomment = s;
        }
        else {
          if (flag)
            fwrite(lastcomment, 1, s - lastcomment, fp);
          end = s;
          fputs("comment \"", fp);
          tail = "\" ,\n        ";
          break;
        }
      }
    }
  }

  putline(fp, 0, 17, newcomment, 78, " ", 1);
  fputs("\" ,\n        comment \"", fp);

  pos = (flag ? count - hcount + 1 : 1);
  fprintf(fp, "SEQIO annotation, lines %d-%d.   %s", pos, pos + ncount - 1,
          get_today());
  fputs(tail, fp);

  /*
   * Output the rest of the entry.
   */
  fwrite(end, 1, entryend - end, fp);

  return STATUS_OK;
}






/*
 * 
 *
 * Section dealing with Database specification and database file locations.
 *
 *
 */


typedef struct {
  char *string;
  int count, size, lastflag, error;
} STRING;


/*
 * addstring
 *
 * This procedure adds the string "s" onto the end of the dynamically 
 * allocated string in the STRING structure, automatically growing that
 * string if necessary.  This just simplifies the database initialization
 * code below.
 *
 * Parameters:  str  -  a STRING structure (where the string in it can be NULL)
 *              s    -  a character string
 *
 * Returns:   nothing
 */

static void addstring(STRING *str, char *s)
{
  int len, newlen, lastflag;
  char *t, *s2, *t2;

  if (str->error)
    return; 

  len = strlen(s);
  if (str->count + len + 2 >= str->size) {
    if (str->size == 0) {
      str->size = 256 + len + 2;
      if ((str->string = (char *) malloc(str->size)) == NULL) {
        str->error = 1;
        return;
      }
    }
    else {
      str->size += str->size + len + 2;
      if ((str->string = (char *) realloc(str->string, str->size)) == NULL) {
        str->error = 1;
        return;
      }
    }
  }

  lastflag = str->lastflag;
  str->lastflag = 0;
  for (s2=s; *s2; s2++) {
    if (*s2 == '@') {
      str->lastflag = 1;
      break;
    }
  }

  if (lastflag && str->lastflag) {
    t2 = NULL;
    for (t=str->string+str->count-1; t > str->string && *(t-1) != '\n'; t--)
      if (*t == '@')
        t2 = t;

    if (s2 - s == t2 - t && strncmp(s, t, s2 - s) == 0) {
      str->string[str->count-1] = ',';
      newlen = len - (s2 - s) - 1;
      memcpy(str->string + str->count, s2 + 1, len - (s2 - s) - 1);
      str->string[str->count+newlen] = '\n';
      str->count += newlen + 1;
      return;
    }
  }

  memcpy(str->string + str->count, s, len);
  str->string[str->count+len] = '\n';
  str->count += len + 1;
}


typedef struct bseq_node {
  char *names, *dir, *fields, *fieldsend, *files;
  int linenum;
  struct bseq_node *next;
} BSEQ_NODE, *BIOSEQ_LIST;

BIOSEQ_LIST bseq_dblist = NULL;
int got_bioseq = 0;

static int int_bioseq_read(char *filename);
static BIOSEQ_LIST dbname_match(BIOSEQ_LIST, char *, int *);
static int sufalias_match(BIOSEQ_LIST, char *, STRING *);
static int file_match(BIOSEQ_LIST, char *, char *, STRING *, int);
static int match_string(char *, char *, char *, char *);
static int complex_match(char *, char *, char *, char *);
static int match_and_expand(BIOSEQ_LIST, char *, char *, char *, char *,
                            STRING *);
static int expand(char *, int, char *, char *, char *, char *, int,
                  STRING *, BIOSEQ_LIST);
static int ident_lookup(char *indexfile, int newfile, char *id,
                        char *idend, STRING *string);
static void ident_close(void);
static int findline(FILE *fp, int low, int high, char *pattern, int patsize);
static int findline2(char *text, int low, int high, char *pattern,
                     int patsize);


int bioseq_read(char *filelist)
{
  int status;
  char *s, *end, buffer[FILENAME_MAX+1];

  if (!ctype_initflag)
    init_ctype();

  param_error(filelist == NULL, return -1, "bioseq_read", "arg 1 is NULL");
  param_error(filelist[0] == '\0', return -1, "bioseq_read",
              "arg 1 is an empty string");

  if (!got_bioseq) {
    got_bioseq = 1;
    if ((s = getenv("BIOSEQ")) != NULL && (status = bioseq_read(s)) < 0)
      return status;
  }

  for (end=filelist; *end; end++) ;
  while (end > filelist) {
    for (s=end-1; s >= filelist && *s != ','; s--) ;
    error_test(s == end - 1, E_PARSEERROR, return -1,
               print_error("%s:  Invalid format of BIOSEQ file list.\n",
                           filelist));

    memcpy(buffer, s + 1, end - s - 1);
    buffer[end-s-1] = '\0';
    if ((status = int_bioseq_read(buffer)) < 0)
      return status;

    end = s;
  }

  return 0;
}


static int int_bioseq_read(char *filename)
{
  int flag, linenum, count, inalias, errflag;
  char *s, *t, *text, *dir, *file, *lastfield;
  BIOSEQ_LIST newlist, newlnode, node, next;

  /*
   * Read in the file.
   */
  text = read_small_file(get_truename(filename, NULL));
  error_test(text == NULL, E_READFAILED, return -1,
             print_error("%s:  Cannot read BIOSEQ file.\n", filename));

  linenum = 1;
  newlist = NULL;
  s = text;
  errflag = 0;

  /*
   * Look for the first header line.
   */
  while (*s != '>') {
    flag = 0;
    for ( ; *s && *s != '\n'; s++) {
      if (*s == '#')
        flag = 1;
      else if (!flag && !isspace(*s) && !errflag) {
        print_error("%s, line %d:  Other text appears before first BIOSEQ "
                    "entry.\n", filename, linenum);
        errflag++;
      }
    }
    if (*s) s++;
    linenum++;
  }

  /*
   * The main loop.
   */
  while (*s) {
    /*
     * Allocate a new node.
     */
    newlnode = (BSEQ_NODE *) malloc(sizeof(BSEQ_NODE));
    if (newlnode == NULL) {
      seqferrno = E_NOMEMORY;
      print_error("Memory Error:  Ran out of memory.\n", filename);
      errflag++;
      goto BSEQ_LOOP_END;
    }

    newlnode->names = s+1;
    newlnode->dir = newlnode->fields = NULL;
    newlnode->fieldsend = newlnode->files = NULL;
    newlnode->linenum = linenum;
    newlnode->next = newlist;
    newlist = newlnode;

    /*
     * Skip to the end of the header line, checking for a directory.
     * NULL-terminate the name list and the directory (if it's
     * there).
     */
    dir = NULL;
    flag = 0;
    for ( ; *s && *s != '\n'; s++) {
      if (*s == ':' && dir == NULL) {
        *s = '\0';
        dir = s+1;
      }

      if (!flag && dir == NULL && !isspace(*s) && *s != ',')
        flag = 1;
    }
    if (!*s) {
      print_error("%s, line %d:  Premature end of BIOSEQ file.\n",
                  filename, linenum);
      errflag++;
      goto BSEQ_LOOP_END;
    }
    if (!flag) {
      print_error("%s, line %d:  No entry name on first line of "
                  "BIOSEQ entry.\n", filename, linenum);
      errflag++;
      if (errflag == 20) {
        print_error("%s, line %d:  Too many errors.\n", filename, linenum);
        goto BSEQ_LOOP_END;
      }
    }

    *s = '\0';

    /*
     * NULL-terminate the directory and check that nothing else appears
     * on the line.
     */
    if (dir != NULL) {
      for ( ; dir < s && isspace(*dir); dir++) ;
      if (dir == s) {
        print_error("%s, line %d:  Invalid first line format of "
                    "BIOSEQ entry.\n", filename, linenum);
        errflag++;
        if (errflag == 20) {
          print_error("%s, line %d:  Too many errors.\n", filename, linenum);
          goto BSEQ_LOOP_END;
        }
      }

      newlnode->dir = dir;

      for ( ; dir < s && !isspace(*dir); dir++) ;
      if (*(dir-1) == dirch)
        *(dir-1) = '\0';

      if (dir < s) {
        *dir = '\0';
        for (dir++; dir < s; dir++) {
          if (!isspace(*dir)) {
            print_error("%s, line %d:  Extra text appears after root "
                        "directory in BIOSEQ entry.\n", filename, linenum);
            errflag++;
            if (errflag == 20) {
              print_error("%s, line %d:  Too many errors.\n", filename,
                          linenum);
              goto BSEQ_LOOP_END;
            }
            break;
          }
        }
      }
    }

    /*
     * Skip through the field lines, NULL-terminating each line and
     * looking for the beginning of the file list.
     */
    s++;
    linenum++;

    newlnode->fields = (*s == '>' ? s : NULL);
    lastfield = NULL;
    for ( ; *s == '>'; s++,linenum++) {
      if (!isspace(s[1])) {
        for (s++; *s && !isspace(*s) && *s != ':'; s++) ;
        if (*s && *s != ':') {
          print_error("%s, line %d:  Invalid format for information field.\n",
                      filename, linenum);
          errflag++;
          if (errflag == 20) {
            print_error("%s, line %d:  Too many errors.\n", filename, linenum);
            goto BSEQ_LOOP_END;
          }
        }

        while (*s && *s != '\n') s++;

        if (!*s) {
          print_error("%s, line %d:  Premature end of BIOSEQ entry.\n",
                      filename, linenum);
          errflag++;
          goto BSEQ_LOOP_END;
        }
        *s = '\0';

        for (t=s; isspace(*(t-1)); t--) 
          *(t-1) = '\0';
        lastfield = t;
      }
      else {
        if (lastfield == NULL) {
          print_error("%s, line %d:  No information fieldname given.\n",
                      filename, linenum);
          errflag++;
          if (errflag == 20) {
            print_error("%s, line %d:  Too many errors.\n", filename, linenum);
            goto BSEQ_LOOP_END;
          }

          while (*s && *s != '\n') s++;
        }
        else {
          for (s++; *s && isspace(*s) && *s != '\n'; s++) 
            *(s-1) = '\0';

          *lastfield++ = ' ';
          while (*s && *s != '\n') {
            *lastfield++ = *s;
            *s++ = '\0';
          }
          *lastfield = '\0';

          if (!*s) {
            print_error("%s, line %d:  Premature end of BIOSEQ entry.\n",
                        filename, linenum);
            errflag++;
            goto BSEQ_LOOP_END;
          }
          *s = '\0';

          for (t=lastfield; isspace(*(t-1)); t--) 
            *(t-1) = '\0';
          lastfield = t;
        }
      }
    }
    
    if (newlnode->fields)
      newlnode->fieldsend = s;

    /*
     * Scan the files lines, performing minimal syntax checking and
     * looking either for the end of the file or a line beginning
     * with '>'.  Don't NULL-terminate lines here until the very end.
     */
    newlnode->files = s;
    count = 0;
    inalias = 0;
    flag = 0;
    while (*s) {
      while (*s && (*s == ' ' || *s == '\t' || *s == ',')) s++;
      if (!*s)
        break;

      if (*s == '\n') {
        if (!s[1] || s[1] == '>')
          break;
        else {
          s++;
          linenum++;
          continue;
        }
      }

      if (*s == '#') {
        while (*s && *s != '\n') s++;
        continue;
      }

      file = s;
      while (*s && !isspace(*s) && *s != ',' && *s != '#' && *s != '(' &&
             *s != ')')
        s++;

      if (file != s)
        flag = 1;

      if (file != s && *s != '(' && *(s-1) == dirch) {
        print_error("%s, line %d:  Filenames in BIOSEQ entries cannot end "
                    "with `%c'.\n", filename, linenum, dirch);
        errflag++;
        if (errflag == 20) {
          print_error("%s, line %d:  Too many errors.\n", filename, linenum);
          goto BSEQ_LOOP_END;
        }
      }

      if (!*s || isspace(*s) || *s == ',')
        continue;

      switch (*s) {
      case '#':
        while (*s && *s != '\n') s++;
        break;

      case '(':
        if (inalias) {
          print_error("%s, line %d:  Parentheses not allowed "
                      "inside alias definition.\n", filename, linenum);
          errflag++;
          if (errflag == 20) {
            print_error("%s, line %d:  Too many errors.\n", filename,
                        linenum);
            goto BSEQ_LOOP_END;
          }
        }
        else if (*(s-1) == ':') {
          if (count != 0) {
            print_error("%s, line %d:  `:(' may not appear inside "
                        "other parentheses.\n", filename, linenum);
            errflag++;
            if (errflag == 20) {
              print_error("%s, line %d:  Too many errors.\n", filename,
                          linenum);
              goto BSEQ_LOOP_END;
            }
          }
          else {
            inalias = 1;
            for (t=file; t < s-1; t++) {
              if (*t == '*' || *t == '?' || *t == dirch ||
                  (*t == '~' && t != file)) {
                print_error("%s, line %d:  `%c' not permitted in alias "
                            "name.\n", filename, linenum, *t);
                errflag++;
                if (errflag == 20) {
                  print_error("%s, line %d:  Too many errors.\n",
                              filename, linenum);
                  goto BSEQ_LOOP_END;
                }
                break;
              }
            }
          }
        }
        else if (*(s-1) != dirch) {
          print_error("%s, line %d:  Parse error at `%c('.\n",
                      filename, linenum, *(s-1));
          errflag++;
          if (errflag == 20) {
            print_error("%s, line %d:  Too many errors.\n", filename,
                        linenum);
            goto BSEQ_LOOP_END;
          }
        }
        count++;
        s++;
        break;

      case ')':
        if (count == 0) {
          print_error("%s, line %d:  Unmatched `)'.\n", filename, linenum);
          errflag++;
          if (errflag == 20) {
            print_error("%s, line %d:  Too many errors.\n", filename,
                        linenum);
            goto BSEQ_LOOP_END;
          }
        }
        else
          count--;
        inalias = 0;
        s++;
        break;
      }
    }

    if (inalias) {
      print_error("%s, line %d:  End of BIOSEQ entry reached "
                  "inside alias definition.\n", filename, linenum);
      errflag++;
      if (errflag == 20) {
        print_error("%s, line %d:  Too many errors.\n", filename, linenum);
        goto BSEQ_LOOP_END;
      }
    }
    else if (count != 0) {
      print_error("%s, line %d:  End of BIOSEQ entry reached "
                  "with unmatched '('.\n", filename, linenum);
      errflag++;
      if (errflag == 20) {
        print_error("%s, line %d:  Too many errors.\n", filename, linenum);
        goto BSEQ_LOOP_END;
      }
    }

    if (!flag)
      newlnode->files = NULL;

    if (*s) {
      *s = '\0';
      s++;
    }
  }
BSEQ_LOOP_END:

  /*
   * If a parse error occurred while checking the file, free all memory
   * allocated and return an error value.
   */
  if (errflag > 0) {
    while (newlist != NULL) {
      newlnode = newlist->next;
      free(newlist);
      newlist = newlnode;
    }

    if (seqferrno == E_NOERROR)
      seqferrno = E_PARSEERROR;
    return -1;
  }

  /*
   * Now, add the entries to the list of databases.
   *
   * A separate pass is used here (instead of combining this with the
   * above loop) so that if errors occur in the file, the original
   * database list is not affected.
   */
  for (node=newlist; node != NULL; node=next) {
    next = node->next;
    node->next = bseq_dblist;
    bseq_dblist = node;
  }

  return 0;
}


int bioseq_check(char *dbspec)
{
  int i, pos, len, namelen;
  char *s, *dbname, buffer[8];
  BIOSEQ_LIST node;

  if (!ctype_initflag)
    init_ctype();

  if (dbspec == NULL || dbspec[0] == '\0')
    return 0;

  if (!got_bioseq) {
    got_bioseq = 1;
    if ((s = getenv("BIOSEQ")) != NULL && bioseq_read(s) < 0)
      return 0;
  }

  if (dbname_match(bseq_dblist, dbspec, NULL) != NULL)
    return 1;

  pos = -1;
  for (s=dbspec,len=0; len < 6 && *s; s++,len++)
    if (pos == -1 && *s == ':')
      pos = len;

  if (pos >= 2 && pos <= 4) {
    memcpy(buffer, dbspec, pos);
    buffer[pos] = '\0';

    for (i=0; i < idpref_table_size; i++)
      if (mycasecmp(buffer, idpref_table[i].idprefix) == 0)
        break;

    if (i < idpref_table_size) {
      node = bseq_dblist;
      namelen = strlen(idpref_table[i].dbname);
      while ((node = dbname_match(node,
                                  idpref_table[i].dbname, &len)) != NULL) {
        if (len == namelen)
          return 1;
        node = node->next;
      }
      return 0;
    }

    if ((dbname = bioseq_matchinfo("IdPrefix", buffer)) != NULL) {
      free(dbname);
      return 1;
    }
  }

  return 0;
}


char *bioseq_info(char *dbspec, char *fieldname)
{
  int i, flag, isroot, pos, len, copyflag, namelen;
  char *s, *t, *temp, *dbname, buffer[8];
  BIOSEQ_LIST node;

  if (!ctype_initflag)
    init_ctype();

  param_error(dbspec == NULL, return NULL, "bioseq_info", "arg 1 is NULL");
  param_error(dbspec[0] == '\0', return NULL, "bioseq_info",
              "arg 1 is an empty string");
  param_error(fieldname == NULL, return NULL, "bioseq_info", "arg 2 is NULL");
  param_error(fieldname[0] == '\0', return NULL, "bioseq_info",
              "arg 2 is an empty string");

  if (!got_bioseq) {
    got_bioseq = 1;
    if ((s = getenv("BIOSEQ")) != NULL && bioseq_read(s) < 0)
      return NULL;
  }

  error_test(bseq_dblist == NULL, E_DBFILEERROR, return NULL,
             print_error("%s:  No databases are known.  (Env. variable BIOSEQ "
                         "may not be set.)\n", dbspec));

  isroot = (mycasecmp(fieldname, "Root") == 0);

  /*
   * Find the BIOSEQ entry by trying to match to a database name.
   */
  flag = 0;
  node = bseq_dblist;
  while ((node = dbname_match(node, dbspec, NULL)) != NULL) {
    if (isroot) {
      if (node->dir != NULL) {
        temp = mystrdup(node->dir);
        memory_error(temp == NULL, return NULL);
        return temp;
      }
    }
    else {
      /*
       * Check the fields of the matching entry for a match to the parameter.
       */
      if (node->fields != NULL) {
        for (s=node->fields; s < node->fieldsend; ) {
          for (t=fieldname,s++; *t && *s != ':'; t++,s++)
            if (toupper(*t) != toupper(*s))
              break;

          if (!*t && *s == ':') {
            for (s++; *s && isspace(*s); s++) ;
            temp = mystrdup(s);
            memory_error(temp == NULL, return NULL);
            return temp;
          }

          while (*s) s++;
          while (!*s) s++;
        }
      }
    }

    flag = 1;
    node = node->next;
  }
  if (flag)
    return NULL;

  /*
   * Next, try to match to an identifier prefix.
   */
  pos = -1;
  for (s=dbspec,len=0; len < 6 && *s; s++,len++)
    if (pos == -1 && *s == ':')
      pos = len;

  if (pos >= 2 && pos <= 4) {
    memcpy(buffer, dbspec, pos);
    buffer[pos] = '\0';

    for (i=0; i < idpref_table_size; i++)
      if (mycasecmp(buffer, idpref_table[i].idprefix) == 0)
        break;

    if (i < idpref_table_size) {
      dbname = idpref_table[i].dbname;
      copyflag = 0;
    }
    else {
      dbname = bioseq_matchinfo("IdPrefix", buffer);
      if (dbname == NULL)
        return NULL;
      copyflag = 1;
    }

    /*
     * Now, try to search for the information field using this database name.
     */
    node = bseq_dblist;
    namelen = strlen(dbname);
    while ((node = dbname_match(node, dbname, &len)) != NULL) {
      if (len == namelen) {
        if (isroot) {
          if (copyflag)
            free(dbname);

          if (node->dir == NULL)
            return NULL;
          else {
            temp = mystrdup(node->dir);
            memory_error(temp == NULL, return NULL);
            return temp;
          }
        }

        /*
         * Check the fields of the matching entry for a match to the parameter.
         */
        if (node->fields != NULL) {
          for (s=node->fields; s < node->fieldsend; ) {
            for (t=fieldname,s++; *t && *s != ':'; t++,s++)
              if (toupper(*t) != toupper(*s))
                break;

            if (!*t && *s == ':') {
              if (copyflag)
                free(dbname);
              for (s++; *s && isspace(*s); s++) ;
              temp = mystrdup(s);
              memory_error(temp == NULL, return NULL);
              return temp;
            }

            while (*s) s++;
            while (!*s) s++;
          }
        }
      }

      node = node->next;
      flag = 1;
    }

    if (copyflag)
      free(dbname);
  }

  error_test(!flag, E_DBPARSEERROR, return NULL,
             print_error("%s:  No BIOSEQ entry matches database "
                         "specification.\n", dbspec));

  return NULL;
}


char *bioseq_matchinfo(char *fieldname, char *fieldvalue)
{
  int isroot;
  char *s, *t, *temp;
  BIOSEQ_LIST node;

  if (!ctype_initflag)
    init_ctype();

  param_error(fieldname == NULL, return NULL, "bioseq_matchinfo",
              "arg 1 is NULL");
  param_error(fieldname[0] == '\0', return NULL, "bioseq_matchinfo",
              "arg 1 is an empty string");
  param_error(fieldvalue == NULL, return NULL, "bioseq_matchinfo",
              "arg 2 is NULL");
  param_error(fieldvalue[0] == '\0', return NULL, "bioseq_matchinfo",
              "arg 2 is an empty string");

  if (!got_bioseq) {
    got_bioseq = 1;
    if ((s = getenv("BIOSEQ")) != NULL && bioseq_read(s) < 0)
      return NULL;
  }

  error_test(bseq_dblist == NULL, E_DBFILEERROR, return NULL,
             print_error("Error:  No databases are known.  (Env. variable "
                         "BIOSEQ may not be set.)\n"));

  isroot = (mycasecmp(fieldname, "Root") == 0);

  for (node=bseq_dblist; node != NULL; node=node->next) {
    if (isroot && node->dir && mycasecmp(node->dir, fieldvalue) == 0) {
      for (s=node->names; *s && (isspace(*s) || *s == ','); s++) ;
      for (t=s; *s && !isspace(*s) && *s != ','; s++) ;
      temp = mystrdup2(t, s);
      memory_error(temp == NULL, return NULL);
      return temp;
    }
      
    if (isroot || node->fields == NULL)
      continue;

    for (s=node->fields; s < node->fieldsend; ) {
      for (t=fieldname,s++; *t && *s != ':'; t++,s++)
        if (toupper(*t) != toupper(*s))
          break;

      if (!*t && *s == ':') {
        for (s++; *s && isspace(*s); s++) ;
        if (mycasecmp(s, fieldvalue) == 0) {
          for (s=node->names; *s && (isspace(*s) || *s == ','); s++) ;
          for (t=s; *s && !isspace(*s) && *s != ','; s++) ;
          temp = mystrdup2(t, s);
          memory_error(temp == NULL, return NULL);
          return temp;
        }
      }

      while (*s) s++;
      while (!*s) s++;
    }
  }
  return NULL;
}
  

char *bioseq_parse(char *dbspec)
{
  int i, len, namelen, matchlen, idpreflen, dbspeclen;
  int status, flag, lookupflag, copyflag;
  char *s, *t, *t2, *fieldname, *dbname, indexpath[FILENAME_MAX];
  char buffer[FILENAME_MAX];
  BIOSEQ_LIST node, filenode;
  STRING string = { NULL, 0, 0, 0, 0 };

  if (!ctype_initflag)
    init_ctype();

  param_error(dbspec == NULL, return NULL, "bioseq_parse",
              "arg 1 is NULL");
  param_error(dbspec[0] == '\0', return NULL, "bioseq_parse",
              "arg 1 is an empty string");

  if (!got_bioseq) {
    got_bioseq = 1;
    if ((s = getenv("BIOSEQ")) != NULL && bioseq_read(s) < 0)
      return NULL;
  }

  error_test(bseq_dblist == NULL, E_DBFILEERROR, return NULL,
             print_error("%s:  No databases are known.  (Env. variable BIOSEQ "
                         "may not be set.)\n", dbspec));

  /*
   * Find the node which contains the list of files, and find the node
   * which contains an index.
   */
  dbspeclen = strlen(dbspec);
  fieldname = "Index";
  indexpath[0] = '\0';
  filenode = NULL;
  matchlen = 0;

  flag = 0;
  node = bseq_dblist;
  while ((filenode == NULL || indexpath[0] == '\0') &&
         (node = dbname_match(node, dbspec, &len)) != NULL) {
    if (!filenode && node->files) {
      filenode = node;
      matchlen = len;
    }

    if (len < dbspeclen && dbspec[len] == ':' &&
        !indexpath[0] && node->fields != NULL) {
      for (s=node->fields; s < node->fieldsend; ) {
        for (t=fieldname,s++; *t && *s != ':'; t++,s++)
          if (toupper(*t) != toupper(*s))
            break;

        if (!*t && *s == ':') {
          for (s++; *s && isspace(*s); s++) ;
          if (is_absolute(s) || node->dir == NULL)
            strcpy(indexpath, s);
          else {
            strcpy(indexpath, node->dir);
            for (t2=indexpath; *t2; t2++) ;
            *t2++ = dirch;
            strcpy(t2, s);
          }
          break;
        }

        while (*s) s++;
        while (!*s) s++;
      }
    }

    flag = 1;
    node = node->next;
  }

  /*
   * If that failed, try to match to an identifier prefix.
   */
  if (!flag) {
    idpreflen = -1;
    for (s=dbspec,len=0; len < 6 && *s; s++,len++)
      if (idpreflen == -1 && *s == ':')
        idpreflen = len;

    if (idpreflen >= 2 && idpreflen <= 4) {
      memcpy(buffer, dbspec, idpreflen);
      buffer[idpreflen] = '\0';

      for (i=0; i < idpref_table_size; i++)
        if (mycasecmp(buffer, idpref_table[i].idprefix) == 0)
          break;

      if (i < idpref_table_size) {
        dbname = idpref_table[i].dbname;
        copyflag = 0;
      }
      else {
        dbname = bioseq_matchinfo("IdPrefix", buffer);
        if (dbname == NULL)
          return NULL;
        copyflag = 1;
      }

      /*
       * Now, try to search for the information field using this database name.
       */
      node = bseq_dblist;
      namelen = strlen(dbname);
      while ((filenode == NULL || indexpath[0] == '\0') &&
             (node = dbname_match(node, dbname, &len)) != NULL) {
        if (len == namelen) {
          if (!filenode && node->files) {
            filenode = node;
            matchlen = idpreflen;
          }

          if (!indexpath[0] && node->fields != NULL) {
            for (s=node->fields; s < node->fieldsend; ) {
              for (t=fieldname,s++; *t && *s != ':'; t++,s++)
                if (toupper(*t) != toupper(*s))
                  break;

              if (!*t && *s == ':') {
                for (s++; *s && isspace(*s); s++) ;
                if (is_absolute(s) || node->dir == NULL)
                  strcpy(indexpath, s);
                else {
                  strcpy(indexpath, node->dir);
                  for (t2=indexpath; *t2; t2++) ;
                  *t2++ = dirch;
                  strcpy(t2, s);
                }
                break;
              }

              while (*s) s++;
              while (!*s) s++;
            }
          }
        }
      
        node = node->next;
        flag = 1;
      }

      if (copyflag)
        free(dbname);
    }
  }

  error_test(!flag && !filenode && !indexpath[0], E_DBPARSEERROR, return NULL,
             print_error("%s:  No BIOSEQ entry matches database "
                         "specification.\n", dbspec));

  /*
   * Check for the three forms of dbspec:
   *   1) just the database name.
   *        - Check to see if "~" is an alias.  If so, then expand that
   *          alias.  Otherwise, take all of the files in the entry.
   *   2) the database name with a suffix alias.
   *        - Look for that suffix alias, and expand it.
   *   3) the database name, a ':' and list of files and aliases.
   *        - Loop through the list of files and aliases, and search
   *          the entry for each of them.
   */
  status = 0;
  if (filenode && dbspec[matchlen] == '\0') {
    status = sufalias_match(filenode, "", &string);
    if (status == 0)
      status = file_match(filenode, NULL, NULL, &string, 0);
  }
  else if (filenode && dbspec[matchlen] != ':') {
    status = sufalias_match(filenode, dbspec+len, &string);
  }
  else {
    for (s=dbspec; *s && *s != ':'; s++) ;
    if (*s)
      for (s++; *s && (isspace(*s) || *s == ','); s++) ;
    error_test(!*s, E_PARSEERROR, return NULL, 
               print_error("%s:  Invalid database specification.\n",
                           dbspec));

    lookupflag = 0;
    while (*s) {
      for (t=s; *s && !isspace(*s) && *s != ','; s++) ;
      error_test(t == s, E_PARSEERROR, return NULL, 
                 print_error("%s:  Invalid database specification.\n",
                             dbspec));

      status = 0;
      if (filenode)
        status = file_match(filenode, t, s, &string, 0);

      if (status == 0 && indexpath[0]) {
        status = ident_lookup(indexpath, !lookupflag, t, s, &string);
        lookupflag = 1;
      }

      if (status == -1)
        break;
      else if (status == 0) {
        memcpy(buffer, t, s - t);
        buffer[s-t] = '\0';

        set_error(E_DBFILEERROR);
        print_warning("%s:  Unable to find `%s' in database.\n", dbspec,
                      buffer);
      }
      
      while (*s && (isspace(*s) || *s == ',')) s++;
    }

    if (lookupflag)
      ident_close();
  }

  if (status == -1) {
    if (string.string)
      free(string.string);
    return NULL;
  }

  memory_error(string.error, return NULL);
  if (string.string == NULL)
    return NULL;

  string.string[string.count] = '\0';
  return string.string;
}


static BIOSEQ_LIST dbname_match(BIOSEQ_LIST node, char *dbname, int *len_out)
{
  int status;
  char *s, *t;

  for ( ; node != NULL; node=node->next) {
    for (s=node->names; *s && (isspace(*s) || *s == ','); s++) ;
    while (*s) {
      t = dbname;
      for ( ; *s && !isspace(*s) && *s != ',' && *t && *t != ':'; s++,t++)
        if (toupper(*s) != toupper(*t))
          break;

      if (!*s || isspace(*s) || *s == ',') {
        if (!*t || *t == ':' || node->files == NULL) {
          if (len_out)  *len_out = t - dbname;
          return node;
        }
        else if ((status = sufalias_match(node, t, NULL)) != 0) {
          if (status == 1) {
            if (len_out)  *len_out = t - dbname;
            return node;
          }
          else
            return NULL;
        }
      }
    
      while (*s && !isspace(*s) && *s != ',') s++;
      while (*s && (isspace(*s) || *s == ',')) s++;

    }
  }

  return NULL;
}


static int sufalias_match(BIOSEQ_LIST node, char *suffix, STRING *str)
{
  int count;
  char *s, *t, *s2, *t2, *file;

  if (node->files == NULL)
    return 0;

  s = node->files;
  while (*s) {
    while (*s && (isspace(*s) || *s == ',')) s++;
    if (!*s)
      break;

    if (*s == '#') {
      while (*s && *s != '\n') s++;
      continue;
    }

    file = s;
    while (*s && !isspace(*s) && *s != ',' && *s != '#' && *s != '(')
      s++;

    if (!*s)
      break;
    else if (isspace(*s) || *s == ',')
      continue;
    else if (*s == '#') {
      while (*s && *s != '\n') s++;
      continue;
    }

    /*
     * From this point on, *s must be '('.
     */
    if (*(s-1) == ':' && *file == '~') {
      for (s2=file+1,t2=suffix; s2 < s - 1 && *t2; s2++,t2++)
        if (toupper(*s2) != toupper(*t2))
          break;

      if (s2 == s - 1 && !*t2) {
        if (str != NULL) {
          for (s++; *s && (isspace(*s) || *s == ','); s++) ;
          while (*s && *s != ')') {
            if (*s == '#')
              while (*s && *s != '\n') s++;
            else {
              t = s;
              while (*s && !isspace(*s) && *s != ',' && *s != ')' && *s != '#')
                s++;

              if (file_match(node, t, s, str, 0) < 0)
                return -1;
            }

            while (*s && (isspace(*s) || *s == ',')) s++;
          }
        }

        return 1;
      }
    }

    count = 0;
    while (*s && (*s != ')' || --count != 0)) {
      if (*s == '(')
        count++;
      else if (*s == '#') {
        while (*s && *s != '\n') s++;
        if (!*s)
          break;
      }

      s++;
    }
    program_error(!*s, return -1,
                  print_error("    sufalias_match:  Previous syntax checking"
                              " done incorrectly.\n"));
    s++;
  }

  return 0;
}


static int file_match(BIOSEQ_LIST node, char *start, char *end, STRING *str,
                      int depth)
{
  int pos, inalias, found, status;
  char *s, *s2, *t2, *file;
  char path[FILENAME_MAX+1];

  error_test(depth >= 10, E_PARSEERROR, return -1,
             print_error("%s:  Runaway alias recursion in BIOSEQ entry.\n",
                         node->names));

  found = 0;
  inalias = 0;
  pos = 0;
  s = node->files;
  while (*s) {
    while (*s && (isspace(*s) || *s == ',')) s++;
    if (!*s)
      break;

    if (*s == '#') {
      while (*s && *s != '\n') s++;
      continue;
    }

    file = s;
    while (*s && !isspace(*s) && *s != ',' && *s != '#' &&
           *s != '(' && *s != ')')
      s++;

    if (*s == ')' && file == s) {
      if (inalias)
        inalias = 0;
      else {
        program_error(pos == 0, return -1, 
                      print_error("    file_match:  Previous syntax checking"
                                  " done incorrectly.\n"));
        for (pos--,t2=path+pos; pos > 0 && *(t2-1) != '('; t2--,pos--) ;
      }
      s++;
      continue;
    }

    if (*s != '(') {
      if (inalias)
        status = file_match(node, file, s, str, depth+1);
      else if (pos == 0)
        status = match_and_expand(node, file, s, start, end, str);
      else {
        for (s2=file,t2=path+pos; s2 < s; s2++,t2++)
          *t2 = *s2;
        *t2 = '\0';

        status = match_and_expand(node, path, t2, start, end, str);
      }

      if (status == -1)
        return -1;
      else if (status == 1)
        found = 1;

      if (!*s)
        break;
      else if (isspace(*s) || *s == ',' || *s == ')')
        continue;
      else if (*s == '#') {
        while (*s && *s != '\n') s++;
        continue;
      }
    }

    program_error(inalias, return -1,
                  print_error("    file_match:  Previous syntax checking"
                              " done incorrectly.\n"));

    /*
     * From this point on, *s must be '('.
     */
    if (*(s-1) == dirch) {
      for (s2=file,t2=path+pos; s2 < s - 1; s2++,t2++,pos++)
        *t2 = *s2;
      *t2++ = '(';
      pos++;
      s++;
      continue;
    }

    program_error(*(s-1) != ':', return -1,
                  print_error("    file_match:  Previous syntax checking"
                              " done incorrectly.\n"));

    if (*file != '~' && start != NULL) {
      for (s2=file,t2=start; s2 < s - 1 && t2 < end; s2++,t2++)
        if (toupper(*s2) != toupper(*t2))
          break;

      if (s2 == s - 1 && t2 == end) {
        if (str == NULL)
          return 1;
        else {
          inalias = 1;
          found = 1;
          s++;
          continue;
        }
      }
    }

    for (s++; *s && *s != ')'; s++) {
      if (*s == '#') {
        while (*s && *s != '\n') s++;
        if (!*s)
          break;
      }
      else if (*s == '(') {
        program_error(1, return -1,
                      print_error("    file_match:  Previous syntax checking"
                                  " done incorrectly.\n"));
      }
    }
    program_error(!*s, return -1,
                  print_error("    file_match:  Previous syntax checking"
                              " done incorrectly.\n"));
    s++;
  }
  program_error(inalias, return -1,
                print_error("    file_match:  Previous syntax checking"
                            " done incorrectly.\n"));

  return found;
}


static int match_and_expand(BIOSEQ_LIST node, char *file, char *fileend,
                            char *pattern, char *patend, STRING *str)
{
  int pathflag, count, flag;
  char *s, *t, *s2, *t2, *send, *tend, *last, *filestart;
  char path[FILENAME_MAX];

  /*
   * If a user-specified pattern has been given to bioseq_parse (i.e.,
   * specifying a particular file or wildcarded file pattern in the
   * database), perform an initial match of the filename against that
   * user-specified pattern.
   *
   * This matching takes into account that both the user-specified pattern
   * and the filename (gotten from the BIOSEQ entry) can contain wildcards.
   */
  pathflag = 0;
  if (pattern != NULL) {
    for (s=pattern; s < patend; s++) {
      if (*s == dirch) {
        pathflag = 1;
        break;
      }
    }

    if (pathflag) {
      s = file;
      t = pattern;
    }
    else {
      last = file;
      for (s=file; s < fileend; s++)
        if (*s == dirch || *s == '(')
          last = s+1;

      program_error(last == fileend, return -1,
                    print_error("    match_and_expand:  Incorrect syntax "
                                "checking of BIOSEQ entry.\n"));

      s = last;
      t = pattern;
    }

    for ( ; s < fileend && t < patend; s++,t++) {
      if ((toupper(*s) == toupper(*t) && *s != '*') ||
          (*s == '(' && *t == dirch) || (*s == '?' && *t != dirch) || 
          (*t == '?' && *s != dirch && *s != '('))
        continue;

      if (*s != '*' && *t != '*')
        return 0;

      count = 0;
      for (s2=s+1; s2 < fileend && *s2 != dirch && *s2 != '('; s2++)
        if (*s2 == '*')
          count++;
      for (t2=t+1; t2 < patend && *t2 != dirch; t2++)
        if (*t2 == '*')
          count++;

      if (count == 0) {
        send = s2;
        tend = t2;
        for (s2--,t2--; s2 > s && t2 > t; s2--,t2--)
          if (!(toupper(*s2) == toupper(*t2) || *s2 == '?' || *t2 == '?'))
            return 0;

        if (!((*s == '*' && s2 == s) || (*t == '*' && t2 == t)))
          return 0;

        s = send;
        t = tend;
      }
      else {
        if (!complex_match(s, s2, t, t2))
          return 0;

        s = s2;
        t = t2;
      }
    }

    if ((s < fileend && (*s != '*' || s + 1 < fileend)) ||
        (t < patend && (*t != '*' || t + 1 < patend)))
      return 0;
  }

  /*
   * Build the pathname for the new file, converting all of the '('
   * used in the "file" string as stack markers (in file_match)
   * to '/' (or '\\' in Windows).
   *
   * Also, check the copied string to see if any wildcards appear in
   * it.
   */
  s = path;
  if (node->dir) {
    for (t=node->dir; *t; s++,t++) *s = *t;
    *s++ = dirch;
  }
  filestart = s;

  flag = 0;
  for (t=file; t < fileend; t++,s++) {
    if (*t == '(')
      *s = dirch;
    else {
      *s = *t;
      if (*s == '?' || *s == '*')
        flag = 1;
    }
  }
  *s = '\0';

  if (is_absolute(filestart)) {
    strcpy(path, filestart);
    filestart = path;
  }

  /*
   * Either add the string (if no wildcards), or perform a directory list
   * expansion of the wildcards in the path (and, of course, match each
   * of those against the user-specified pattern).
   */
  if (flag)
    return expand(path, 0, filestart, s, pattern, patend, pathflag, str, node);
  else {
    if (seqfisafile(path))
      addstring(str, path);
    else {
      set_error(E_DBFILEERROR);
      print_warning("Warning:  %s:  File listed in BIOSEQ entry `%s' does not "
                    "exist.\n", path, node->names);
    }

    return 1;
  }
}


static int expand(char *initpath, int initpos, char *pat, char *patend,
                  char *fpat, char *fpatend, int pathflag, STRING *str,
                  BIOSEQ_LIST node)
{
  static char path[FILENAME_MAX];
  int flag, status, found;
  char *s, *t, *ptr, *s2, *t2, *ptr2, *dname, *end, *d2;
  DIRPTR dp;

  if (initpos == 0) {
    for (ptr=path,t=initpath; t < pat; ptr++,t++)
      *ptr = *t;
  }
  else
    ptr = path + initpos;

  s = s2 = pat;
  t = t2 = fpat;
  flag = 0;
  while (s < patend) {
    for (s2=s; s < patend && *s != dirch; s++) 
      if (*s == '?' || *s == '*')
        flag = 1;

    if (pathflag)
      while (t < patend && *t != dirch) t++;

    if (flag)
      break;

    for (s=s2; s < patend && *s != dirch; s++,ptr++)
      *ptr = *s;
    if (s < patend)
      *ptr++ = *s++;

    if (pathflag)
      t2 = ++t;
  }

  if (!flag && s == patend) {
    *ptr = '\0';
    if (seqfisafile(path))
      addstring(str, path);
    else {
      set_error(E_DBFILEERROR);
      print_warning("Warning:  %s:  File listed in BIOSEQ entry `%s' does not "
                    "exist.\n", path, node->names);
    }

    return 1;
  }

  /*
   * If we hit a wildcard, read the files/sub-dirs in the path so far
   * and recurse on any matches to the patterns for that directory name.
   */
  if (!pathflag)
    t = fpatend;

  if (ptr == path)
    status = open_directory(".", &dp);
  else {
    *(ptr-1) = '\0';
    status = open_directory(get_truename(path, ptr), &dp);
    *(ptr-1) = dirch;
  }
  error_test(status != STATUS_OK, E_OPENFAILED, return -1,
             print_error("%s:  Cannot open directory listed in "
                         " BIOSEQ entry `%s'.\n", (ptr == path ? "." : path),
                         node->names));

  found = 0;
  while ((dname = read_dirname(dp)) != NULL) {
    if (dname[0] == '.' &&
        (dname[1] == '\0' || (dname[1] == '.' && dname[2] == '\0')))
      continue;

    for (end=dname; *end; end++) ;

    status = 0;
    if (match_string(dname, end, s2, s) && 
        (!pathflag || match_string(dname, end, t2, t))) {
      for (ptr2=ptr,d2=dname; d2 < end; ptr2++,d2++)
        *ptr2 = *d2;
      *ptr2 = '\0';

      if (s == patend) {
        if (pathflag || match_string(dname, end, t2, t)) {
          addstring(str, path);
          status = 1;
        }
      }
      else {
        *ptr2++ = dirch;
        status = expand(NULL, ptr2 - path, s+1, patend,
                        (pathflag ? t+1 : fpat), fpatend, pathflag, str, node);
      }
    }

    if (status == -1) {
      found = -1;
      break;
    }
    else if (status == 1)
      found = 1;
  }
  close_directory(dp);

  return found;
}


static int match_string(char *s, char *send, char *t, char *tend)
{
  int count;
  char *s2, *t2;

  for ( ; s < send && t < tend; s++,t++) {
    if ((toupper(*s) == toupper(*t) && *s != '*') ||
        (*s == '?' && *t != dirch) || (*t == '?' && *s != dirch))
      continue;

    if (*s != '*' && *t != '*')
      return 0;

    count = 0;
    for (s2=s+1; s2 < send && *s2 != dirch; s2++)
      if (*s2 == '*')
        count++;
    for (t2=t+1; t2 < tend && *t2 != dirch; t2++)
      if (*t2 == '*')
        count++;

    if (count == 0) {
      send = s2;
      tend = t2;
      for (s2--,t2--; s2 > s && t2 > t; s2--,t2--)
        if (!(toupper(*s2) == toupper(*t2) || *s2 == '?' || *t2 == '?'))
          return 0;

      if (!((*s == '*' && s2 == s) || (*t == '*' && t2 == t)))
        return 0;

      s = send;
      t = tend;
    }
    else {
      if (!complex_match(s, s2, t, t2))
        return 0;

      s = s2;
      t = t2;
    }
  }

  return 1;
}


static int complex_match(char *s, char *send, char *t, char *tend)
{
  char *s2, *t2, *temp;

  /*
   * If s is sitting on an asterisk, loop through the substrings on t (using
   * loop variable "temp"), trying to match the strings from s+1 and temp 
   * (thus, matching the asterisk at s to the substring from t to temp-1).
   * If, in that matching, another asterisk is reached, recurse to match
   * any possible substring of the other string against that asterisk.
   */
  if (*s == '*') {
    if (!s[1])
      return 1;

    for (temp=t; temp < tend; temp++) {
      for (s2=s+1,t2=temp; s2 < send && t2 < tend; s2++,t2++) {
        if ((toupper(*s2) == toupper(*t2) && *s2 != '*') ||
            *s2 == '?' || *t2 == '?')
          continue;

        if ((*s2 == '*' || *t2 == '*') &&
            complex_match(s2, send, t2, tend))
          return 1;
        else 
          break;
      }
      if (s2 == send && t2 == tend)
        return 1;
    }
  }

  /*
   * Do the vice versa thing with the asterisk sitting on t.
   */
  if (*t == '*') {
    if (!t[1])
      return 1;

    temp = (*s == '*' ? s+1 : s);
    for ( ; temp < send; temp++) {
      for (s2=temp,t2=t+1; s2 < send && t2 < tend; s2++,t2++) {
        if ((toupper(*s2) == toupper(*t2) && *s2 != '*') ||
            *s2 == '?' || *t2 == '?')
          continue;

        if ((*s2 == '*' || *t2 == '*') &&
            complex_match(s2, send, t2, tend))
          return 1;
        else 
          break;
      }
      if (s2 == send && t2 == tend)
        return 1;
    }
  }

  return 0;
}


static FILE *idxfp = NULL;
static char *idxbuffer = NULL;
int idxtype, idxsize, idxstart, idxbufsize;

static int ident_lookup(char *indexfile, int newfile, char *id,
                        char *idend, STRING *string)
{
  int size, count, offset, status, patsize, filenum, fileoffset, fileidlen;
  char *s, *s2, *t, *t2, *end, *end2, *temp, line[128], pattern[128];
  char buffer[FILENAME_MAX+32];

  if (idend == NULL)
    for (idend=id; *idend; idend++) ;

  for (s=pattern,t=id; t < idend && *t != '*' && *t != '?'; s++,t++)
    *s = *t;
  *s = '\0';
  patsize = s - pattern;

  /*
   * If we're starting queries on a new index file, open up that
   * file and store either the file pointer or the complete text
   * of the file (depending on the index file size) into the
   * global data structures.
   */
  if (newfile || (idxfp == NULL && idxbuffer == NULL)) {
    if (idxfp != NULL || idxbuffer != NULL)
      ident_close();

    idxsize = get_filesize(indexfile);

    /*
     * Open up the index file.
     */
    idxfp = fopen(get_truename(indexfile, NULL), "r");
    error_test(idxfp == NULL, E_OPENFAILED, return -1,
               print_error("%s:  Unable to open index file.\n", indexfile));

    /*
     * Get the first line of the file, extract the size of the header
     * lines (listing the database files) and check the line to make
     * sure it's a SEQIO Index File.
     */
    temp = fgets(line, 128, idxfp);
    error_test(temp == NULL, E_READFAILED, return -1,
               print_error("%s:  Unable to read index file.\n", indexfile));

    idxstart = myatoi(line, 10, '0');
    for (s=line; s < line + 128 && *s && (isdigit(*s) || isspace(*s)); s++) ;
    error_test(idxstart <= 0 || !*s || s >= line + 128 ||
               strncmp(s, "# SEQIO Index File", 18) != 0,
               E_PARSEERROR, return -1,
               print_error("%s:  File is not a SEQIO Index File.\n",
                           indexfile));

    /*
     * Seek back to the beginning of the file, and either read the
     * complete file (if it's small enough), or just read the header
     * lines containing the list of files.
     */
    status = fseek(idxfp, 0, SEEK_SET);
    error_test(status == -1, E_READFAILED, return -1,
               print_error("%s:  Unable to read index file.\n", indexfile));

    if (idxsize <= 50000) {
      idxbufsize = idxsize;
      idxtype = 0;
    }
    else {
      idxbufsize = idxstart;
      idxtype = 1;
    }

    idxbuffer = (char *) malloc(idxbufsize + 1);
    memory_error(idxbuffer == NULL, return -1);

    size = fread(idxbuffer, 1, idxbufsize, idxfp);
    error_test(size != idxbufsize, E_READFAILED, return -1,
               print_error("%s:  Unable to read index file.\n", indexfile));

    idxbuffer[idxbufsize] = '\0';
  }

  if (patsize == 0) {
    offset = idxstart;

    if (idxtype == 1 && fseek(idxfp, offset, SEEK_SET) == -1) {
      raise_error(E_READFAILED, return -1,
                  print_error("%s:  Read error occurred while accessing "
                              "index file.\n", indexfile));
    }
  }
  else {
    /*
     * If the current file is small enough to be stored in memory, then
     * do a binary search through the lines to find the index, then
     * scan the lines containing the identifier and construct the 
     * filename/byte-offset specifications.
     */
    if (idxtype == 0)
      offset = findline2(idxbuffer, idxstart, idxbufsize, pattern, patsize);
    else {
      offset = findline(idxfp, idxstart, idxsize, pattern, patsize);
      error_test(offset == -2, E_PARSEERROR, return -1,
                 print_error("%s:  Read/Parse error occurred while accessing "
                             "index file.\n", indexfile));
      if (offset != -1 && fseek(idxfp, offset, SEEK_SET) == -1) {
        raise_error(E_READFAILED, return -1,
                    print_error("%s:  Read error occurred while accessing "
                                "index file.\n", indexfile));
      }

    }

    if (offset == -1)
      return 0;
  }

  while (1) {
    if (idxtype == 0) {
      if (offset >= idxbufsize)
        break;

      s = idxbuffer + offset;
      fileidlen = 0;
      for (end=s; end < idxbuffer + idxbufsize && *end != '\n'; end++,offset++)
        if (!fileidlen && *end == '\t')
          fileidlen = end - s;
      offset++;
    }
    else {
      if (fgets(line, 128, idxfp) == NULL)
        break;

      s = line;
      for (end=s,fileidlen=0; end < line + 128 && *end && *end != '\n'; end++)
        if (!fileidlen && *end == '\t')
          fileidlen = end - s;
      error_test(end >= line + 128 || !*end || !fileidlen,
                 E_PARSEERROR, return -1,
                 print_error("%s:  Invalid format of index file lines.\n",
                             indexfile));
    }

    if (myncasecmp(pattern, s, patsize) < 0)
      break;

    if (!match_string(s, s + fileidlen, id, idend))
      continue;

    /*
     * Found a line matching that identifier.  Construct the
     * filename/byte-offset string and add it to the STRING structure.
     *
     * Recall, the format of the line is "ident\t#1\t#2\n", where "ident"
     * is the identifier, "#1" is the file number (i.e., the index into
     * the list of files), and "#2" is the file offset.  Also, "#1" and
     * "#2" are the encoded as base 64 numbers (the 64 ASCII characters
     * beginning with '0').
     */
    while (s < end && *s != '\t') s++;
    s++;
    error_test(s >= end || (*s - '0') < 0 || (*s - '0') >= 64,
               E_PARSEERROR, return -1,
               print_error("%s:  Invalid format of index file lines.\n",
                           indexfile));
    filenum = myatoi(s, 64, '0');

    while (s < end && *s != '\t') s++;
    s++;
    error_test(s >= end || (*s - '0') < 0 || (*s - '0') >= 64,
               E_PARSEERROR, return -1,
               print_error("%s:  Invalid format of index file lines.\n",
                           indexfile));
    fileoffset = myatoi(s, 64, '0');

    /*
     * Scan the list of files to find the filenum'th file.
     */
    end2 = idxbuffer + idxstart;
    for (s2=idxbuffer; s2 < end2 && *s2 != '\n'; s2++) ;
    s2++;
    for (count=0; count < filenum && s2 < end2; count++,s2++)
      for ( ; s2 < end2 && *s2 != '\n'; s2++)
        ;

    error_test(s2 >= end2, E_PARSEERROR, return -1,
               print_error("%s:  Invalid header format of index file.\n",
                           indexfile));

    for (t2=buffer; s2 < end2 && *s2 != '\n'; s2++,t2++)
      *t2 = *s2;
    *t2++ = '@';
    *t2++ = '#';
    t2 = myitoa(t2, fileoffset, 10, '0');
    *t2 = '\0';

    addstring(string, buffer);
  }

  return 1;
}

static void ident_close(void)
{
  if (idxfp != NULL) {
    fclose(idxfp);
    idxfp = NULL;
  }
  if (idxbuffer != NULL) {
    free(idxbuffer);
    idxbuffer = NULL;
    idxsize = idxbufsize = 0;
  }
}


static int findline(FILE *fp, int low, int high, char *pattern, int patsize)
{
  int middle, start, size, pagesize, status, offset, highflag, len;
  char *s, *t, *end, page[8196];

  start = low;
  size = high;
  highflag = 0;
  while (high - low > 8192) {
    middle = (high + low) / 2 - 4096;
    if (fseek(fp, middle, SEEK_SET) == -1)
      return -2;
    pagesize = fread(page, 1, 8192, fp);
    if (pagesize < 160)
      return -2;

    /*
     * If the search might read a page that begins and ends in the
     * middle of lines (when flag is 1), then adjust the beginning
     * and end to skip past the line fragment that may appear on
     * the edges of the read page.
     */
    for (s=page; s < page + pagesize && *s != '\n'; s++,middle++) ;
    s++;  middle++;
    for (end=page+pagesize; end > s && *(end-1) != '\n'; end--) ;

    if (s >= end)
      return -2;

    /*
     * Check to see if the pattern is smaller than (or equal to)
     * the first line.
     */
    for (t=s; t < end && *t != '\n'; t++) ;
    len = patsize;
    if (t - s < patsize)
      len = t - s;

    if ((status = myncasecmp(pattern, s, len)) <= 0) {
      high = middle;
      highflag = (status == 0);
      continue;
    }

    /*
     * Check to see if the pattern is larger than the first line.
     */
    for (t=end-1; t > s && *(t-1) != '\n'; t--) ;
    len = patsize;
    if (end - t < patsize)
      len = end - t;

    if (myncasecmp(pattern, t, len) > 0) {
      low = middle + (end - s);
      continue;
    }

    /*
     * If neither is the case, then the first line containing the pattern
     * must occur in the current page, so do a binary search on those
     * lines.
     */
    offset = findline2(page, s - page, end - page, pattern, patsize);
    if (offset == -1)
      return (highflag ? high : -1);
    else
      return middle + (offset - (s - page));
  }

  /*
   * If we've gotten down to where we can read the whole page into
   * memory, just do a binary search on that page.
   */
  if (fseek(fp, low, SEEK_SET) == -1)
    return -2;
  pagesize = fread(page, 1, high - low, fp);
  if (pagesize != high - low)
    return -2;

  offset = findline2(page, 0, high - low, pattern, patsize);
  if (offset == -1)
    return (highflag ? high : -1);
  else
    return low + offset;
}

static int findline2(char *text, int low, int high, char *pattern, int patsize)
{
  int middle, size, len, status;
  char *s, *end;

  /*
   * For this search to work correctly, the string starting with text[low]
   * must be the beginning of a line, and text[high] must be one larger then
   * the end of a line (i.e., either it must be one larger than the last 
   * newline, or for strings that don't contain that trailing newline, it
   * must be one larger than the last character on the last line).
   *
   * The search can handle text where the trailing newline on the last
   * line of the text is missing.
   */
  size = high;
  while (low < high) {
    middle = (high + low) / 2;
    for (s=end=text+middle; s > text + low && *(s-1) != '\n'; s--) ;
    for ( ; end < text + high && *end != '\n'; end++) ;

    len = patsize;
    if (end - s < len)
      len = end - s;

    status = myncasecmp(pattern, s, len);
    if (status <= 0)
      high = (s - text);
    else if (status > 0)
      low = (end - text) + 1;
  }

  /*
   * At the end of this, low == high and either they both equal
   * the search size (in which case all the values in the file are
   * smaller than the pattern), or the contents of the next line
   * is where the lines containing the pattern should begin.
   */
  if (high >= size)
    return -1;

  if (myncasecmp(pattern, text + high, patsize) == 0)
    return high;
  else
    return -1;
}



/*
 *
 * Interfaces to the file i/o and error reporting procedures:
 *
 *     open_raw_file, open_raw_stdin, is_raw_stdin,
 *     close_raw_file, isa_file, isa_dir, match_file
 *
 * These files encapsulate all of the file I/O, so for portability only
 * these procedures should be different on different platforms.
 *
 *
 */
static int open_raw_file(char *filename, FILEPTR *ptr_out)
{
  if ((*ptr_out = open(filename, O_RDONLY)) >= 0)
    return STATUS_OK;
  else
    return STATUS_ERROR;
}

static int read_raw_file(FILEPTR ptr, char *buffer, int size)
{
  return read(ptr, buffer, size);
}

static int seek_raw_file(FILEPTR ptr, int bytepos)
{
  if (lseek(ptr, bytepos, SEEK_SET) >= 0)
    return STATUS_OK;
  else
    return STATUS_ERROR;
}

static int close_raw_file(FILEPTR ptr)
{
  if (close(ptr) >= 0)
    return STATUS_OK;
  else
    return STATUS_ERROR;
}

static int open_raw_stdin(FILEPTR *ptr_out)
{
  *ptr_out = 0;
  return STATUS_OK;
}

static int open_stdout(FILE **ptr_out)
{
  *ptr_out = stdout;
  return STATUS_OK;
}

static void puterror(char *s)
{
  fputs(s, stderr);
}

static char *read_small_file(char *filename)
{
  int fd;
  char *buffer;
  struct stat sbuf;
  
  if (stat(filename, &sbuf) < 0 || (sbuf.st_mode & S_IFMT) != S_IFREG)
    return NULL;

  fd = -1;
  buffer = NULL;
  if ((buffer = (char *) malloc(sbuf.st_size + 1)) == NULL ||
      (fd = open(filename, O_RDONLY)) < 0 ||
      read(fd, buffer, sbuf.st_size) != sbuf.st_size) {
    if (buffer != NULL)  free(buffer);
    if (fd >= 0)  close(fd);
    return NULL;
  }
  close(fd);
  buffer[sbuf.st_size] = '\0';

  return buffer;
}
  
static int open_directory(char *dirname, DIRPTR *dp_out)
{
#ifdef WIN32

  char curdir[FILENAME_MAX];
  DIRPTR dp;

  if ((dp = (DIRPTR) malloc(sizeof(DIRSTRUCT))) == NULL)
    return STATUS_ERROR;

  if (GetCurrentDirectory(FILENAME_MAX, curdir) == 0) {
    free(dp);
    return STATUS_ERROR;
  }
  if (!SetCurrentDirectory(dirname)) {
    free(dp);
    return STATUS_ERROR;
  }

  dp->init_flag = 1;
  dp->handle = FindFirstFile("*.*", &dp->dirinfo);
  if (dp->handle == NULL) {
    free(dp);
    return STATUS_ERROR;
  }

  if (!SetCurrentDirectory(curdir)) {
    CloseHandle(dp->handle);
    free(dp);
    return STATUS_ERROR;
  }

  *dp_out = dp;
  return STATUS_OK;

#else

  if ((*dp_out = opendir(dirname)) != NULL)
    return STATUS_OK;
  else
    return STATUS_ERROR;

#endif
}

static char *read_dirname(DIRPTR dp)
{
#ifdef WIN32

  if (dp->init_flag) {
    dp->init_flag = 0;
    return dp->dirinfo.cFileName;
  }

  if (FindNextFile(dp->handle, &dp->dirinfo))
    return dp->dirinfo.cFileName;
  else
    return NULL;

#else

  struct dirent *dent;

  dent = readdir(dp);
  if (dent == NULL)
    return NULL;
  else
    return dent->d_name;

#endif
}

static void close_directory (DIRPTR dp)
{
#ifdef WIN32

  CloseHandle(dp->handle);
  free(dp);

#else

  closedir(dp);

#endif
}

static int isa_file(char *filename)
{
  struct stat sbuf;

  if (stat(filename, &sbuf) >= 0 && (sbuf.st_mode & S_IFMT) == S_IFREG)
    return 1;
  else
    return 0;
}


/*
 * Not currently used, but I'm leaving it here just in case.
 *
static int isa_dir(char *directory)
{
  struct stat sbuf;
  
  return (stat(directory, &sbuf) >= 0 && (sbuf.st_mode & S_IFMT) == S_IFDIR);
}
 *
 *
 */


static int get_filesize(char *filename)
{
  struct stat sbuf;
  
  if (stat(filename, &sbuf) >= 0 && (sbuf.st_mode & S_IFMT) == S_IFREG)
    return sbuf.st_size;
  else
    return 0;
}


static char *get_truename(char *filename, char *fileend)
{
  static char buf[FILENAME_MAX+1];
  int len;
  char ch, *s, *t, *s2;

  s = filename;
  t = buf;
  if (*s == '~' && (s2 = getenv("HOME")) != NULL) {
    while ((*t++ = *s2++)) ;
    t--;

    s++;
    ch = (fileend != NULL && s == fileend ? '\0' : *s);

    if (ch == dirch) {
      if (*(t-1) == dirch)
        t--;
    }
    else if (isalpha(ch))
      while (t > buf && *(t-1) != dirch) t--;
    else {
      t = buf;
      s = filename;
    }
  }

  len = (fileend != NULL ? fileend - s : strlen(s));
  if (len > FILENAME_MAX - (t - buf))
    len = FILENAME_MAX - (t - buf);
  memcpy(t, s, len);
  t[len] = '\0';

  return buf;
}

static int is_absolute(char *path)
{
  int abspath;

  if (path[0] != '~') {
#ifdef WIN32
    abspath = (path[0] == dirch ||
               (isalpha(path[0]) && path[1] == ':' && path[2] == dirch));
#else
    abspath = (path[0] == dirch);
#endif

    return abspath;
  }

  return 1;
}

static char *get_today()
{
  static char buffer[32];
  static int flag = 0;
  time_t now;
  char *s;

  if (!flag) {
    now = time(NULL);
    s = ctime(&now);
    buffer[0] = (isspace(s[8]) ? '0' : s[8]);
    buffer[1] = s[9];
    buffer[2] = '-';
    buffer[3] = s[4];
    buffer[4] = s[5];
    buffer[5] = s[6];
    buffer[6] = '-';
    buffer[7] = s[20];
    buffer[8] = s[21];
    buffer[9] = s[22];
    buffer[10] = s[23];
    buffer[11] = '\0';
    buffer[12] = s[11];
    buffer[13] = s[12];
    buffer[14] = s[13];
    buffer[15] = s[14];
    buffer[16] = s[15];
    buffer[17] = '\0';
  }
 
  return buffer;
}
  
