# ifndef MS_H

# define MS_H

# define MAXIND 10000
# define MAXBREAK 50000

//ReExact add
struct fnode {
    int s1;
    int s2;
    int samp;
};

struct ind_count {
    int samp;
    int count;
    long breaks[MAXBREAK];
};
//~

struct devent {
	double time;
	int popi;
	int popj;
	double paramv;
	double **mat ;
	char detype ;
	struct devent *nextde;
	} ;
struct c_params {
	int npop;
	int nsam;
	int *config;
	double **mig_mat;
	double r;
	int nsites;
	double f;
	double track_len;
	double *size;
	double *alphag;
	struct devent *deventlist ;
	} ;
struct m_params {
	 double theta;
	int segsitesin;
	int treeflag;
	int timeflag;
	int mfreq;
	 } ;
struct params { 
	struct c_params cp;
	struct m_params mp;
	int commandlineseedflag ;
	int output_precision;
	};

struct node{
	int abv;
	int ndes;
	float time;
};


/*KRT -- prototypes added*/
void ordran(int n, double pbuf[]);
void ranvec(int n, double pbuf[]);
void order(int n, double pbuf[]);

void biggerlist(int nsam,  char **list );
int poisso(double u);
void locate(int n,double beg, double len,double *ptr);
void mnmial(int n, int nclass, double p[], int rv[]);
void usage();
int tdesn(struct node *ptree, int tip, int node );
int pick2(int n, int *i, int *j);
int xover(int nsam,int ic, int is);
int links(int c);

# endif
