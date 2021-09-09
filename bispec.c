#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include"fftw3.h"
#include<omp.h>
#include <pthread.h>

#define NUM_THREADS 4

/*------------------------------GLOBAL VARIABLES-------------------------------*/

float  pi=M_PI;

int N1, N2, N3; // box dimension (grid) 
float LL; // grid spacing in Mpc
float norml, Lcube, vol, tpibyL; 

float ***ro; // for density
fftwf_complex *comp_ro;

fftwf_plan p_ro; // for FFT

//-----------------------------------------------------------------------------//

int Nk1bin, Nnbin, Ncostbin;
float fac1, fac2, fac3;

float scalek1, scalen, scalecost, k2byk1;
float k1max, k1min;
int k2flag[8][3];

double ***bispec, ***bispec1, ***bispec2, ***bispec3, ***bispec4;
long ***no, ***no1, ***no2, ***no3, ***no4;

//-----------------------------------------------------------------------------//

float  ***allocate_fftwf_3d(long N1,long N2,long N3);

double  ***allocate_double_3d(int N1, int N2, int N3);

long  ***allocate_long_3d(int N1, int N2, int N3);

void bispec_cal(int xstart, int xend, int xinc, int ystart, int yend, int yinc, int zstart, int zend, int zinc, int skip1, int skip2);

void * bispec_cal_c1();
void * bispec_cal_c2();
void * bispec_cal_c3();
void * bispec_cal_c4();


//-----------------------------------------------------------------------------//


void main()
{
  pthread_t threads[NUM_THREADS];
  
  int i, j, k, ii, jj, kk;
  double mean=0.0;
  float z, k1, k2, cost;
  
  double t,T=omp_get_wtime();  // for timing
  
  char file[500],file1[500];
  FILE  *inp,*outpp; 
  
  /*-------------------------------------------------------*/
  
  inp=fopen("input.bispec","r");
  fscanf(inp,"%f",&LL);
  fscanf(inp, "%d", &Nk1bin);
  fscanf(inp, "%d", &Nnbin);  
  fscanf(inp, "%d", &Ncostbin); 
  
  fclose(inp);
  
  /*- allocate memory for bispectrum, bins  ---------------*/
  
  bispec = allocate_double_3d(Nnbin, Ncostbin, Nk1bin); 
  bispec1 = allocate_double_3d(Nnbin, Ncostbin, Nk1bin); 
  bispec2 = allocate_double_3d(Nnbin, Ncostbin, Nk1bin); 
  bispec3 = allocate_double_3d(Nnbin, Ncostbin, Nk1bin); 
  bispec4 = allocate_double_3d(Nnbin, Ncostbin, Nk1bin); 
  
  no = allocate_long_3d(Nnbin, Ncostbin, Nk1bin); 
  no1 = allocate_long_3d(Nnbin, Ncostbin, Nk1bin); 
  no2 = allocate_long_3d(Nnbin, Ncostbin, Nk1bin); 
  no3 = allocate_long_3d(Nnbin, Ncostbin, Nk1bin); 
  no4 = allocate_long_3d(Nnbin, Ncostbin, Nk1bin); 
  
  for(ii=0;ii<Nnbin;ii++)
    for(jj=0;jj<Ncostbin;jj++)
      for(kk=0;kk<Nk1bin;kk++)
        {
	  bispec[ii][jj][kk] = 0.0;
	  bispec1[ii][jj][kk] = 0.0;
	  bispec2[ii][jj][kk] = 0.0;
	  bispec3[ii][jj][kk] = 0.0;
	  bispec4[ii][jj][kk] = 0.0;
	  
	  no[ii][jj][kk] = 0;
	  no1[ii][jj][kk] = 0;
	  no2[ii][jj][kk] = 0;
	  no3[ii][jj][kk] = 0;
	  no4[ii][jj][kk] = 0;
	}
  
  /*-----Reading the simulation data file------------------*/
  
  strcpy(file, "c_data8.0_100");
  
  printf("Reading dT file %s\n",file);
  
  inp = fopen(file, "r");
  fread(&N1,sizeof(int),1,inp);
  fread(&N2,sizeof(int),1,inp);
  fread(&N3,sizeof(int),1,inp);
  printf("N1=%d, N2=%d, N3=%d\n",N1,N2,N3);
  
  /* allocate memory for HI map  ---------------*/
  
  ro=allocate_fftwf_3d(N1,N2,N3+2);
  
  for(ii=0;ii<N1;ii++)
    for(jj=0;jj<N2;jj++)
      for(kk=0;kk<N3;kk++)
        {
	  fread(&ro[ii][jj][kk],sizeof(float),1,inp);
	  mean += ro[ii][jj][kk];
	}
  fclose(inp);
  
  mean /= (1.*N1*N2*N3);
  printf("mean=%f mK\n",mean);  
  
  /***Normalization factors for converting the box into an unit box****/
  
  norml=1./((1.0*N1)*(1.0*N2)*(1.0*N3));
  Lcube=powf(LL,3.);
  vol=Lcube/norml;
  printf("volume = [%.4f Mpc]^3 = %e Mpc^3\n", LL*N1, vol);
  tpibyL=2.0*pi/LL; // 2 pi /LL 
  
  /*************** TAKING FOURIER TRANSFORM OF RO. **************/
  
  /* Creating the plan for forward FFT */
  
  p_ro = fftwf_plan_dft_r2c_3d(N1, N2, N3, &(ro[0][0][0]), (fftwf_complex*)&(ro[0][0][0]), FFTW_ESTIMATE);  
  
  for(i=0;i<N1;i++)
    for(j=0;j<N2;j++)
      for(k=0;k<N3;k++)
	ro[i][j][k]=ro[i][j][k]*Lcube;
  
  fftwf_execute(p_ro);  // ro mow contains delta(k)
    
  //Typecasting ro as a complex variable
  comp_ro = (fftwf_complex *)&(ro[0][0][0]);  
  
  /*-------------------------------------------- NOW START ESTIMATING THE Bispectrum -----------------------------------------------------*/
  
  fac1=1./(1.*N1*N1);
  fac2=1./(1.*N2*N2);
  fac3=1./(1.*N3*N3);
  
  k1max = pi/LL;
  k1min = 2.*pi/(N3*LL);
  //k1min = 2.*pi/(260.*LL);

  scalek1 = (log10(k1max)-log10(k1min))/Nk1bin;
  scalen = (1.0001 - 0.5)/Nnbin;
  scalecost = (1.0001 - 0.5)/Ncostbin;
  
  printf("Delta n=%.4f, Delta cost=%.4f, (Delta k1)/k1=%.4f\n", scalen, scalecost, scalek1);

  //-----------------------------------------------------

  int tmp, jk, ll;

  for(jk=0; jk<8; jk++)
    {
      k2flag[jk][2]=1; k2flag[jk][1]=1; k2flag[jk][0]=1;
      
      tmp=jk;
      for(ll=0; tmp>0; ll++)
	{
	  k2flag[jk][ll] = 1 - 2*(tmp%2);
	  tmp=tmp/2;
	}
      //printf("%d %d %d %d\n", jk, k2flag[jk][2], k2flag[jk][1], k2flag[jk][0]);
    }

  /*-------------------------------------------- cubes -----------------------------------------------------*/
  /*                        creating threads which executes the 1/4 cubes                                   */
  
  if(pthread_create(&threads[0], NULL, bispec_cal_c1, NULL)) fprintf(stderr, "Error creating thread\n");
  if(pthread_create(&threads[1], NULL, bispec_cal_c2, NULL)) fprintf(stderr, "Error creating thread\n");
  if(pthread_create(&threads[2], NULL, bispec_cal_c3, NULL)) fprintf(stderr, "Error creating thread\n");
  if(pthread_create(&threads[3], NULL, bispec_cal_c4, NULL)) fprintf(stderr, "Error creating thread\n");
  
  /*-------------------------------------------- planes -----------------------------------------------------*/
  
  bispec_cal(1, N1/2, 1, 1, N2/2, 1, 0, N3/2+1, N3/2, 0, 1); // xy-planes (skip for 0, 1)
  bispec_cal(N1/2+1, N1, 1, 1, N2/2, 1, 0, N3/2+1, N3/2, 4, 5); // xy-planes (skip for 4, 5)

  bispec_cal(0, N1/2+1, N1/2, 1, N2/2, 1, 1, N3/2, 1, 0, 4); // yz-axis (skip for 0, 4)
  bispec_cal(0, N1/2+1, N1/2, N2/2+1, N2, 1, 1, N3/2, 1, 2, 6); // yz-axis (skip for 2, 6)
  
  bispec_cal(1, N1/2, 1, 0, N2/2+1, N2/2, 1, N3/2, 1, 0, 2); // xz-axis (skip for 0, 2)
  bispec_cal(N1/2+1, N1, 1, 0, N2/2+1, N2/2, 1, N3/2, 1, 4, 6); // xz-axis (skip for 4, 6)
  
  printf("planes are done\n");  

  /*-------------------------------------------- lines -----------------------------------------------------*/
  
  bispec_cal(1, N1/2, 1, 0, N2/2+1, N2/2, 0, N3/2+1, N3/2, 0, 2); // x-axis (skip for 0, 1, 2, 3)
  bispec_cal(0, N1/2+1, N1/2, 1, N2/2, 1, 0, N3/2+1, N3/2, 0, 4); // y-axis (skip for 0, 1, 4, 5)
  bispec_cal(0, N1/2+1, N1/2, 0, N2/2+1, N2/2, 1, N3/2, 1, 0, 2); // z-axis (skip for 0, 2, 4, 6)
  
  printf("lines are done\n");
  
  /*-------------------------------------------- Points  -----------------------------------------------------*/
  
  /* bispec_cal(0, N1/2+1, N1/2, 0, N2/2+1, N2/2, 0, N3/2+1, N3/2, 9, 9); // do not contribute */
  
  /*------------------------------ wait for the threads to finish ------------------------------------------*/
  
  for(i=0; i<NUM_THREADS; i++)
    if(pthread_join(threads[i], NULL)) fprintf(stderr, "Error joining thread\n");
  
  /*-------------------------------------------- The end -----------------------------------------------------*/
  
  for(ii=0;ii<Nnbin;ii++)
    for(jj=0;jj<Ncostbin;jj++)
      for(kk=0;kk<Nk1bin;kk++)
        {
	  bispec[ii][jj][kk] += bispec1[ii][jj][kk] + bispec2[ii][jj][kk] + bispec3[ii][jj][kk] + bispec4[ii][jj][kk];
	  
	  no[ii][jj][kk] += no1[ii][jj][kk] + no2[ii][jj][kk] + no3[ii][jj][kk] + no4[ii][jj][kk];
	}
  
  //-----------------------------------------------------------------------------//
  
  int print_flag;
  
  for(ii=0;ii<Nnbin;ii++)
    {
      k2byk1 = 0.5 + (2.*ii + 1)*0.5*scalen;
      
      sprintf(file, "mkdir k2byk1_%1.3f", k2byk1);
      system(file);
      
      for(jj=0;jj<Ncostbin;jj++)
	{
	  cost = 0.5 + (2.*jj + 1)*0.5*scalecost;
	  
	  print_flag = 0;
	  for(kk=0;kk<Nk1bin;kk++)
	    {
	      if(no[ii][jj][kk]>0)
		print_flag = 1;
	    }
	  
	  if(print_flag > 0)
	    {
	      sprintf(file1,"%s%1.3f%s%1.3f","k2byk1_",k2byk1,"/bispec_cosalpha",-cost);
	      outpp=fopen(file1,"w");
	      //printf("Writting bispectrum file %s\n",file1);
	      
	      
	      for(kk=0;kk<Nk1bin;kk++)
		{
		  k1 = k1min*pow(10.,(2.*kk + 1)*0.5*scalek1);
		  //printf("%e\t%e\t%e\t%e\t%ld\n", k2byk1, cost, k1, bispec[ii][jj][kk], no[ii][jj][kk]);
		  
		  if(no[ii][jj][kk]>0)
		    {
		      bispec[ii][jj][kk] = bispec[ii][jj][kk]/(1.*no[ii][jj][kk]*vol);
		      //printf("%e\t%e\t%e\t%e\n", k2byk1, cost, k1, bispec[ii][jj][kk]);
		      
		      fprintf(outpp,"%e\t%e\t%ld\n", k1, bispec[ii][jj][kk], no[ii][jj][kk]);
		    }
		}
	      fclose(outpp);
	    }
	}
    }//printing of bispectrum done
  
  printf("done. Total time taken = %dhr %dmin %dsec\n",(int)((omp_get_wtime()-T)/3600), (int)((omp_get_wtime()-T)/60)%60, (int)(omp_get_wtime()-T)%60);  
  
}


//-----------------------------------------------------------------------------//


float  ***allocate_fftwf_3d(long N1,long N2,long N3)
{
  long ii,jj;
  long asize,index;
  float ***phia, *phi;
  
  phia=(float ***) fftwf_malloc (N1 *  sizeof(float **));
  
  for(ii=0;ii<N1;++ii)
    phia[ii]=(float **) fftwf_malloc (N2 *  sizeof(float *));
  
  asize = N1*N2;
  asize = asize*N3;
  
  if(!(phi = (float *) calloc(asize,sizeof(float))))
    {
      printf("error in allocate_fftwf_3d");
      exit(0);
    }
  
  for(ii=0;ii<N1;++ii)
    for(jj=0;jj<N2;++jj)
      {
	index = N2*N3;
	index = index*ii + N3*jj;
	phia[ii][jj]=phi+ index;
      }
  return(phia);
}


//-----------------------------------------------------------------------------//


double  ***allocate_double_3d(int N1, int N2, int N3)
{
  int ii,jj;
  double ***phia, *phi;
  
  phia = (double ***) malloc(N1*sizeof(double**));
  
  for(ii=0;ii<N1;ii++)
    phia[ii] = (double **) malloc(N2*sizeof(double*));
  
  phi = (double *) calloc((N1*N2*N3),sizeof(double));
  
  for(ii=0;ii<N1;++ii)
    for(jj=0;jj<N2;++jj)
      phia[ii][jj] = phi + N2*N3*ii + N3*jj;
  
  return(phia);
}


//-----------------------------------------------------------------------------//


long  ***allocate_long_3d(int N1, int N2, int N3)
{
  int ii,jj;
  long ***phia, *phi;
  
  phia = (long ***) malloc(N1*sizeof(long**));
  
  for(ii=0;ii<N1;ii++)
    phia[ii] = (long **) malloc(N2*sizeof(long*));
  
  phi = (long *) calloc((N1*N2*N3),sizeof(long));
  
  for(ii=0;ii<N1;++ii)
    for(jj=0;jj<N2;++jj)
      phia[ii][jj] = phi + N2*N3*ii + N3*jj;
  
  return(phia);
}


//-----------------------------------------------------------------------------//


void bispec_cal(int xstart, int xend, int xinc, int ystart, int yend, int yinc, int zstart, int zend, int zinc, int skip1, int skip2)
{
  int i, j, k, ii, jj, kk;
  int ii2, jj2, iii, jjj, kkk, ll, jk;
  int a1, b1, a2, b2, c2;
  
  long index1, index2, index3;
  int d1, d2, d3;
  int k2max, k2b2max, k2c2min, k2c2max;
  int delta2flag, delta3flag;
  float k2a2, k2b2, costa2, costb2, k1, k2, cost;
  
  //-----------------------------------------------------
  
  for(i=xstart;i<xend;i=i+xinc)
    {
      a1 = (i>N1/2)? i-N1: i;
      for(j=ystart;j<yend;j=j+yinc)
	{
	  b1 = (j>N2/2)? j-N2: j;
	  for(k=zstart;k<zend;k=k+zinc)
	    {
	      index1 = i*N2*(N3/2+1) + j*(N3/2+1) + k;
	      
	      k1 = sqrtf(fac1*a1*a1 + fac2*b1*b1 + fac3*k*k); // is |k1|/tpibyL
	      
	      d1 = floorf((log10(tpibyL*k1)-log10(k1min))/scalek1); //logarithmic bins
	      //if(d1<0 || d1>=Nk1bin){printf("%d %d %d %d\n",d1, a1, b1, k);}
	      
	      if(d1>=0 && d1<Nk1bin)
		{
		  k2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k)));
		  
		  for(jk=0; jk<8; jk++)
		    {
		      jk = (jk==skip1)? jk+1 : jk;
		      jk = (jk==skip2)? jk+1 : jk;
		      
		      for(ii=0; ii<=k2max; ii++)
			{
			  a2 = k2flag[jk][2]*ii;
			  k2a2 = fac1*a2*a2;
			  costa2 = - fac1*a1*a2/k1;
			  
			  k2b2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii)));
			  
			  for(jj=0; jj<=k2b2max; jj++)
			    {
			      b2 = k2flag[jk][1]*jj;
			      k2b2 = k2a2 + fac2*b2*b2; 
			      costb2 = costa2 - fac2*b1*b2/k1;
			      
			      k2c2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii - jj*jj)));
			      k2c2min = (0.25*(a1*a1 + b1*b1 + k*k) <= 1.*(ii*ii + jj*jj))? 0 : ceilf(sqrtf(0.25*(a1*a1 + b1*b1 + k*k) - 1.*(ii*ii + jj*jj)));
			      
			      for(kk=k2c2min; kk<=k2c2max; kk++)
				{
				  c2 = k2flag[jk][0]*kk;
				  
				  cost = costb2 - fac3*k*c2/k1; // is cost*k2
				  
				  if(cost/k1 >= 0.5)
				    {
				      k2 = sqrtf(k2b2 + fac3*c2*c2); // is |k2|/tpibyL
				      
				      d2 = floorf((k2/k1 - 0.5)/scalen); //if(d2<0 || d2>=Nnbin){printf("Not ok at d2=%d\n", d2);}

				      d3 = (cost/k2 <= 0.5)? 0 : floorf((cost/k2 - 0.5)/scalecost); //if(d3<0 || d3>=Ncostbin){printf("Not ok at d3=%d\n", d3);}
				      
				      //-----------------------------------------------------
				      
				      if(c2 < 0)
					{
					  ii2 = (a2>0)? N1-a2 : -a2;
					  jj2 = (b2>0)? N2-b2 : -b2;
					  
					  delta2flag = -1; 
					}
				      else
					{
					  ii2 = (a2<0)? N1+a2 : a2; 
					  jj2 = (b2<0)? N2+b2 : b2;
					  
					  delta2flag = 1;
					}
				      
				      //-----------------------------------------------------
				      
				      if(-(k  + c2) < 0)
					{
					  iii = (-(a1 + a2)>0)? N1+(a1 + a2) : (a1 + a2);
					  jjj = (-(b1 + b2)>0)? N2+(b1 + b2) : (b1 + b2);
					  
					  delta3flag = -1; 
					}
				      else
					{
					  iii = (-(a1 + a2)<0)? N1-(a1 + a2) : -(a1 + a2); 
					  jjj = (-(b1 + b2)<0)? N2-(b1 + b2) : -(b1 + b2);
					  
					  delta3flag = 1;
					}
				      
				      //-----------------------------------------------------
				      
				      index2 = ii2*N2*(N3/2+1) + jj2*(N3/2+1) + kk;
				      index3 = iii*N2*(N3/2+1) + jjj*(N3/2+1) + abs(k + c2);
				      
				      bispec[d2][d3][d1] += comp_ro[index1][0]*comp_ro[index2][0]*comp_ro[index3][0] - comp_ro[index1][1]*delta2flag*comp_ro[index2][1]*comp_ro[index3][0] - comp_ro[index1][0]*delta2flag*comp_ro[index2][1]*delta3flag*comp_ro[index3][1] - comp_ro[index1][1]*comp_ro[index2][0]*delta3flag*comp_ro[index3][1];
				      no[d2][d3][d1] += 1;
				    
				    } // end of d3 check 
				  
				} // end of k2 z-direction
			      
			    } // end of k2 y-direction
			  
			} // end of k2 x-direction
		      
		    } // end of k2 
		  
		} // end of d1 check 
	      
	    } // end of k1 z-direction
	  
	} // end of k1 y-direction
      
    } // end of k1 x-direction
  
}


//-----------------------------------------------------------------------------//


void * bispec_cal_c1()
{
  int i, j, k, ii, jj, kk;
  int ii2, jj2, iii, jjj, kkk, ll, jk;
  int a1, b1, a2, b2, c2;
  
  long index1, index2, index3;
  int d1, d2, d3;
  int k2max, k2b2max, k2c2min, k2c2max;
  int delta2flag, delta3flag;
  float k2a2, k2b2, costa2, costb2, k1, k2, cost;
  
  int mm;
  int i2, j2;
  int skip1[4];
  int k1a1max, k1b1max, k1c1max;
  
  //-----------------------------------------------------

  k1a1max = floorf(N1*k1max/tpibyL);
  if(k1a1max > N1/2)
    {
      printf("Check k1max k1a1max=%d\n", k1a1max);
      exit(0);
    }
  skip1[0] = 0;
  
  //-----------------------------------------------------

  for(i=1;i<k1a1max;i++)
    {
      a1 = i;
      i2 = (a1<0)? N1+a1 : a1; 
      k1b1max = floorf(sqrtf(1.*(k1a1max*k1a1max - i*i)));
      
      for(j=1;j<=k1b1max;j++)
	{
	  b1 = j;
	  j2 = (b1<0)? N2+b1 : b1; 
	  k1c1max = floorf(sqrtf(1.*(k1a1max*k1a1max - i*i - j*j)));
	  
	  for(k=1;k<=k1c1max;k++)
	    {
	      index1 = i2*N2*(N3/2+1) + j2*(N3/2+1) + k;
	      
	      k1 = sqrtf(fac1*a1*a1 + fac2*b1*b1 + fac3*k*k); // is |k1|/tpibyL
	      
	      d1 = floorf((log10(tpibyL*k1) - log10(k1min))/scalek1); //logarithmic bins
	      //if(d1<0 || d1>=Nk1bin){printf("%d %d %d %d\n",d1, a1, b1, k);}

	      if(d1>=0 && d1<Nk1bin)
		{
		  //-----------------------------------------------------
		  
		  skip1[1] = (i/(k1*N1) < 0.5)? 4 : 9;
		  skip1[2] = (j/(k1*N2) < 0.5)? 2 : 9;
		  skip1[3] = (k/(k1*N3) < 0.5)? 1 : 9;
		  //printf("%d %d %d %d\n", skip1[0], skip1[1], skip1[2], skip1[3]);
		  
		  //-----------------------------------------------------
		  
		  k2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k)));
		  
		  for(jk=0; jk<8; jk++)
		    {
		      if(jk == skip1[0]){continue;}
		      if(jk == skip1[1]){continue;}
		      if(jk == skip1[2]){continue;}
		      if(jk == skip1[3]){continue;}
		      
		      delta2flag = (k2flag[jk][0] < 0)?  -1 :  1;
		      
		      for(ii=0; ii<=k2max; ii++)
			{
			  a2 = k2flag[jk][2]*ii;
			  k2a2 = fac1*a2*a2;
			  costa2 = - fac1*a1*a2/k1;
			  
			  if(k2flag[jk][0] < 0) ii2 = (a2>0)? N1-a2 : -a2; 
			  else ii2 = (a2<0)? N1+a2 : a2;
			  
			  k2b2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii)));
			  
			  for(jj=0; jj<=k2b2max; jj++)
			    {
			      b2 = k2flag[jk][1]*jj;
			      k2b2 = k2a2 + fac2*b2*b2; 
			      costb2 = costa2 - fac2*b1*b2/k1;
			      
			      if(k2flag[jk][0] < 0) jj2 = (b2>0)? N2-b2 : -b2;
			      else jj2 = (b2<0)? N2+b2 : b2;
			      
			      k2c2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii - jj*jj)));
			      k2c2min = (0.25*(a1*a1 + b1*b1 + k*k) <= 1.*(ii*ii + jj*jj))? 0 : ceilf(sqrtf(0.25*(a1*a1 + b1*b1 + k*k) - 1.*(ii*ii + jj*jj)));
			      
			      for(kk=k2c2min; kk<=k2c2max; kk++)
				{
				  c2 = k2flag[jk][0]*kk;
				  
				  cost = costb2 - fac3*k*c2/k1; // is cost*k2
				  
				  if(cost/k1 >= 0.5)
				    {
				      k2 = sqrtf(k2b2 + fac3*c2*c2); // is |k2|/tpibyL
				      
				      d2 = floorf((k2/k1 - 0.5)/scalen); //if(d2<0 || d2>=Nnbin){printf("Not ok at d2=%d\n", d2);}
				      
				      d3 = (cost/k2 <= 0.5)? 0 : floorf((cost/k2 - 0.5)/scalecost); //if(d3<0 || d3>=Ncostbin){printf("Not ok at d3=%d\n", d3);}
				      
				      //-----------------------------------------------------
				      
				      if(c2 == 0)
					{
					  ii2 = (a2<0)? N1+a2 : a2; 
					  jj2 = (b2<0)? N2+b2 : b2;
					  
					  delta2flag = 1;
					}
				      
				      //-----------------------------------------------------
				      
				      if(-(k  + c2) < 0)
					{
					  iii = (-(a1 + a2)>0)? N1+(a1 + a2) : (a1 + a2);
					  jjj = (-(b1 + b2)>0)? N2+(b1 + b2) : (b1 + b2);
					  
					  delta3flag = -1; 
					}
				      else
					{
					  iii = (-(a1 + a2)<0)? N1-(a1 + a2) : -(a1 + a2); 
					  jjj = (-(b1 + b2)<0)? N2-(b1 + b2) : -(b1 + b2);
					  
					  delta3flag = 1;
					}
				      
				      //-----------------------------------------------------
				      
				      index2 = ii2*N2*(N3/2+1) + jj2*(N3/2+1) + kk;
				      index3 = iii*N2*(N3/2+1) + jjj*(N3/2+1) + abs(k + c2);
				      
				      bispec1[d2][d3][d1] += comp_ro[index1][0]*comp_ro[index2][0]*comp_ro[index3][0] - comp_ro[index1][1]*delta2flag*comp_ro[index2][1]*comp_ro[index3][0] - comp_ro[index1][0]*delta2flag*comp_ro[index2][1]*delta3flag*comp_ro[index3][1] - comp_ro[index1][1]*comp_ro[index2][0]*delta3flag*comp_ro[index3][1];
				      
				      no1[d2][d3][d1] += 1;
				      
				    } // end of d3 check 
				  
				} // end of k2 z-direction
			      
			    } // end of k2 y-direction
			  
			} // end of k2 x-direction
		      
		    } // end of k2
		  
		} // end of d1 check
	      
	    } // end of k1 z-direction
	  
	} // end of k1 y-direction
      
    } // end of k1 x-direction
  
  printf("cube %d/4 is done\n",1);  
}

//-----------------------------------------------------------------------------//


void * bispec_cal_c2()
{
  int i, j, k, ii, jj, kk;
  int ii2, jj2, iii, jjj, kkk, ll, jk;
  int a1, b1, a2, b2, c2;
  
  long index1, index2, index3;
  int d1, d2, d3;
  int k2max, k2b2max, k2c2min, k2c2max;
  int delta2flag, delta3flag;
  float k2a2, k2b2, costa2, costb2, k1, k2, cost;
  
  int mm;
  int i2, j2;
  int skip1[4];
  int k1a1max, k1b1max, k1c1max;
  
  //-----------------------------------------------------
  
  k1a1max = floorf(N1*k1max/tpibyL);
  skip1[0] = 2;
  
  for(i=1;i<k1a1max;i++)
    {
      a1 = i;
      i2 = (a1<0)? N1+a1 : a1; 
      k1b1max = floorf(sqrtf(1.*(k1a1max*k1a1max - i*i)));
      
      for(j=1;j<=k1b1max;j++)
	{
	  b1 = -j;
	  j2 = (b1<0)? N2+b1 : b1; 
	  k1c1max = floorf(sqrtf(1.*(k1a1max*k1a1max - i*i - j*j)));
 	  
	  for(k=1;k<=k1c1max;k++)
	    {
	      index1 = i2*N2*(N3/2+1) + j2*(N3/2+1) + k;
	      
	      k1 = sqrtf(fac1*a1*a1 + fac2*b1*b1 + fac3*k*k); // is |k1|/tpibyL
	      
	      d1 = floorf((log10(tpibyL*k1) - log10(k1min))/scalek1); //logarithmic bins
	      //if(d1<0 || d1>=Nk1bin){printf("%d %d %d %d\n",d1, a1, b1, k);}

	      if(d1>=0 && d1<Nk1bin)
		{
		  //-----------------------------------------------------
		  
		  skip1[1] = (i/(k1*N1) < 0.5)? 6 : 9;
		  skip1[2] = (j/(k1*N2) < 0.5)? 0 : 9;
		  skip1[3] = (k/(k1*N3) < 0.5)? 3 : 9;
		  //printf("%d %d %d %d\n", skip1[0], skip1[1], skip1[2], skip1[3]);
		  
		  //-----------------------------------------------------
		  
		  k2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k)));
		  
		  for(jk=0; jk<8; jk++)
		    {
		      if(jk == skip1[0]){continue;}
		      if(jk == skip1[1]){continue;}
		      if(jk == skip1[2]){continue;}
		      if(jk == skip1[3]){continue;}
		      
		      delta2flag = (k2flag[jk][0] < 0)?  -1 :  1;
		      
		      for(ii=0; ii<=k2max; ii++)
			{
			  a2 = k2flag[jk][2]*ii;
			  k2a2 = fac1*a2*a2;
			  costa2 = - fac1*a1*a2/k1;
			  
			  if(k2flag[jk][0] < 0) ii2 = (a2>0)? N1-a2 : -a2; 
			  else ii2 = (a2<0)? N1+a2 : a2;
			  
			  k2b2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii)));
			  
			  for(jj=0; jj<=k2b2max; jj++)
			    {
			      b2 = k2flag[jk][1]*jj;
			      k2b2 = k2a2 + fac2*b2*b2; 
			      costb2 = costa2 - fac2*b1*b2/k1;
			      
			      if(k2flag[jk][0] < 0) jj2 = (b2>0)? N2-b2 : -b2;
			      else jj2 = (b2<0)? N2+b2 : b2;
			      
			      k2c2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii - jj*jj)));
			      k2c2min = (0.25*(a1*a1 + b1*b1 + k*k) <= 1.*(ii*ii + jj*jj))? 0 : ceilf(sqrtf(0.25*(a1*a1 + b1*b1 + k*k) - 1.*(ii*ii + jj*jj)));
			      
			      for(kk=k2c2min; kk<=k2c2max; kk++)
				{
				  c2 = k2flag[jk][0]*kk;
				  
				  cost = costb2 - fac3*k*c2/k1; // is cost*k2
				  
				  if(cost/k1 >= 0.5)
				    {
				      k2 = sqrtf(k2b2 + fac3*c2*c2); // is |k2|/tpibyL
				      
				      d2 = floorf((k2/k1 - 0.5)/scalen); //if(d2<0 || d2>=Nnbin){printf("Not ok at d2=%d\n", d2);}
				      
				      d3 = (cost/k2 <= 0.5)? 0 : floorf((cost/k2 - 0.5)/scalecost); //if(d3<0 || d3>=Ncostbin){printf("Not ok at d3=%d\n", d3);}

				      //-----------------------------------------------------
				      
				      if(c2 == 0)
					{
					  ii2 = (a2<0)? N1+a2 : a2; 
					  jj2 = (b2<0)? N2+b2 : b2;
					  
					  delta2flag = 1;
					}
				      
				      //-----------------------------------------------------
				      
				      if(-(k  + c2) < 0)
					{
					  iii = (-(a1 + a2)>0)? N1+(a1 + a2) : (a1 + a2);
					  jjj = (-(b1 + b2)>0)? N2+(b1 + b2) : (b1 + b2);
					  
					  delta3flag = -1; 
					}
				      else
					{
					  iii = (-(a1 + a2)<0)? N1-(a1 + a2) : -(a1 + a2); 
					  jjj = (-(b1 + b2)<0)? N2-(b1 + b2) : -(b1 + b2);
					  
					  delta3flag = 1;
					}
				      
				      //-----------------------------------------------------
				      
				      index2 = ii2*N2*(N3/2+1) + jj2*(N3/2+1) + kk;
				      index3 = iii*N2*(N3/2+1) + jjj*(N3/2+1) + abs(k + c2);
				      
				      bispec2[d2][d3][d1] += comp_ro[index1][0]*comp_ro[index2][0]*comp_ro[index3][0] - comp_ro[index1][1]*delta2flag*comp_ro[index2][1]*comp_ro[index3][0] - comp_ro[index1][0]*delta2flag*comp_ro[index2][1]*delta3flag*comp_ro[index3][1] - comp_ro[index1][1]*comp_ro[index2][0]*delta3flag*comp_ro[index3][1];
				      
				      no2[d2][d3][d1] += 1;
				      
				    } // end of d3 check 
				  
				} // end of k2 z-direction
			      
			    } // end of k2 y-direction
			  
			} // end of k2 x-direction
		      
		    } // end of k2 

		}  // end of d1 check
	      
	    } // end of k1 z-direction
 	  
	} // end of k1 y-direction
      
    } // end of k1 x-direction
  
  printf("cube %d/4 is done\n",2);
  
}

//-----------------------------------------------------------------------------//


void * bispec_cal_c3()
{
  int i, j, k, ii, jj, kk;
  int ii2, jj2, iii, jjj, kkk, ll, jk;
  int a1, b1, a2, b2, c2;
  
  long index1, index2, index3;
  int d1, d2, d3;
  int k2max, k2b2max, k2c2min, k2c2max;
  int delta2flag, delta3flag;
  float k2a2, k2b2, costa2, costb2, k1, k2, cost;
  
  int mm;
  int i2, j2;
  int skip1[4];
  int k1a1max, k1b1max, k1c1max;
  
  //-----------------------------------------------------
  
  k1a1max = floorf(N1*k1max/tpibyL);
  skip1[0] = 4;
  
  for(i=1;i<k1a1max;i++)
    {
      a1 = -i;
      i2 = (a1<0)? N1+a1 : a1; 
      k1b1max = floorf(sqrtf(1.*(k1a1max*k1a1max - i*i)));
      
      for(j=1;j<=k1b1max;j++)
	{
	  b1 = j;
	  j2 = (b1<0)? N2+b1 : b1; 
	  k1c1max = floorf(sqrtf(1.*(k1a1max*k1a1max - i*i - j*j)));
	  
	  for(k=1;k<=k1c1max;k++)
	    {
	      index1 = i2*N2*(N3/2+1) + j2*(N3/2+1) + k;
	      
	      k1 = sqrtf(fac1*a1*a1 + fac2*b1*b1 + fac3*k*k); // is |k1|/tpibyL
	      
	      d1 = floorf((log10(tpibyL*k1) - log10(k1min))/scalek1); //logarithmic bins
	      //if(d1<0 || d1>=Nk1bin){printf("%d %d %d %d\n",d1, a1, b1, k);}

	      if(d1>=0 && d1<Nk1bin)
		{
		  //-----------------------------------------------------
		  
		  skip1[1] = (i/(k1*N1) < 0.5)? 0 : 9;
		  skip1[2] = (j/(k1*N2) < 0.5)? 6 : 9;
		  skip1[3] = (k/(k1*N3) < 0.5)? 5 : 9;
		  //printf("%d %d %d %d\n", skip1[0], skip1[1], skip1[2], skip1[3]);
		  
		  //-----------------------------------------------------
		  
		  k2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k)));
		  
		  for(jk=0; jk<8; jk++)
		    {
		      if(jk == skip1[0]){continue;}
		      if(jk == skip1[1]){continue;}
		      if(jk == skip1[2]){continue;}
		      if(jk == skip1[3]){continue;}
		      
		      delta2flag = (k2flag[jk][0] < 0)?  -1 :  1;
		      
		      for(ii=0; ii<=k2max; ii++)
			{
			  a2 = k2flag[jk][2]*ii;
			  k2a2 = fac1*a2*a2;
			  costa2 = - fac1*a1*a2/k1;
			  
			  if(k2flag[jk][0] < 0) ii2 = (a2>0)? N1-a2 : -a2; 
			  else ii2 = (a2<0)? N1+a2 : a2;
			  
			  k2b2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii)));
			  
			  for(jj=0; jj<=k2b2max; jj++)
			    {
			      b2 = k2flag[jk][1]*jj;
			      k2b2 = k2a2 + fac2*b2*b2; 
			      costb2 = costa2 - fac2*b1*b2/k1;
			      
			      if(k2flag[jk][0] < 0) jj2 = (b2>0)? N2-b2 : -b2;
			      else jj2 = (b2<0)? N2+b2 : b2;
			      
			      k2c2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii - jj*jj)));
			      k2c2min = (0.25*(a1*a1 + b1*b1 + k*k) <= 1.*(ii*ii + jj*jj))? 0 : ceilf(sqrtf(0.25*(a1*a1 + b1*b1 + k*k) - 1.*(ii*ii + jj*jj)));
			      
			      for(kk=k2c2min; kk<=k2c2max; kk++)
				{
				  c2 = k2flag[jk][0]*kk;
				  
				  cost = costb2 - fac3*k*c2/k1; // is cost*k2
				  
				  if(cost/k1 >= 0.5)
				    {
				      k2 = sqrtf(k2b2 + fac3*c2*c2); // is |k2|/tpibyL
				      
				      d2 = floorf((k2/k1 - 0.5)/scalen); //if(d2<0 || d2>=Nnbin){printf("Not ok at d2=%d\n", d2);}
				      
				      d3 = (cost/k2 <= 0.5)? 0 : floorf((cost/k2 - 0.5)/scalecost); //if(d3<0 || d3>=Ncostbin){printf("Not ok at d3=%d\n", d3);}
				      
				      //-----------------------------------------------------
				      
				      if(c2 == 0)
					{
					  ii2 = (a2<0)? N1+a2 : a2; 
					  jj2 = (b2<0)? N2+b2 : b2;
					  
					  delta2flag = 1;
					}
				      
				      //-----------------------------------------------------
				      
				      if(-(k  + c2) < 0)
					{
					  iii = (-(a1 + a2)>0)? N1+(a1 + a2) : (a1 + a2);
					  jjj = (-(b1 + b2)>0)? N2+(b1 + b2) : (b1 + b2);
					  
					  delta3flag = -1; 
					}
				      else
					{
					  iii = (-(a1 + a2)<0)? N1-(a1 + a2) : -(a1 + a2); 
					  jjj = (-(b1 + b2)<0)? N2-(b1 + b2) : -(b1 + b2);
					  
					  delta3flag = 1;
					}
				      
				      //-----------------------------------------------------
				      
				      index2 = ii2*N2*(N3/2+1) + jj2*(N3/2+1) + kk;
				      index3 = iii*N2*(N3/2+1) + jjj*(N3/2+1) + abs(k + c2);
				      
				      bispec3[d2][d3][d1] += comp_ro[index1][0]*comp_ro[index2][0]*comp_ro[index3][0] - comp_ro[index1][1]*delta2flag*comp_ro[index2][1]*comp_ro[index3][0] - comp_ro[index1][0]*delta2flag*comp_ro[index2][1]*delta3flag*comp_ro[index3][1] - comp_ro[index1][1]*comp_ro[index2][0]*delta3flag*comp_ro[index3][1];
				      
				      no3[d2][d3][d1] += 1;
				      
				    } // end of d3 check 
				  
				} // end of k2 z-direction
			      
			    } // end of k2 y-direction
			  
			} // end of k2 x-direction
		      
		    } // end of k2 

		} // end of d1 check
	      
	    } // end of k1 z-direction
	  
	} // end of k1 y-direction
      
    } // end of k1 x-direction
  
  printf("cube %d/4 is done\n",3);
}

//-----------------------------------------------------------------------------//


void * bispec_cal_c4()
{
  int i, j, k, ii, jj, kk;
  int ii2, jj2, iii, jjj, kkk, ll, jk;
  int a1, b1, a2, b2, c2;
  
  long index1, index2, index3;
  int d1, d2, d3;
  int k2max, k2b2max, k2c2min, k2c2max;
  int delta2flag, delta3flag;
  float k2a2, k2b2, costa2, costb2, k1, k2, cost;
  
  int mm;
  int i2, j2;
  int skip1[4];
  int k1a1max, k1b1max, k1c1max;
  
  //-----------------------------------------------------
  
  k1a1max = floorf(N1*k1max/tpibyL);
  skip1[0] = 6;

  for(i=1;i<k1a1max;i++)
    {
      a1 = -i;
      i2 = (a1<0)? N1+a1 : a1; 
      k1b1max = floorf(sqrtf(1.*(k1a1max*k1a1max - i*i)));
      
      for(j=1;j<=k1b1max;j++)
	{
	  b1 = -j;
	  j2 = (b1<0)? N2+b1 : b1; 
	  k1c1max = floorf(sqrtf(1.*(k1a1max*k1a1max - i*i - j*j)));
	  
	  for(k=1;k<=k1c1max;k++)
	    {
	      index1 = i2*N2*(N3/2+1) + j2*(N3/2+1) + k;
	      
	      k1 = sqrtf(fac1*a1*a1 + fac2*b1*b1 + fac3*k*k); // is |k1|/tpibyL
	      
	      d1 = floorf((log10(tpibyL*k1) - log10(k1min))/scalek1); //logarithmic bins
	      //if(d1<0 || d1>=Nk1bin){printf("%d %d %d %d\n",d1, a1, b1, k);}

	      if(d1>=0 && d1<Nk1bin)
		{
		  //-----------------------------------------------------

		  skip1[1] = (i/(k1*N1) < 0.5)? 2 : 9;
		  skip1[2] = (j/(k1*N2) < 0.5)? 4 : 9;
		  skip1[3] = (k/(k1*N3) < 0.5)? 7 : 9;
		  //printf("%d %d %d %d\n", skip1[0], skip1[1], skip1[2], skip1[3]);
		  
		  //-----------------------------------------------------
		  
		  k2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k)));
		  
		  for(jk=0; jk<8; jk++)
		    {
		      if(jk == skip1[0]){continue;}
		      if(jk == skip1[1]){continue;}
		      if(jk == skip1[2]){continue;}
		      if(jk == skip1[3]){continue;}
		      
		      delta2flag = (k2flag[jk][0] < 0)?  -1 :  1;
		      
		      for(ii=0; ii<=k2max; ii++)
			{
			  a2 = k2flag[jk][2]*ii;
			  k2a2 = fac1*a2*a2;
			  costa2 = - fac1*a1*a2/k1;
			  
			  if(k2flag[jk][0] < 0) ii2 = (a2>0)? N1-a2 : -a2; 
			  else ii2 = (a2<0)? N1+a2 : a2;
			  
			  k2b2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii)));
			  
			  for(jj=0; jj<=k2b2max; jj++)
			    {
			      b2 = k2flag[jk][1]*jj;
			      k2b2 = k2a2 + fac2*b2*b2; 
			      costb2 = costa2 - fac2*b1*b2/k1;
			      
			      if(k2flag[jk][0] < 0) jj2 = (b2>0)? N2-b2 : -b2;
			      else jj2 = (b2<0)? N2+b2 : b2;
			      
			      k2c2max = floorf(sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii - jj*jj)));
			      k2c2min = (0.25*(a1*a1 + b1*b1 + k*k) <= 1.*(ii*ii + jj*jj))? 0 : ceilf(sqrtf(0.25*(a1*a1 + b1*b1 + k*k) - 1.*(ii*ii + jj*jj)));
			      
			      for(kk=k2c2min; kk<=k2c2max; kk++)
				{
				  c2 = k2flag[jk][0]*kk;
				  
				  cost = costb2 - fac3*k*c2/k1; // is cost*k2
				  
				  if(cost/k1 >= 0.5)
				    {
				      k2 = sqrtf(k2b2 + fac3*c2*c2); // is |k2|/tpibyL
				      
				      d2 = floorf((k2/k1 - 0.5)/scalen); //if(d2<0 || d2>=Nnbin){printf("Not ok at d2=%d\n", d2);}
				      
				      d3 = (cost/k2 <= 0.5)? 0 : floorf((cost/k2 - 0.5)/scalecost); //if(d3<0 || d3>=Ncostbin){printf("Not ok at d3=%d\n", d3);}
				      
				      //-----------------------------------------------------
				      
				      if(c2 == 0)
					{
					  ii2 = (a2<0)? N1+a2 : a2; 
					  jj2 = (b2<0)? N2+b2 : b2;
					  
					  delta2flag = 1;
					}
				      
				      //-----------------------------------------------------
				      
				      if(-(k  + c2) < 0)
					{
					  iii = (-(a1 + a2)>0)? N1+(a1 + a2) : (a1 + a2);
					  jjj = (-(b1 + b2)>0)? N2+(b1 + b2) : (b1 + b2);
					  
					  delta3flag = -1; 
					}
				      else
					{
					  iii = (-(a1 + a2)<0)? N1-(a1 + a2) : -(a1 + a2); 
					  jjj = (-(b1 + b2)<0)? N2-(b1 + b2) : -(b1 + b2);
					  
					  delta3flag = 1;
					}
				      
				      //-----------------------------------------------------
				      
				      index2 = ii2*N2*(N3/2+1) + jj2*(N3/2+1) + kk;
				      index3 = iii*N2*(N3/2+1) + jjj*(N3/2+1) + abs(k + c2);
				      
				      bispec4[d2][d3][d1] += comp_ro[index1][0]*comp_ro[index2][0]*comp_ro[index3][0] - comp_ro[index1][1]*delta2flag*comp_ro[index2][1]*comp_ro[index3][0] - comp_ro[index1][0]*delta2flag*comp_ro[index2][1]*delta3flag*comp_ro[index3][1] - comp_ro[index1][1]*comp_ro[index2][0]*delta3flag*comp_ro[index3][1];
				      
				      no4[d2][d3][d1] += 1;
				      
				    } // end of d3 check 
				  
				} // end of k2 z-direction
			      
			    } // end of k2 y-direction
			  
			} // end of k2 x-direction
		      
		    } // end of k2
		  
		} // end of d1 check
	      
	    } // end of k1 z-direction
	  
	} // end of k1 y-direction
      
    } // end of k1 x-direction
  
  printf("cube %d/4 is done\n",4);
}

//-----------------------------------------------------------------------------//












  /*
  bispec_cal(1, N1/2, 1, 1, N2/2, 1, 1, N3/2, 1, 0, 9); // xy ++ cube
  printf("1st cube is done\n");
  
  bispec_cal(1, N1/2, 1, N2/2+1, N2, 1, 1, N3/2, 1, 2, 9); // xy +- cube
  printf("2nd cube is done\n");
  
  bispec_cal(N1/2+1, N1, 1, 1, N2/2, 1, 1, N3/2, 1, 4, 9); // xy -+ cube
  printf("3rd cube is done\n");
  
  bispec_cal(N1/2+1, N1, 1, N2/2+1, N2, 1, 1, N3/2, 1, 6, 9); // xy -- cube
  printf("4th cube is done\n");
  */



//printf("%d %d %d | %d %d %d | %f %e %d %d | %f %d \n", i, j, k, ii, jj, kk, k2/k1, cost/k2, d2, d3, sqrtf(1.*(a1*a1 + b1*b1 + k*k - ii*ii - jj*jj)), k2c2max);


//-----------------------------------------------------------------------------//
//-----------------------------------------------------------------------------//
/*
int *crossProduct(int a1, int b1, int c1, int a2, int b2, int c2) 
{
  int *cross_P;
  cross_P = (int *) calloc(3,sizeof(int));
  
  cross_P[0] = b1 * c2 - c1 * b2; 
  cross_P[1] = c1 * a2 - a1 * c2; 
  cross_P[2] = a1 * b2 - b1 * a2;

  return(cross_P);
}
*/
