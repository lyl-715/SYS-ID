#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/* usage: sod x.bin K cand.bin M N0 -> prints first candidate (descending order given) with N>=N0, and its N */
int main(int argc,char**argv){
  int K=atoi(argv[2]); long M=atol(argv[4]); int N0=atoi(argv[5]);
  double*x=malloc(K*sizeof(double)); FILE*f=fopen(argv[1],"rb"); fread(x,8,K,f); fclose(f);
  double*c=malloc(M*sizeof(double)); f=fopen(argv[3],"rb"); fread(c,8,M,f); fclose(f);
  long evals=0;
  for(long m=0;m<M;m++){ double D=c[m]; double last=x[0]; int n=1;
    for(int k=1;k<K;k++){ if(fabs(x[k]-last)>=D){n++; last=x[k];} }
    evals++;
    if(n>=N0){ printf("%.18g %d %ld\n",D,n,evals); return 0; }
  }
  printf("none\n"); return 0; }
