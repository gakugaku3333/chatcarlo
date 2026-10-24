#ifndef P8G_LOGS_H
#define P8G_LOGS_H
#include "p8f_materials.h"
#ifndef P8G_CHECK
#define P8G_CHECK 0
#endif
#ifndef P8G_FAULT
#define P8G_FAULT 0
#endif
#define P8G_SITES 18
struct P8gMaterial : P8Material { double *logs; unsigned long long *reads; int mid; const double *faultlogs; };
P5_FN int p8g_length(const P8gMaterial &m,int s) {
 int n=m.p.ratecols==3?m.p.nrates:m.p.nt;
 return s==0?m.p.nbrems:s==1?3*m.p.nbrems:s==2?n-1:s==3?2*n:s==4?m.p.nt-1:
 s>=5 && s<=7?m.npair:s==8 || s==9?m.nstop:s==10 || s==11?m.nphoto:
 s==12 || s==13?m.nphoto-1:s==14?m.nphoto:s>=15?m.nz:0;
}
P5_FN int p8g_offset(const P8gMaterial &m,int s){int o=0;for(int k=0;k<s;++k)o+=p8g_length(m,k);return o;}
P5_FN bool p8g_valid(const P8gMaterial &m,int s,int j) {
 if(j<0 || j>=p8g_length(m,s))return false;
 if(s==1 && j<3)return false;
 if((s==6 || s==7) && m.pair_E[j]<=m.thresholds[s-6])return false;
 if(s==9 && !m.ncollision)return false;
 if((s==12 || s==13) && m.photo_E[j+1]==m.photo_E[j])return false;
 if((s==15 || s==16) && m.z[j]==1)return false;
 return true;
}
// Auxiliary layout: value, quotient, q, q^3, argument. Not acceptance criteria.
P5_FN double p8g_expression(const P8gMaterial &m,int s,int j,double *r=0) {
 const P5Tables *t=&m.p;double arg=0.;
 if(s==0)arg=m.brems_E[j];
 if(s==1)arg=m.brems_V[5*(j/3)+j%3];
 if(s==2){int i=j,shift=t->ratecols==3?1:0;double e0=shift?t->rates[i*t->ratecols]:t->ts[i];double e1=shift?t->rates[(i+1)*t->ratecols]:t->ts[i+1];arg=e1/e0;if(r)r[1]=arg;}
 if(s==3){int shift=t->ratecols==3?1:0;arg=t->rates[(j/2)*t->ratecols+shift+j%2];}
 if(s==4){arg=t->ts[j+1]/t->ts[j];if(r)r[1]=arg;}
 if(s==5)arg=m.pair_E[j];
 if(s==6 || s==7){double q=1-m.thresholds[s-6]/m.pair_E[j];arg=(s==6?m.pair_n[j]:m.pair_e[j])/((q*q)*q);if(r){r[1]=arg;r[2]=q;r[3]=(q*q)*q;}}
 if(s==8)arg=m.stop_T[j];if(s==9)arg=m.stop_collision[j];
 if(s==10)arg=m.photo_E[j];if(s==11 || s==14)arg=m.photo[j];
 if(s==12){arg=m.photo_E[j+1]/m.photo_E[j];if(r)r[1]=arg;}
 if(s==13){arg=m.photo[j+1]/m.photo[j];if(r)r[1]=arg;}
 if(s==15)arg=1194.*pow(m.z[j],-2./3.);
 if(s==16)arg=184.15*pow(m.z[j],-1./3.);
 if(s==17)arg=m.z[j];
 if(r)r[4]=arg;return log(arg);
}
P5_FN P8gMaterial p8g_material(const double *data,const long long *q,const double *constants){
 P8gMaterial m;static_cast<P8Material&>(m)=p8f_material(data,q,constants);
 m.logs=const_cast<double*>(data)+q[132];m.reads=reinterpret_cast<unsigned long long*>(const_cast<double*>(data)+q[133]);m.mid=q[134];m.faultlogs=data+q[135];return m;
}
P5_FN void p8g_initialize(P8gMaterial &m){
 for(int s=0;s<P8G_SITES;++s)for(int j=0;j<p8g_length(m,s);++j)m.logs[p8g_offset(m,s)+j]=p8g_valid(m,s,j)?p8g_expression(m,s,j):NAN;
#if P8G_FAULT == 1
 if(m.mid==0)m.logs[p8g_offset(m,11)]=nextafter(m.logs[p8g_offset(m,11)],INFINITY);
#endif
 for(int j=0;j<2*P8G_SITES+1;++j)m.reads[j]=0;
}
P5_FN void p8g_count(unsigned long long *p){
#ifdef P5_GPU
 atomicAdd(p,1ULL);
#else
 ++*p;
#endif
}
P5_FN double p8g_read(const P8gMaterial &m,int s,int j){
 bool target=m.mid==0 && ((P8G_FAULT==1 && s==11 && j==0) || (P8G_FAULT!=1 && s==17 && j==1));
#if P8G_FAULT == 3
 if(target)j=p8g_length(m,s);
#endif
#if P8G_CHECK
 p8g_count(m.reads+2*s);if(target)p8g_count(m.reads+2*P8G_SITES);
 if(!p8g_valid(m,s,j)){p8g_count(m.reads+2*s+1);return NAN;}
#endif
 const double *base=m.logs+p8g_offset(m,s);
#if P8G_FAULT == 2
 if(m.mid==0 && s==17)base=m.faultlogs;
#endif
 double value=base[j];
#if P8G_CHECK
 if(!isfinite(value))p8g_count(m.reads+2*s+1);
#endif
 return value;
}
#endif
