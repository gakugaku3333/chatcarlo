#ifndef G1_HISTORY_H
#define G1_HISTORY_H
#include "g1_physics.h"
#include "g1_geometry.h"
#include "g2_tally.h"
#ifndef P5_GPU
using std::isfinite;
#endif
// One history, ported in order from kernel.py:_transport_one and
// transport.py:transport_photons (terminal detection precedes boundary/event).
// Integer record: interactions, Compton+Rayleigh, fluorescence, ending
// (0 absorbed/1 escaped/2 detected), iu, iv, category, RNG calls, error.
// Double record: final energy, deposited total, per-material deposits.
P5_FN void g1_history(const G1Tables *t,const G1Geometry *g,RngState *rng,
                      const double *origin,const double *direction,double energy,
                      bool fluor,double *out,long long *ints,
                      double *trace=0,int trace_capacity=0,const G2Tally *tally=0) {
    double box_local[64]={};
    for(int j=0;j<2+t->nm;++j)out[j]=0.;
    for(int j=0;j<9;++j)ints[j]=0;
    ints[4]=ints[5]=ints[6]=-1;ints[8]=1;
    double x=origin[0],y=origin[1],z=origin[2],dx=direction[0],dy=direction[1],dz=direction[2];
    double e=energy,tau=-log(rng_uniform(rng));
    double p[32],c[32],r[32],f[32];int idx[32];
    for(int step=0;step<1000000;++step) {
        int m=g1_material(x,y,z,g);
        if(trace && step<trace_capacity){double *a=trace+step*8;a[0]=step;a[1]=x;a[2]=y;a[3]=z;a[4]=m;a[5]=rng->calls;a[6]=e;a[7]=tau;}
        double part[4];g1_mu(e,m,t,p,c,r,idx,f,part);
        double boundary[2];g1_boundary(x,y,z,dx,dy,dz,g,boundary);
        double mu=part[0],mu_safe=mu>0.?mu:1e-30,tb=mu*boundary[0];
        bool interact=tau<tb;double ds=interact?tau/mu_safe:boundary[0];
        double o[3]={x,y,z},dir[3]={dx,dy,dz},hit[4];g1_detector(o,dir,ds,g->det,hit);
        if(tally)g2_segment(tally,m,o,dir,hit[3]?hit[0]:ds,e,box_local);
        if(hit[3]){ints[3]=2;ints[4]=(long long)hit[1];ints[5]=(long long)hit[2];
            ints[6]=ints[2]>0?3:ints[1]==0?0:ints[1]==1?1:2;ints[8]=0;break;}
        x+=dx*ds;y+=dy*ds;z+=dz*ds;
        if(!interact){tau-=tb;x+=dx*1e-6;y+=dy*1e-6;z+=dz*1e-6;
            if(boundary[1]){ints[3]=1;ints[8]=0;break;}continue;}
        ++ints[0];double rt=rng_uniform(rng),total=part[1]+part[2]+part[3];
        double pp=part[1]/total,pc=part[2]/total,deposit=0.,newdir[3];
        if(rt<pp){
            int ei=g1_select(int(g1_row(t,NE)[m]),g1_row(t,FRACS,m),p,part[1],rng);
            double line=fluor?g1_fluorescence(m,ei,e,idx[ei],f[ei],t,rng):0.;
            if(line>0.){deposit=e-line;out[1]+=deposit;out[2+m]+=deposit;++ints[2];
                g1_isotropic(rng,newdir);dx=newdir[0];dy=newdir[1];dz=newdir[2];e=line;tau=-log(rng_uniform(rng));}
            else {out[1]+=e;out[2+m]+=e;ints[3]=0;ints[8]=0;break;}
        } else if(rt<pp+pc){
            ++ints[1];double sample[3];g1_compton(m,e,t,c,part[2],rng,sample);
            double enew=e*sample[0];deposit=e-enew;out[1]+=deposit;out[2+m]+=deposit;
            g1_scatter(dx,dy,dz,sample[1],rng,newdir);dx=newdir[0];dy=newdir[1];dz=newdir[2];e=enew;tau=-log(rng_uniform(rng));
        } else {
            ++ints[1];double sample[2];g1_rayleigh(m,e,t,r,part[3],rng,sample);
            g1_scatter(dx,dy,dz,sample[0],rng,newdir);dx=newdir[0];dy=newdir[1];dz=newdir[2];tau=-log(rng_uniform(rng));
        }
        if(step==999999)ints[8]=1;
    }
    if(tally)g2_end_history(tally,box_local);
    out[0]=e;ints[7]=(long long)rng->calls;
}
P5_FN void g1_add_record(const G1Tables *t,const G1Geometry *g,const double *d,const long long *i,double *agg) {
    // nm deposits, 3 endings, scatter sum, fluorescence sum, then 7 images.
    for(int m=0;m<t->nm;++m) {
#ifdef P5_GPU
        atomicAdd(agg+m,d[2+m]);
#else
        agg[m]+=d[2+m];
#endif
    }
    int nm=t->nm;
    int np=g->det?int(g->det[14])*int(g->det[15]):0;
    bool invalid=i[8]!=0 || i[3]<0 || i[3]>2;
    for(int j=0;j<2+nm;++j)if(!isfinite(d[j]) || d[j]<0.)invalid=true;
    if(i[3]==2 && (!g->det || i[4]<0 || i[4]>=g->det[14] || i[5]<0 || i[5]>=g->det[15]))invalid=true;
    if(invalid){
#ifdef P5_GPU
        atomicAdd(agg+nm+5+11*np,1.);
#else
        agg[nm+5+11*np]+=1.;
#endif
        return;
    }
#ifdef P5_GPU
    atomicAdd(agg+nm+i[3],1.);atomicAdd(agg+nm+3,double(i[0]));atomicAdd(agg+nm+4,double(i[2]));
#else
    agg[nm+i[3]]+=1.;agg[nm+3]+=i[0];agg[nm+4]+=i[2];
#endif
    if(i[3]!=2)return;
    int nu=int(g->det[14]),nv=int(g->det[15]),npix=nu*nv;
    int pix=int(i[4])*nv+int(i[5]);double area=g->det[12]*g->det[13]/(nu*nv);
    int a=nm+5;double e=d[0];
#ifdef P5_GPU
    atomicAdd(agg+a+7*npix+int(i[6])*npix+pix,1.);
    atomicAdd(agg+a+int(i[6])*npix+pix,e/area);atomicAdd(agg+a+4*npix+pix,1.);
    atomicAdd(agg+a+5*npix+pix,e);atomicAdd(agg+a+6*npix+pix,e*e);
#else
    agg[a+7*npix+int(i[6])*npix+pix]+=1.;
    agg[a+int(i[6])*npix+pix]+=e/area;agg[a+4*npix+pix]+=1.;
    agg[a+5*npix+pix]+=e;agg[a+6*npix+pix]+=e*e;
#endif
}
#endif
