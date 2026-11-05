#ifndef G1_HISTORY_H
#define G1_HISTORY_H
#define G5_PHOTON_LOCAL
#include "photon_physics.h"
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
P5_FN void g1_history_legacy(const G1Tables *t,const G1Geometry *g,RngState *rng,
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

#ifndef G5_TEST_TWO_PHOTONS
#define G5_TEST_TWO_PHOTONS 0
#endif
#ifndef G5_PHOTON_STACK_CAP
#define G5_PHOTON_STACK_CAP 32
#endif
struct G5Photon {double xyz[3],dir[3],energy;int scatter,fluorescent;};
// All descendants share box_local. No per-history moment is committed until
// the stack is empty, and each terminal photon is aggregated separately.
P5_FN void g1_history(const G1Tables *t,const G1Geometry *g,RngState *rng,
 const double *origin,const double *direction,double energy,bool fluor,double *out,long long *ints,
 double *trace=0,int trace_capacity=0,const G2Tally *tally=0,double *agg=0,int history=0,int terminal_capacity=0){
#if G5_TEST_LEGACY_LOCAL
    g1_history_legacy(t,g,rng,origin,direction,energy,fluor,out,ints,trace,trace_capacity,tally);
    if(agg){g1_add_record(t,g,out,ints,agg);int np=g->det?int(g->det[14])*int(g->det[15]):0;
#ifdef P5_GPU
        atomicAdd(agg+t->nm+6+11*np,1.);atomicAdd(agg+t->nm+7+11*np,1.);
#else
        agg[t->nm+6+11*np]+=1.;agg[t->nm+7+11*np]+=1.;
#endif

        if(terminal_capacity){int base=t->nm+6+11*np,at=base+10003;int slot;
#ifdef P5_GPU
            slot=int(atomicAdd(agg+at,1.));
#else
            slot=int(agg[at]++);
#endif
            double *row=agg+at+1+4*slot;row[0]=history;row[1]=ints[3]==2?ints[4]*int(g->det[15])+ints[5]:ints[3]==1?-1:-2;row[2]=ints[6];row[3]=out[0];}
        if(ints[3]==1){int bin=int(out[0]);if(bin>10000)bin=10000;
#ifdef P5_GPU
            atomicAdd(agg+t->nm+8+11*np+bin,1.);
#else
            agg[t->nm+8+11*np+bin]+=1.;
#endif
}}
#else
    int np=g->det?int(g->det[14])*int(g->det[15]):0,base=t->nm+6+11*np;
    double box_local[64]={};G5Photon stack[G5_PHOTON_STACK_CAP];int top=0;
#if G5_PHOTON_FAULT==5
    for(int slot=0;slot<G5_PHOTON_STACK_CAP;++slot)stack[slot].fluorescent=0;
#endif
    for(int k=0;k<3;++k){stack[0].xyz[k]=origin[k];stack[0].dir[k]=direction[k];}
    stack[0].energy=energy;stack[0].scatter=stack[0].fluorescent=0;
    for(int j=0;j<t->nm+2;++j)out[j]=0.;for(int j=0;j<9;++j)ints[j]=0;
    int created=1,ended=0,hit_count=0,pixels[128];
#if G5_TEST_TWO_PHOTONS
    // Fixed verification-only pair, each carrying half the incident energy.
    top=1;created=2;stack[0].energy=energy/2;stack[1]=stack[0];
    stack[0].dir[0]=.1;stack[1].dir[0]=-.1;stack[0].dir[2]=stack[1].dir[2]=sqrt(.99);
    stack[0].fluorescent=stack[1].fluorescent=1;
#endif
    double hit_energy[128];bool failed=false;
    while(top>=0 && !failed){
        G5Photon photon=stack[top--];double e=photon.energy,x=photon.xyz[0],y=photon.xyz[1],z=photon.xyz[2];
        double dx=photon.dir[0],dy=photon.dir[1],dz=photon.dir[2],tau=-log(rng_uniform(rng));
        long long rec[9]={};rec[4]=rec[5]=rec[6]=-1;rec[8]=1;double final[34]={};
        for(int step=0;step<1000000;++step){
            int m=g1_material(x,y,z,g);const double *pt=g1_photons(t,m);
            if(trace && step<trace_capacity){double *a=trace+8*step;a[0]=step;a[1]=x;a[2]=y;a[3]=z;a[4]=m;a[5]=rng->calls;a[6]=e;a[7]=tau;}
            if(e<5){out[1]+=e;out[2+m]+=e;rec[3]=0;rec[8]=0;break;}
            double pp=g5_total(pt,e,0),pc=g5_total(pt,e,1),pr=g5_total(pt,e,2),total=pp+pc+pr;
            double mu=total*g1_row(t,DENS)[m],boundary[2];g1_boundary(x,y,z,dx,dy,dz,g,boundary);
            double tb=mu*boundary[0];bool interact=tau<tb;double ds=interact?tau/(mu>0?mu:1e-30):boundary[0];
            double o[3]={x,y,z},dir[3]={dx,dy,dz},hit[4];g1_detector(o,dir,ds,g->det,hit);
            if(tally)g2_segment(tally,m,o,dir,hit[3]?hit[0]:ds,e,box_local);
            if(hit[3]){rec[3]=2;rec[4]=(long long)hit[1];rec[5]=(long long)hit[2];rec[6]=photon.fluorescent?3:photon.scatter==0?0:photon.scatter==1?1:2;rec[8]=0;break;}
            x+=dx*ds;y+=dy*ds;z+=dz*ds;
            if(!interact){tau-=tb;x+=dx*1e-6;y+=dy*1e-6;z+=dz*1e-6;if(boundary[1]){rec[3]=1;rec[8]=0;break;}continue;}
            ++ints[0];double rt=rng_uniform(rng)*total,newdir[3];
            if(rt<pp){
                int el=g5_choose_element(pt,e,1,rng),sh=g5_shell(pt,el,e,rng);double emitted[36],local;long long tr[27];int events;
                int used=g5_relax(pt,el,sh,rng,fluor,emitted,&local,tr,&events);
                if(used<0 || top+used>=G5_PHOTON_STACK_CAP || G5_PHOTON_FAULT==3){failed=true;break;}
                double carried=0.;for(int j=0;j<used;++j)carried+=emitted[4*j];
                double dep=e-carried;out[1]+=dep;out[2+m]+=dep;created+=used;ints[2]+=used;
                for(int j=0;j<used;++j){if(G5_PHOTON_FAULT==4 && j==0)continue;
                    G5Photon &child=stack[++top];child.xyz[0]=x;child.xyz[1]=y;child.xyz[2]=z;
                    for(int k=0;k<3;++k)child.dir[k]=emitted[4*j+1+k];child.energy=emitted[4*j];
                    child.scatter=photon.scatter;if(G5_PHOTON_FAULT!=5)child.fluorescent=1;
                    if(child.fluorescent!=1)failed=true;}
                rec[3]=0;rec[8]=0;break;
            }else if(rt<pp+pc){
                double sample[2];g5_local_compton(pt,e,rng,sample);
                double enew=e*sample[0];out[1]+=e-enew;out[2+m]+=e-enew;e=enew;
                g1_scatter(dx,dy,dz,sample[1],rng,newdir);
            }else g1_scatter(dx,dy,dz,g5_rayleigh(pt,e,rng),rng,newdir);
            ++photon.scatter;++ints[1];dx=newdir[0];dy=newdir[1];dz=newdir[2];tau=-log(rng_uniform(rng));
        }
        if(rec[8]){failed=true;break;}++ended;final[0]=e;
        if(agg && terminal_capacity){int at=base+10003;int slot;
#ifdef P5_GPU
            slot=int(atomicAdd(agg+at,1.));
#else
            slot=int(agg[at]++);
#endif
            if(slot>=terminal_capacity){failed=true;break;}
            double *row=agg+at+1+4*slot;row[0]=history;row[1]=rec[3]==2?rec[4]*int(g->det[15])+rec[5]:rec[3]==1?-1:-2;
            row[2]=rec[6];row[3]=e;
        }
        if(agg){g1_add_record(t,g,final,rec,agg);
            if(rec[3]==1){int bin=int(e);if(bin>10000)bin=10000;
#ifdef P5_GPU
                atomicAdd(agg+base+2+bin,1.);
#else
                agg[base+2+bin]+=1.;
#endif
}
            if(rec[3]==2){if(hit_count==128){failed=true;break;}
                pixels[hit_count]=int(rec[4])*int(g->det[15])+int(rec[5]);hit_energy[hit_count++]=e;}}
        for(int j=3;j<7;++j)ints[j]=rec[j];out[0]=e;
    }
    // Photon-number conservation detects dropped secondaries independently of
    // energy bookkeeping. Slot ancestry is assigned on every push.
    if(created!=ended)failed=true;
    ints[8]=failed?1:0;ints[7]=rng->calls;
    if(agg){
        // Per-material deposition is committed once for the incident history.
        for(int m=0;m<t->nm;++m){
#ifdef P5_GPU
            atomicAdd(agg+m,out[2+m]);
#else
            agg[m]+=out[2+m];
#endif
        }

        for(int i=0;i<hit_count;++i)for(int j=0;j<i;++j)if(pixels[i]==pixels[j]){
            double cross=2*hit_energy[i]*hit_energy[j];int at=t->nm+5+6*np+pixels[i];
#ifdef P5_GPU
            atomicAdd(agg+at,cross);
#else
            agg[at]+=cross;
#endif
        }
#ifdef P5_GPU
        atomicAdd(agg+t->nm+3,double(ints[0]));atomicAdd(agg+t->nm+4,double(ints[2]));atomicAdd(agg+base,failed?0.:1.);atomicAdd(agg+base+1,double(created));if(failed)atomicAdd(agg+base-1,1.);
#else
        agg[t->nm+3]+=ints[0];agg[t->nm+4]+=ints[2];agg[base]+=failed?0.:1.;agg[base+1]+=created;if(failed)agg[base-1]+=1.;
#endif
    }
    if(tally && !failed)g2_end_history(tally,box_local);
#endif
}
#endif
