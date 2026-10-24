#include "p5d_gxx_compat.h"
#include "p8g_kernel.h"
#include "p8g_reference.h"
extern "C" void p8g_run(const double *data,const long long *meta,const double *constants,const P8gGeometry *g,uint64_t seed,uint64_t start,int n,int maxsteps,P8gOutput *o,double *histories,long long *hcalls,double *sources,double *deps,double *diagnostic,long long *dm) {
    RngState rng;rng_init(&rng,seed,0);P8gLocal local={};
    for(int h=0;h<n;++h){
#ifdef P5_RNG_PHILOX
        rng_init(&rng,seed,start+uint64_t(h));
#endif
        uint64_t before=rng.calls;int b=h*100/n;if(b>99)b=99;
        P5Diagnostic d={diagnostic?diagnostic+14*h:0,diagnostic?2:0,0,0};
        p8g_history(data,meta,constants,g,&rng,b,maxsteps,o,diagnostic?&d:0,&local);
        if(histories)for(int j=0;j<g->nb;++j)histories[h*g->nb+j]=local.bins[j];
        if(hcalls)hcalls[h]=rng.calls-before;if(sources)sources[h]=local.source;if(deps)deps[h]=local.dep;
        if(dm)dm[h]=d.used<2?d.used:2;
#if P8F_FAULT != 7
        for(int j=0;j<g->nb;++j)local.bins[j]=0.;
#endif
        for(int j=0;j<g->nc;++j)local.coarse[j]=0.;for(int j=0;j<g->nf;++j)local.fine[j]=0.;
    }
}
// Batched function probes retain the MT state across inputs. The complete MT
// array/index/call counter and subsequent draws are exported, not just a seed.
extern "C" void p8g_probe(int which,const double *data,const long long *meta,const double *constants,int mid,const double *x,int n,int width,uint64_t seed,double *out,long long *state,double *next) {
    P8gMaterial m=p8g_material(data,meta+136*mid,constants);RngState rng;rng_init(&rng,seed,0);
    for(int i=0;i<n;++i){double T=x[i*width],*y=out+128*i;
        if(which==0)y[0]=p8g_collision(T,m);
        if(which==1)p8g_hard(T,m,y,y+1);
        if(which==2)p8g_coefficients(T,m,y);
        if(which==3)p8g_quantities(T,m,y);
        if(which==4)y[0]=p8g_shape(T,x[i*width+1],m);
        if(which==5)y[0]=p8g_annihilation_rate(T,m);
        if(which==6)p8g_shell_distribution(T,m,y);
        if(which==7){int e,s;p8g_select_shell(T,m,&rng,&e,&s);y[0]=e;y[1]=s;}
        if(which==8){int excess;y[0]=p8g_sample_k(T,m,&rng,&excess);y[1]=excess;}
        if(which==9){long long trace[24];int events;double photons[32],local;int used=p8g_relax(int(T),int(x[i*width+1]),m,&rng,bool(x[i*width+2]),photons,&local,trace,&events);y[0]=used;y[1]=events;y[2]=local;for(int j=0;j<32;++j)y[3+j]=photons[j];for(int j=0;j<24;++j)y[35+j]=trace[j];}
        if(which==10)p8g_table_rates(T,m,y,y+1);
        if(which==11){double total,G1;p8g_table_rates(T,m,&total,&G1);y[0]=p8g_gs_mu_cached(T,x[i*width+1],m,total,G1,&rng);}
        if(which==12)p5_compton(T,0.,0.,1.,&m.p,&rng,y);
        if(which==13)p5_pair(T,0.,0.,1.,&m.p,&rng,y);
        if(which==14)p5_annihilation(&rng,y);
        if(which==15)p5_flight_photons(T,0.,0.,1.,&m.p,&rng,y);
        if(which==16){P5State a;p5_initial(&a,T,0.,0.,0.,0.,0.,1.,0,0,&m.p,&rng);for(int j=0;j<23;++j)y[j]=a.a[j];}
        if(which==17){double a[23];for(int j=0;j<23;++j)a[j]=x[i*width+j];p8g_material_boundary(a,int(x[i*width+23]),int(x[i*width+24]),m,x[i*width+25],x[i*width+26],x[i*width+27],&rng,y+23,x[i*width+28]);for(int j=0;j<23;++j)y[j]=a[j];}
        if(which==18)y[0]=p8g_sum(x+i*width,width);
        if(which==20){double T=x[i*width],angle=x[i*width+1],v[3];p5_rotate(0.,0.,1.,cos(m.p.constants[0]/(T+m.p.constants[0])),angle,v);for(int j=0;j<3;++j)y[j]=v[j];}
        if(which==21){if(T){double u=rng_uniform(&rng);y[0]=-log(u);}else y[0]=INFINITY;}
        if(which==22)y[0]=p5_max(0.,T-x[i*width+1]*x[i*width+2]);
        if(which==23){P5State a;for(int j=0;j<23;++j)a.a[j]=x[i*width+j];p5_energy_init(&a,&rng);for(int j=0;j<23;++j)y[j]=a.a[j];}
        if(which==26){int touch[2]={int(T),int(x[i*width+1])};y[0]=p5_touched_sum(x+i*width+2,touch);}
        if(which==25)y[0]=p8g_event(T,x[i*width+1],x[i*width+2],x[i*width+3],x[i*width+4],x[i*width+5],x[i*width+6]);
        if(which==24){int e,sh;p8g_choose_shell(x+i*width+1,(width-1)/5,T,&e,&sh);y[0]=e;y[1]=sh;}

        if(which==19){double v[3];p5_rotate(x[i*width],x[i*width+1],x[i*width+2],x[i*width+3],x[i*width+4],v);for(int j=0;j<3;++j)y[j]=v[j];}
    }
#ifdef P5_RNG_MT
    for(int j=0;j<624;++j)state[j]=rng.mt[j];state[624]=rng.index;
#endif
    state[625]=rng.calls;for(int j=0;j<16;++j)next[j]=rng_uniform(&rng);
}
extern "C" void p8g_score_probe(const P8gGeometry *g,const double *x,int n,double *out,double *bound) {
    for(int j=0;j<n;++j){p8g_score(out+j*g->nb,g->nb,x[5*j],x[5*j+1],int(x[5*j+2]),x[5*j+3],g->bins,g->owner);
        int nr;p8g_boundary(x[5*j],x[5*j+4],int(x[5*j+2]),g->faces,bound+2*j,&nr);bound[2*j+1]=nr;}
}

extern "C" void p8g_histories(const double *data,const long long *meta,const double *constants,const P8gGeometry *g,uint64_t seed,uint64_t start,int n,double *d,long long *ints){
    P8gLocal l={};P8gRecord r;RngState rng;rng_init(&rng,seed,0);
    for(int h=0;h<n;++h){
#ifdef P5_RNG_PHILOX
        rng_init(&rng,seed,start+uint64_t(h));
#endif
        p8g_record_history(data,meta,constants,g,&rng,10000000,&r,&l);
        for(int j=0;j<p8g_dwidth(g);++j)d[h*p8g_dwidth(g)+j]=r.d[j];
        for(int j=0;j<p8g_iwidth(g);++j)ints[h*p8g_iwidth(g)+j]=r.i[j];p8g_clear(g,&l);
    }
}

extern "C" void p8g_fixed(const double *data,const long long *meta,const double *constants,const P8gGeometry *g,uint64_t seed,const double *state,P8gOutput *o,double *events,long long *mt,double *next){
    RngState rng;rng_init(&rng,seed,0);P8gLocal l={};P5State a;for(int j=0;j<23;++j)a.a[j]=state[j];
    p8g_history(data,meta,constants,g,&rng,0,10000000,o,0,&l,&a);
    for(int j=0;j<160;++j)events[j]=l.events[j];
#ifdef P5_RNG_MT
    for(int j=0;j<624;++j)mt[j]=rng.mt[j];mt[624]=rng.index;
#endif
    mt[625]=rng.calls;mt[626]=l.event_used;for(int j=0;j<16;++j)next[j]=rng_uniform(&rng);
}

extern "C" void p8g_fixed_transitions(const double *data,const long long *meta,const double *constants,const P8gGeometry *g,uint64_t seed,const double *state,long long *transitions){
    P5State a;for(int j=0;j<23;++j)a.a[j]=state[j];P8gLocal l={};P8gRecord r;RngState rng;rng_init(&rng,seed,0);
    p8g_record_history(data,meta,constants,g,&rng,10000000,&r,&l,&a);
    for(int j=0;j<80;++j)transitions[j]=l.transitions[j];
}

extern "C" void p8g_init(double *data,const long long *meta,const double *constants,int nm){for(int mid=0;mid<nm;++mid){P8gMaterial m=p8g_material(data,meta+136*mid,constants);p8g_initialize(m);}}
extern "C" void p8g_values(double *data,const long long *meta,const double *constants,int mid,double *out){P8gMaterial m=p8g_material(data,meta+136*mid,constants);for(int s=0;s<P8G_SITES;++s)for(int j=0;j<p8g_length(m,s);++j){double *r=out+5*(p8g_offset(m,s)+j);for(int k=0;k<5;++k)r[k]=NAN;if(p8g_valid(m,s,j))p8g_expression(m,s,j,r);r[0]=p8g_reference_value(m,s,j);}}
// Direct comparison of the two lower-level copies (including raw outputs).
extern "C" void p8g_parts(const double *data,const long long *meta,const double *constants,int mid,const double *x,int n,double *out){
 P8gMaterial m=p8g_material(data,meta+136*mid,constants);
 for(int i=0;i<n;++i){double *y=out+40*i;for(int k=0;k<40;++k)y[k]=0.;int idx1,idx2;double T=x[i];
 quantities(T,m.brems_E,m.brems_V,m.p.nbrems,m.log_stop_T,m.stop_rad,m.nstop,.01,0,y,y+1,y+2,y+3,&idx1,y+4);
 p8g_quantities_raw(T,m.brems_E,m.brems_V,m.p.nbrems,m.log_stop_T,m.stop_rad,m.nstop,.01,0,y+20,y+21,y+22,y+23,&idx2,y+24,m);
 y[16]=idx1;y[36]=idx2;
 }
}
// Synthetic rounded-log and duplicate-node probe. Same initializer expressions.
extern "C" void p8g_interp_probe(const double *x,const double *y,int n,const double *E,int count,double *out){
 double logs[512];P8gMaterial m={};unsigned long long reads[37]={};m.p.nbrems=0;m.p.nt=1;m.npair=n;m.nstop=n;m.nphoto=1;m.nz=0;m.ncollision=n;m.logs=logs;m.reads=reads;m.mid=-1;
 double thresholds[2]={.5,.5};m.pair_E=x;m.thresholds=thresholds;
 // Both search directions, with duplicate/rounded ties and endpoint extrapolation.
 for(int j=0;j<n;++j){logs[p8g_offset(m,8)+j]=log(x[j]);logs[p8g_offset(m,9)+j]=log(y[j]);}
 for(int j=0;j<n;++j){logs[p8g_offset(m,5)+j]=log(x[j]);double q=1-.5/x[j];logs[p8g_offset(m,6)+j]=log(y[j]/((q*q)*q));}
 for(int j=0;j<count;++j){out[4*j]=p8f_log_interp(E[j],x,y,n);out[4*j+1]=p8g_log_interp(E[j],x,y,n,m,8,9);out[4*j+2]=p5_pair_interp(E[j],x,y,n,1,.5);out[4*j+3]=p8g_pair_interp(E[j],x,y,n,1,.5,m,6);}
}
