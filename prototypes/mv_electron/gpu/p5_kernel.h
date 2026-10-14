#ifndef P5_KERNEL_H
#define P5_KERNEL_H
#include "p5_rng.h"

// Tables and scalar constants come from Python, without regenerated C tables.
struct P5Tables {
    const double *constants,*loget,*ed,*ts,*ks,*us,*ys,*rates,*pes,*water,*thresholds;
    int ne,nt,nk,nu,nrates,ratecols,np,watercols;
    const double *er,*Z,*NJ,*dl,*brems_e,*brems_v,*edges,*cum;
    double bpar,delcm,delpos;
    int nz,nbrems,nspec;
};
// constants: M,C,I,RE,NE. Analytic constants retain the original expressions.
P5_FN double p5_min(double a,double b) {return b<a?b:a;}
P5_FN double p5_max(double a,double b) {return b>a?b:a;}
P5_FN int p5_clip_bin(int b,int n) {return b<0?0:(b>=n?n-1:b);}
P5_FN double p5_collision(double T,const P5Tables *t) {
    double M=t->constants[0],C=t->constants[1],I=t->constants[2];
    double tau=T/M,g=1+tau,b2=1-1/(g*g),d=tau/2;
    double F=-1-b2+log((tau-d)*d)+tau/(tau-d)
        +(d*d/2+(2*tau+1)*log1p(-d/tau))/(g*g);
    double density=interp(log(T),t->loget,t->ed,t->ne);
    double im=I/M;
    return C/b2*(log(2*(tau+2)/(im*im))+F-density);
}
P5_FN void p5_moller_params(double T,const P5Tables *t,double *a,double *b,double *c) {
    double g=1+T/t->constants[0],gg=g*g,b2=1-1/gg,gm=g-1;
    *a=gm*gm/gg;*b=(2*g-1)/gg;*c=t->constants[1]/(b2*T);
}
P5_FN double p5_primitive0(double x,double a,double b) {
    return a*x-1/x+1/(1-x)-b*log(x/(1-x));
}
P5_FN double p5_primitive1(double x,double a,double b) {
    return a*x*x/2+log(x)+1/(1-x)+(1+b)*log1p(-x);
}
P5_FN void p5_hard(double T,const P5Tables *t,double *rate,double *stop) {
    *rate=0.;*stop=0.;if(T<=2*.01)return;
    double a,b,c;p5_moller_params(T,t,&a,&b,&c);double x=.01/T;
    if(.5-x<1e-7) {
        double h=(T/2-.01)/T,mid=.5-h/2,om=1-mid;
        double v=a+1/(mid*mid)+1/(om*om)-b/(mid*(1-mid));
        *rate=c*h*v;*stop=c*T*h*mid*v;return;
    }
    *rate=c*(p5_primitive0(.5,a,b)-p5_primitive0(x,a,b));
    *stop=c*T*(p5_primitive1(.5,a,b)-p5_primitive1(x,a,b));
}
P5_FN double p5_sample_W(double T,const P5Tables *t,RngState *s) {
    double a,b,c;p5_moller_params(T,t,&a,&b,&c);double lo=.01/T;
    for(;;) {
        double u=rng_uniform(s);
        double x=1/(1/lo-u*(1/lo-2));
        double q=x/(1-x),v=1+a*x*x+q*q-b*x/(1-x);
        double accept=rng_uniform(s);
        if(accept*(2+a/4)<=v)return T*x;
    }
}
P5_FN double p5_kn_total(double E,const P5Tables *t) {
    double x=E/t->constants[0],L=log1p(2*x),one=1+2*x;
    double RE=t->constants[3],NE=t->constants[4];
    return 2*3.141592653589793*(RE*RE)*NE*((1+x)/(x*x)
        *(2*(1+x)/(1+2*x)-L/x)+L/(2*x)-(1+3*x)/(one*one));
}
P5_FN double p5_log_interp(double E,const double *es,const double *vs,int n,int stride) {
    // np.interp searches the LOG grid, including ties after rounding log().
    double x=log(E);
    if(x<log(es[0]))return exp(log(vs[0]));
    if(x>log(es[n-1]))return exp(log(vs[(n-1)*stride]));
    int lo=0,hi=n;
    while(lo<hi) {int m=lo+(hi-lo)/2;if(log(es[m])<x)lo=m+1;else hi=m;}
    int j=lo;
    if(j<n && x==log(es[j]))return exp(log(vs[j*stride]));
    --j;
    double y0=log(vs[j*stride]),y1=log(vs[(j+1)*stride]);
    double slope=(y1-y0)/(log(es[j+1])-log(es[j]));
    return exp(slope*(x-log(es[j]))+y0);
}
P5_FN double p5_pair_value(int j,const double *es,const double *vs,int stride,double th) {
    double q=1-th/es[j];return log(vs[j*stride]/((q*q)*q));
}
P5_FN double p5_pair_interp(double E,const double *es,const double *vs,int n,int stride,double th) {
    if(E<=th)return 0.;
    int start=0;while(start<n && es[start]<=th)++start;
    double x=log(E),y;
    if(x<log(es[start]))y=p5_pair_value(start,es,vs,stride,th);
    else if(x>log(es[n-1]))y=p5_pair_value(n-1,es,vs,stride,th);
    else {
        int lo=start,hi=n;
        while(lo<hi) {int m=lo+(hi-lo)/2;if(log(es[m])<x)lo=m+1;else hi=m;}
        int j=lo;
        if(j<n && x==log(es[j]))y=p5_pair_value(j,es,vs,stride,th);
        else {
            --j;double y0=p5_pair_value(j,es,vs,stride,th);
            double y1=p5_pair_value(j+1,es,vs,stride,th);
            double slope=(y1-y0)/(log(es[j+1])-log(es[j]));
            y=slope*(x-log(es[j]))+y0;
        }
    }
    double q=1-th/E;return exp(y)*((q*q)*q);
}
P5_FN void p5_coefficients(double E,const P5Tables *t,double *c) {
    c[0]=p5_kn_total(E,t);
    c[1]=p5_log_interp(E,t->pes,t->water+2,t->np,t->watercols);
    c[2]=p5_pair_interp(E,t->pes,t->water+3,t->np,t->watercols,t->thresholds[0]);
    c[3]=p5_pair_interp(E,t->pes,t->water+4,t->np,t->watercols,t->thresholds[1]);
}
P5_FN void p5_table_rates(double T,const P5Tables *t,double *total,double *G1) {
    int shift=t->ratecols==3?1:0,n=shift?t->nrates:t->nt,lo=0,hi=n;
    while(lo<hi) {int m=lo+(hi-lo)/2;double e=shift?t->rates[m*t->ratecols]:t->ts[m];
        if(e<T)lo=m+1;else hi=m;}
    int i=lo-1;if(i<0)i=0;if(i>n-2)i=n-2;
    double e0=shift?t->rates[i*t->ratecols]:t->ts[i];
    double e1=shift?t->rates[(i+1)*t->ratecols]:t->ts[i+1];
    double a=log(T/e0)/log(e1/e0);
    *total=exp((1-a)*log(t->rates[i*t->ratecols+shift])+a*log(t->rates[(i+1)*t->ratecols+shift]));
    *G1=exp((1-a)*log(t->rates[i*t->ratecols+shift+1])+a*log(t->rates[(i+1)*t->ratecols+shift+1]));
}
P5_FN void p5_bracket(const double *grid,int n,double x,int *i,double *a) {
    *i=lower_bound(x,grid,n)-1;if(*i<0)*i=0;if(*i>n-2)*i=n-2;
    *a=(x-grid[*i])/(grid[*i+1]-grid[*i]);
}
P5_FN double p5_gs_mu_cached(double T,double K,const P5Tables *t,double total,double G1,RngState *s) {
    double p0=exp(-K/G1*total),zero=rng_uniform(s);
    if(zero<p0)return 1.;
    double u=rng_uniform(s);int i=lower_bound(T,t->ts,t->nt)-1;
    if(i<0)i=0;if(i>t->nt-2)i=t->nt-2;
    double a=log(T/t->ts[i])/log(t->ts[i+1]/t->ts[i]);int j,h;double b,c;
    p5_bracket(t->ks,t->nk,K,&j,&b);p5_bracket(t->us,t->nu,u,&h,&c);
    int q00=(i*t->nk+j)*t->nu+h,q01=q00+t->nu;
    int q10=q00+t->nk*t->nu,q11=q10+t->nu;
    double y00=(1-c)*t->ys[q00]+c*t->ys[q00+1];
    double y01=(1-c)*t->ys[q01]+c*t->ys[q01+1];
    double y10=(1-c)*t->ys[q10]+c*t->ys[q10+1];
    double y11=(1-c)*t->ys[q11]+c*t->ys[q11+1];
    return 1-((1-a)*((1-b)*y00+b*y01)+a*((1-b)*y10+b*y11));
}
P5_FN void p5_local(double *score,int n,double z,double loss,double dz) {
    int b=p5_clip_bin(int(z/dz),n);score[b]+=loss;
}
P5_FN void p5_score_segment(double *score,int n,double z0,double z1,double loss,double dz) {
    if(loss==0)return;
    if(fabs(z1-z0)<1e-15) {p5_local(score,n,z0,loss,dz);return;}
    double lo=p5_min(z0,z1),hi=p5_max(z0,z1),width=hi-lo;
    int b0=p5_clip_bin(int(lo/dz),n),b1=p5_clip_bin(int(hi/dz),n);
    for(int b=b0;b<=b1;++b) {
        double length=p5_max(0.,p5_min(hi,(b+1)*dz)-p5_max(lo,b*dz));
        score[b]+=loss*length/width;
    }
}

// Integer powers are expanded in the same multiply order as Numba int_power.
P5_FN void p5_rotate(double ux,double uy,double uz,double mu,double phi,double *v) {
    double sn=sqrt(p5_max(0.,1-mu*mu)),r=sqrt(p5_max(0.,1-uz*uz));
    double ax,ay,az,bx,by;
    if(r>1e-12) {ax=ux*uz/r;ay=uy*uz/r;az=-r;bx=-uy/r;by=ux/r;}
    else {ax=1.;ay=0.;az=0.;bx=0.;by=uz>=0?1.:-1.;}
    double cp=cos(phi),sp=sin(phi);
    v[0]=mu*ux+sn*(cp*ax+sp*bx);v[1]=mu*uy+sn*(cp*ay+sp*by);v[2]=mu*uz+sn*cp*az;
}
P5_FN void p5_compton(double E,double ux,double uy,double uz,const P5Tables *t,RngState *s,double *out) {
    double M=t->constants[0],qmin=1/(1+2*E/M),a=log(1/qmin),b=(1-qmin*qmin)/2,q,mu;
    for(;;) {
        double choose=rng_uniform(s);
        if(choose*(a+b)<a) {double u=rng_uniform(s);q=exp(-u*a);}
        else {double u=rng_uniform(s);q=sqrt(qmin*qmin+u*(1-qmin*qmin));}
        mu=1-M/E*(1/q-1);double sin2=p5_max(0.,1-mu*mu);
        double accept=rng_uniform(s);if(accept*(q+1/q)<=q+1/q-sin2)break;
    }
    double u=rng_uniform(s),v[3];p5_rotate(ux,uy,uz,mu,2*3.141592653589793*u,v);
    double ep=E*q,T=E-ep,px=E*ux-ep*v[0],py=E*uy-ep*v[1],pz=E*uz-ep*v[2];
    double p=sqrt(px*px+py*py+pz*pz);out[0]=ep;
    for(int j=0;j<3;++j)out[j+1]=v[j];
    if(p==0) {out[4]=0.;out[5]=ux;out[6]=uy;out[7]=uz;return;}
    out[4]=T;out[5]=px/p;out[6]=py/p;out[7]=pz/p;
}
P5_FN double p5_pair_small_total(double E,const P5Tables *t,RngState *s) {
    double M=t->constants[0];
    if(E<=2.1) {double u=rng_uniform(s);return M+u*(E/2-M);}
    for(;;) {
        double r1=rng_uniform(s);double r=rng_uniform(s),br;int lvl;
        if(r>=t->bpar) {
            double r2=rng_uniform(s);double r3=rng_uniform(s);
            br=.5*(1-p5_max(p5_max(r1,r2),r3));lvl=0;
        } else {br=r1*.5;lvl=2;}
        if(E*br<M)continue;
        double d=1/(E*br*(1-br));if(d>=t->delpos)continue;
        double delta=t->delcm*d,rej;const double *v=t->dl+lvl*6;
        if(delta<1)rej=v[0]+delta*(v[1]+delta*v[2]);
        else rej=v[3]+v[4]*log(delta+v[5]);
        double accept=rng_uniform(s);if(accept<=rej)return E*br;
    }
}
P5_FN void p5_pair(double E,double ux,double uy,double uz,const P5Tables *t,RngState *s,double *p) {
    double e2=p5_pair_small_total(E,t,s),e1=E-e2,M=t->constants[0],mu=cos(M/E);
    double u=rng_uniform(s),phi=2*3.141592653589793*u,d1[3],d2[3];
    p5_rotate(ux,uy,uz,mu,phi,d1);p5_rotate(ux,uy,uz,mu,phi+3.141592653589793,d2);
    double charge=rng_uniform(s),charge1=charge<=.5?1.:-1.;
    p[0]=e1-M;p[5]=e2-M;p[4]=charge1;p[9]=-charge1;
    for(int j=0;j<3;++j) {p[1+j]=d1[j];p[6+j]=d2[j];}
}
P5_FN void p5_annihilation(RngState *s,double *v) {
    double u=rng_uniform(s),mu=2*u-1;double r=rng_uniform(s),phi=2*3.141592653589793*r;
    double sn=sqrt(1-mu*mu);v[0]=sn*cos(phi);v[1]=sn*sin(phi);v[2]=mu;
}
P5_FN double p5_annihilation_rate(double T,const P5Tables *t) {
    double g=1+T/t->constants[0],p=sqrt(g*g-1),RE=t->constants[3];
    double sigma=3.141592653589793*(RE*RE)/(g+1)
        *((g*g+4*g+1)/(g*g-1)*log(g+p)-(g+3)/p);
    return t->constants[4]*sigma;
}
P5_FN void p5_flight_photons(double T,double ux,double uy,double uz,const P5Tables *t,RngState *s,double *v) {
    double M=t->constants[0],g=1+T/M,lo=1/(g+1+sqrt(g*g-1));
    double width=-log(lo/(1-lo)),A=g*g+4*g+1,z;
    for(;;) {
        double u=rng_uniform(s),odds=exp(-width+width*u);z=odds/(1+odds);
        double gp=g+1;
        double acceptance=1-(2*(gp*gp)*z*(1-z)+(1-z)/z+z/(1-z))/A;
        double accept=rng_uniform(s);if(accept<=acceptance)break;
    }
    double energy=T+2*M,mu=(g+1-1/z)/sqrt(g*g-1);
    double k1=z*energy,k2=(1-z)*energy;mu=p5_max(-1.,p5_min(1.,mu));
    double u=rng_uniform(s),v1[3];p5_rotate(ux,uy,uz,mu,2*3.141592653589793*u,v1);
    double p=sqrt(T*(T+2*M)),x=p*ux-k1*v1[0],y=p*uy-k1*v1[1],zz=p*uz-k1*v1[2];
    double norm=sqrt(x*x+y*y+zz*zz);
    v[0]=k1;v[1]=v1[0];v[2]=v1[1];v[3]=v1[2];v[4]=k2;v[5]=x/norm;v[6]=y/norm;v[7]=zz/norm;
}
P5_FN double p5_shape(double T,double k,const P5Tables *t) {
    if(k<0. || k>T)return 0.;
    double M=t->constants[0],RE=t->constants[3],alpha=t->constants[5];
    double E=T+M,rem=E-k,y=rem/E,result=0.;
    for(int j=0;j<t->nz;++j) {
        double z=t->Z[j],az=alpha*z,az2=az*az;
        double fc=az2*(1/(1+az2)+.20206+az2*(-.0369+az2*(.0083-.002*az2)));
        double xi=z==1?6.144/(5.31-fc):log(1194.*pow(z,-2./3.))/(log(184.15*pow(z,-1./3.))-fc);
        double xl=4./3.*log(z),d=136.*pow(z,-1./3.)*M/E*k/rem;
        if(d>=exp((21.12-xl)/4.184)-.952)continue;
        double s1,s2;
        if(d<=1) {s1=20.867+d*(-3.242+.625*d)-xl;s2=20.209+d*(-1.930-.086*d)-xl;}
        else {s1=21.12-4.184*log(d+.952)-xl;s2=s1;}
        result+=t->NJ[j]*(RE*RE)*alpha*z*(z+xi)*((1+y*y)*s1-2./3.*y*s2);
    }
    return result;
}
P5_FN void p5_quantities(double T,const P5Tables *t,double *q) {
    int idx;double raw[12];
    quantities(T,t->brems_e,t->brems_v,t->nbrems,t->loget,
               t->er,t->ne,.01,0,q,q+1,q+2,q+3,&idx,raw);
}
P5_FN double p5_sample_k(double T,const P5Tables *t,RngState *s,int *excess) {
    double q[4];p5_quantities(T,t,q);double envelope=q[3],lo=log(.01),width=log(T/.01);
    *excess=0;
    for(;;) {
        double u=rng_uniform(s),k=p5_min(T,p5_max(.01,exp(lo+width*u)));
        double f=p5_shape(T,k,t);if(f>envelope)++*excess;
        double accept=rng_uniform(s);if(accept*envelope<=f)return k;
    }
}

#ifndef P5_STACK_CAP
#define P5_STACK_CAP 64
#endif
#ifndef P5_TRANSPORT_FAULT
#define P5_TRANSPORT_FAULT 0
#endif
struct P5State {double a[23];};
struct P5Output {
    double *batches,*fine_batches,*totals,*maxima,*brems;
    long long *violations,*counts,*nspec,*step_batches;
    int maxdepth;
};
struct P5Diagnostic {double *data;int capacity,used;long long overflow;};
P5_FN void p5_record(P5Diagnostic *d,const P5State *p,uint64_t calls) {
    if(!d)return;
    if(d->used<d->capacity) {
        double *v=d->data+7*d->used;
        v[0]=p->a[19];v[1]=p->a[6];v[2]=p->a[0];v[3]=p->a[1];v[4]=p->a[2];
        v[5]=double(calls);v[6]=double(d->used);
    } else ++d->overflow;
    ++d->used;
}
P5_FN double p5_rule(double T) {return T<=.02?.1:(T<=1.?.05:.025);}
P5_FN void p5_energy_init(P5State *p,RngState *s) {
    double *a=p->a;a[7]=a[6];a[11]=p5_min(.05*a[7],p5_max(0.,a[7]-.01));
    double u=rng_uniform(s);a[10]=u*a[11];a[8]=a[10];a[9]=a[11]-a[10];a[18]=0.;
}
P5_FN void p5_initial(P5State *p,double T,double x,double y,double z,double ux,double uy,double uz,int region,int ancestry,const P5Tables *t,RngState *s) {
    (void)t;
    for(int j=0;j<23;++j)p->a[j]=0.;double *a=p->a;
    a[2]=z;a[3]=ux;a[4]=uy;a[5]=uz;a[6]=T;a[17]=0.;a[16]=region;
    p5_energy_init(p,s);double k=p5_rule(T);a[14]=k;
    double scatter=rng_uniform(s);a[12]=scatter*k;a[13]=k-a[12];
    double hard=rng_uniform(s);a[15]=-log(hard);
    a[19]=-1.;double br=rng_uniform(s);a[22]=-log(br);a[0]=x;a[1]=y;a[20]=ancestry;
}
P5_FN void p5_photon(P5State *p,double E,double x,double y,double z,double ux,double uy,double uz,int region,int ancestry) {
    for(int j=0;j<23;++j)p->a[j]=0.;double *a=p->a;
    a[2]=z;a[3]=ux;a[4]=uy;a[5]=uz;a[6]=E;a[20]=ancestry;a[0]=x;a[1]=y;a[16]=region;
}
P5_FN void p5_boundary(double z,double w,int region,double *distance,int *next) {
    *next=region;
    if(w>0) {*distance=p5_max(0.,(30.-z)/w);*next=region+1;}
    else if(w<0) {*distance=p5_max(0.,(0.-z)/w);*next=region-1;}
    else *distance=INFINITY;
}
P5_FN void p5_track_local(double *score,int nb,double z,double loss,double dz,int *touch) {
#if P5_TRANSPORT_FAULT == 2
    // Fault affects charged deposits only (the caller chooses this path).
    z+=dz;
#endif
    p5_local(score,nb,z,loss,dz);int b=p5_clip_bin(int(z/dz),nb);
    if(b<touch[0])touch[0]=b;if(b>touch[1])touch[1]=b;
}
P5_FN void p5_track_segment(double *score,int nb,double z0,double z1,double loss,double dz,int *touch) {
#if P5_TRANSPORT_FAULT == 2
    z0+=dz;z1+=dz;
#endif
    p5_score_segment(score,nb,z0,z1,loss,dz);if(loss==0)return;
    int b0=p5_clip_bin(int(p5_min(z0,z1)/dz),nb),b1=p5_clip_bin(int(p5_max(z0,z1)/dz),nb);
    if(b0<touch[0])touch[0]=b0;if(b1>touch[1])touch[1]=b1;
}
P5_FN double p5_touched_sum(const double *score,const int *touch) {
    double result=0.;for(int i=touch[0];i<=touch[1];++i)result+=score[i];return result;
}

P5_FN void history(const P5Tables *t,RngState *s,int batch,int maxsteps,P5Output *o,P5Diagnostic *diagnostic) {
    P5State stack[P5_STACK_CAP],state;double hs[150]={0.},fine[600]={0.};int touch[4]={150,-1,600,-1};
    double M=t->constants[0],hn=0.,expected_n=0.,de=0.,radiation=0.;
    double u1=rng_uniform(s);double u2=rng_uniform(s);int j=0;
    while(t->cum[j]<u1)++j;
    double width=t->edges[j+1]-t->edges[j],source_energy=t->edges[j]+u2*width;
    int cell=int((source_energy-t->edges[j])/width*10);if(cell>9)cell=9;++o->nspec[j*10+cell];
    p5_photon(stack,source_energy,0.,0.,0.,0.,0.,1.,0,0);
    int top=0,steps=0;double dep=0.,escaped=0.,rest=0.,positron_escape_rest=0.;bool failed=false;
    long long *counts=o->counts;
    while(top>=0) {
        state=stack[top--];double *a=state.a;
        if(a[19]==0) {
            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];break;}
            double E=a[6];
            if(E<.01) {
                // The fault macro for charged deposits must not change photons.
                p5_local(hs,150,a[2],E,.2);p5_local(fine,600,a[2],E,.05);
                int b0=p5_clip_bin(int(a[2]/.2),150),b1=p5_clip_bin(int(a[2]/.05),600);
                if(b0<touch[0])touch[0]=b0;if(b0>touch[1])touch[1]=b0;
                if(b1<touch[2])touch[2]=b1;if(b1>touch[3])touch[3]=b1;
                dep+=E;continue;
            }
            double c[4];p5_coefficients(E,t,c);double mu=0.;for(int z=0;z<4;++z)mu+=c[z];
            double bb;int nr;p5_boundary(a[2],a[5],int(a[16]),&bb,&nr);
            double flight_u=rng_uniform(s),tau=-log(flight_u),flight=tau/mu,length=p5_min(flight,bb);
            a[0]+=length*a[3];a[1]+=length*a[4];a[2]+=length*a[5];
            if(bb<=flight) {escaped+=E;++counts[7];continue;}
#if P5_TRANSPORT_FAULT == 3
            double unused=rng_uniform(s);(void)unused;
#endif
            double select=rng_uniform(s),r=select*mu;int process=0;
            while(process<3 && r>=c[process]) {r-=c[process];++process;}
            ++counts[8+process];
            if(top+3>=P5_STACK_CAP) {failed=true;++counts[6];break;}
            if(process==0) {
                double p[8];p5_compton(E,a[3],a[4],a[5],t,s,p);
                ++top;p5_photon(stack+top,p[0],a[0],a[1],a[2],p[1],p[2],p[3],int(a[16]),int(a[20]));
                ++top;p5_initial(stack+top,p[4],a[0],a[1],a[2],p[5],p[6],p[7],int(a[16]),int(a[20]),t,s);
            } else if(process==1) {
                ++top;p5_initial(stack+top,E,a[0],a[1],a[2],a[3],a[4],a[5],int(a[16]),int(a[20]),t,s);
            } else {
                double p[10];p5_pair(E,a[3],a[4],a[5],t,s,p);rest+=2*M;
                P5State first,second;
                p5_initial(&first,p[0],a[0],a[1],a[2],p[1],p[2],p[3],int(a[16]),1,t,s);first.a[19]=p[4];
                p5_initial(&second,p[5],a[0],a[1],a[2],p[6],p[7],p[8],int(a[16]),1,t,s);second.a[19]=p[9];
                if(first.a[19]==1) {double u=rng_uniform(s);first.a[21]=-log(u);}
                if(second.a[19]==1) {double u=rng_uniform(s);second.a[21]=-log(u);}
                stack[++top]=first;stack[++top]=second;
            }
            if(top+1>o->maxdepth)o->maxdepth=top+1;continue;
        }
        double cached_ev=-1.,arate=0.,S=0.,rate=0.,total=0.,G1=0.,brate=0.,bsoft=0.;
        bool annihilated=false,suspended=false;
        for(;;) {
            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];break;}
            int region=int(a[16]);
            if(region<0 || region>=1) {escaped+=a[6];if(a[19]==1) {++counts[13];positron_escape_rest+=2*M;}break;}
            if(a[6]<=.01+1e-14 && a[18]!=1) {
                p5_track_local(hs,150,a[2],a[6],.2,touch);p5_track_local(fine,600,a[2],a[6],.05,touch+2);
                dep+=a[6];a[6]=0.;break;
            }
            double ev=p5_max(.01,a[7]);
            if(ev!=cached_ev) {
                double q[4];p5_quantities(ev,t,q);brate=q[0];bsoft=q[2];double hardrate,hardstop;
                p5_hard(ev,t,&hardrate,&hardstop);S=p5_collision(ev,t)-hardstop+bsoft;rate=hardrate;
                p5_table_rates(ev,t,&total,&G1);
                if(a[19]==1)arate=p5_annihilation_rate(ev,t);cached_ev=ev;
            }
            double Sq=S*1.,rateq=rate*1.,G1q=G1*1.,brateq=brate*1.,bsoftq=bsoft*1.,arateq=arate*1.;
            double eb=a[8]/Sq,mb=G1q>0?a[12]/G1q:INFINITY,hb=rateq>0?a[15]/rateq:INFINITY;
            if(a[18]==1) {mb=INFINITY;hb=INFINITY;}
            double bb;int nr;p5_boundary(a[2],a[5],region,&bb,&nr);++counts[4];
            double rb=brateq>0 && a[18]!=1?a[22]/brateq:INFINITY;
            double step=p5_min(p5_min(p5_min(p5_min(p5_min(eb,mb),hb),bb),rb),a[6]/Sq),ab=INFINITY;
            if(a[19]==1 && a[18]!=1) {ab=arateq>0?a[21]/arateq:INFINITY;step=p5_min(step,ab);}
            if(!isfinite(step) || step<0) {failed=true;++counts[6];break;}
            double z0=a[2];a[0]+=step*a[3];a[1]+=step*a[4];a[2]+=step*a[5];
            double loss=step*Sq;a[6]-=loss;dep+=loss;radiation+=step*bsoftq;
            if(a[18]!=1) {expected_n+=step*brateq;a[22]=p5_max(0.,a[22]-step*brateq);}
            p5_track_segment(hs,150,z0,a[2],loss,.2,touch);p5_track_segment(fine,600,z0,a[2],loss,.05,touch+2);
            a[8]=p5_max(0.,a[8]-loss);a[12]=p5_max(0.,a[12]-step*G1q);a[15]=p5_max(0.,a[15]-step*rateq);
            if(a[19]==1 && a[18]!=1)a[21]=p5_max(0.,a[21]-step*arateq);++counts[0];
            if(bb<=step+1e-13) {a[2]=a[5]>0?30.:0.;a[16]=nr;++counts[3];continue;}
            if(eb<=step+1e-13) {
                if(a[18]==1) {
                    p5_track_local(hs,150,a[2],a[6],.2,touch);p5_track_local(fine,600,a[2],a[6],.05,touch+2);
                    dep+=a[6];a[6]=0.;break;
                }
                a[7]=p5_max(.01,a[7]-a[11]);
                if(a[7]<=.01+1e-14) {a[8]=a[9];a[9]=0.;a[10]=0.;a[18]=1.;continue;}
                double prev=a[9];a[11]=p5_min(.05*a[7],a[7]-.01);double u=rng_uniform(s);a[10]=u*a[11];
                a[9]=a[11]-a[10];a[8]=prev+a[10];continue;
            }
            if(mb<=step+1e-13) {
                double mu=p5_gs_mu_cached(ev,a[14],t,total,G1,s);
                double angle=rng_uniform(s),v[3];p5_rotate(a[3],a[4],a[5],mu,2*3.141592653589793*angle,v);
                for(int z=0;z<3;++z)a[3+z]=v[z];
                double u=rng_uniform(s),knew=p5_rule(ev);a[12]=a[13]+u*knew;a[13]=(1-u)*knew;a[14]=knew;
                ++counts[1];continue;
            }
            if(hb<=step+1e-13) {
                p5_energy_init(&state,s);double u=rng_uniform(s);a[15]=-log(u);if(a[6]<=2*.01)continue;
                double W=p5_sample_W(a[6],t,s),dp[3],ds[3],k=(a[6]+2*M)/a[6];
                double mp=sqrt((a[6]-W)/(a[6]-W+2*M)*k),ms=sqrt(W/(W+2*M)*k);
                double angle=rng_uniform(s),phi=2*3.141592653589793*angle;
                p5_rotate(a[3],a[4],a[5],mp,phi,dp);p5_rotate(a[3],a[4],a[5],ms,phi+3.141592653589793,ds);
                a[6]-=W;for(int z=0;z<3;++z)a[3+z]=dp[z];p5_energy_init(&state,s);++counts[2];
                if(top+2>=P5_STACK_CAP) {a[6]+=W;failed=true;++counts[6];break;}
                stack[++top]=state;++top;p5_initial(stack+top,W,a[0],a[1],a[2],ds[0],ds[1],ds[2],int(a[16]),int(a[20]),t,s);
                if(top+1>o->maxdepth)o->maxdepth=top+1;break;
            }
            if(rb<=step+1e-13) {
                p5_energy_init(&state,s);double T=a[6],u=rng_uniform(s);a[22]=-log(u);if(T<=.01)continue;
                int excess;double photon_k=p5_sample_k(T,t,s,&excess);
#if P5_TRANSPORT_FAULT == 1
                photon_k*=1+1e-12;
#endif
                o->brems[2]+=excess;if(photon_k>T)o->brems[3]+=1.;
                if(excess || photon_k>T) {failed=true;++counts[6];break;}
                double q[4];p5_quantities(T,t,q);hn+=1.;de+=photon_k-q[1]/q[0];radiation+=photon_k;
                o->brems[0]+=1.;o->brems[1]+=photon_k;
                double angle=rng_uniform(s),theta=M/(T+M),v[3];
                p5_rotate(a[3],a[4],a[5],cos(theta),2*3.141592653589793*angle,v);
                if(top+2>=P5_STACK_CAP) {failed=true;++counts[6];break;}
                a[6]-=photon_k;p5_energy_init(&state,s);stack[++top]=state;++top;
                p5_photon(stack+top,photon_k,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));
                if(top+1>o->maxdepth)o->maxdepth=top+1;suspended=true;break;
            }
            if(a[19]==1 && ab<=step+1e-13) {
                double T=a[6];if(T<=.01) {double u=rng_uniform(s);a[21]=-log(u);continue;}
                if(top+3>=P5_STACK_CAP) {failed=true;++counts[6];break;}
                double g[8];p5_flight_photons(T,a[3],a[4],a[5],t,s,g);++top;
                p5_photon(stack+top,g[0],a[0],a[1],a[2],g[1],g[2],g[3],int(a[16]),int(a[20]));++top;
                p5_photon(stack+top,g[4],a[0],a[1],a[2],g[5],g[6],g[7],int(a[16]),int(a[20]));
                rest-=2*M;a[6]=0.;annihilated=true;++counts[14];if(a[17]==1)++counts[15];
                if(top+1>o->maxdepth)o->maxdepth=top+1;break;
            }
            p5_track_local(hs,150,a[2],a[6],.2,touch);p5_track_local(fine,600,a[2],a[6],.05,touch+2);
            dep+=a[6];a[6]=0.;break;
        }
        if(failed)break;
        if(!suspended && !annihilated && a[19]==1 && a[6]==0 && a[16]>=0 && a[16]<1) {
            if(top+3>=P5_STACK_CAP) {a[6]=0.;failed=true;++counts[6];break;}
            double v[3];p5_annihilation(s,v);rest-=2*M;++counts[12];++top;
            p5_photon(stack+top,M,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));++top;
            p5_photon(stack+top,M,a[0],a[1],a[2],-v[0],-v[1],-v[2],int(a[16]),int(a[20]));
            if(top+1>o->maxdepth)o->maxdepth=top+1;
        }
    }
    double unresolved=0.;if(failed) {unresolved=state.a[6];for(int z=0;z<=top;++z)unresolved+=stack[z].a[6];}
    double residual=source_energy-dep-escaped-unresolved-rest;
    double coarse_error=fabs(p5_touched_sum(hs,touch)-dep),fine_error=fabs(p5_touched_sum(fine,touch+2)-dep);
    double dn=hn-expected_n;
    o->brems[4]+=dn;o->brems[5]+=dn*dn;o->brems[6]+=de;o->brems[7]+=de*de;o->brems[8]+=radiation;o->brems[9]+=radiation*radiation;
    o->maxima[0]=p5_max(o->maxima[0],fabs(residual)/source_energy);
    o->maxima[1]=p5_max(o->maxima[1],coarse_error/source_energy);o->maxima[2]=p5_max(o->maxima[2],fine_error/source_energy);
    if(fabs(residual)>source_energy*1e-9)++o->violations[0];if(coarse_error>source_energy*1e-9)++o->violations[1];if(fine_error>source_energy*1e-9)++o->violations[2];
    o->totals[0]+=source_energy;o->totals[1]+=dep;o->totals[2]+=escaped;o->totals[3]+=unresolved;o->totals[4]+=rest;o->totals[5]+=positron_escape_rest;
    for(int z=touch[0];z<=touch[1];++z)o->batches[batch*150+z]+=hs[z];
    for(int z=touch[2];z<=touch[3];++z)o->fine_batches[batch*600+z]+=fine[z];
    o->step_batches[batch]+=steps;
}
#endif
