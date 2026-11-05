#if defined(G5_PHOTON_LOCAL)
#include "g1_common.h"
#endif
// Shared A4u elemental layout. All energies in this section are keV.
#ifndef G5_PHOTON_TABLE_H
#define G5_PHOTON_TABLE_H
#ifndef G5_PHOTON_FAULT
#define G5_PHOTON_FAULT 0
#endif
// Photon transport cutoff of the electron-transport specialization, in MeV.
// USER DECISION 2026-11-04, which CHANGES the approved plan: S2 had deliberately
// lowered this to 5 keV under D3 ("線の下限は 5 keV にそろえる"), and D15 migrated
// the legacy 7 keV contracts to 4 keV for it. Raised back to 10 keV so that both
// codes kill photons at the same energy: EGS5 runs AP=0.01 MeV in every stored MV
// and low-energy reference. Recorded in the plan's out-of-scope table.
#ifndef G5_ELECTRON_PHOTON_CUT
#define G5_ELECTRON_PHOTON_CUT .01
#endif
P5_FN double g5_interp(double x,const double *a,const double *v,int n,int stride=1,int column=0,bool logarithmic=false) {
    int lo=0,hi=n-1;if(x<=a[0])return v[column];if(x>=a[hi])return v[hi*stride+column];
    while(hi-lo>1){int mid=(hi+lo)/2;if(a[mid]<=x)lo=mid;else hi=mid;}
    double u=logarithmic?log(x/a[lo])/log(a[hi]/a[lo]):(x-a[lo])/(a[hi]-a[lo]);
    double l=v[lo*stride+column],r=v[hi*stride+column];
    return logarithmic && l>0 && r>0?exp(log(l)+u*log(r/l)):l+u*(r-l);
}
P5_FN const double *g5_element(const double *p,int i){return p+2+17*i;}
P5_FN const double *g5_field(const double *p,int i,int f){return p+int(g5_element(p,i)[4+f]);}
P5_FN double g5_value(const double *p,int i,int field,double e){const double *d=g5_element(p,i);return g5_interp(e,g5_field(p,i,0),g5_field(p,i,field),int(d[2]),1,0,true);}
P5_FN double g5_total(const double *p,double e,int channel){
    const double *tail=p+int(p[1]);int nz=int(p[0]);double sum=0.;
    for(int i=0;i<nz;++i){const double *d=g5_element(p,i);double hi=tail[1+3*i+channel];
        if(e>hi && int(tail[0])==0)return NAN;
        sum+=d[1]*g5_value(p,i,channel+1,e);
    }return sum;
}
P5_FN int g5_choose_element(const double *p,double e,int field,RngState *rng){
    double total=0.;for(int i=0;i<int(p[0]);++i)total+=g5_element(p,i)[1]*g5_value(p,i,field,e);
    double r=rng_uniform(rng)*total;int chosen=int(p[0])-1;
    for(int i=0;i<int(p[0]);++i){r-=g5_element(p,i)[1]*g5_value(p,i,field,e);if(r<=0){chosen=i;break;}}return chosen;
}
P5_FN double g5_bound(const double *p,int element,double e,double mu){
    double q=e*sqrt((1-mu)/2)/12.3984193;
    return g5_interp(q,g5_field(p,element,9),g5_field(p,element,10),2000)/g5_element(p,element)[0];
}
// Local-deposition specialization: preserve its KN proposal and random order.
// The bound-electron factor and elemental tables are shared with p5_compton.
P5_FN void g5_local_compton(const double *pt,double e,RngState *rng,double *sample){
    int el=g5_choose_element(pt,e,2,rng);double alpha=e/511.,emin=1/(1+2*alpha),ep,co;
    for(;;){ep=emin+rng_uniform(rng)*(1-emin);co=1-(1/ep-1)/alpha;
        if(rng_uniform(rng)*(1/emin+emin)>1/ep+ep-(1-co*co))continue;
        if(rng_uniform(rng)<=g5_bound(pt,el,e,co))break;}
    sample[0]=ep;sample[1]=co;
}
P5_FN double g5_rayleigh(const double *p,double e,RngState *rng){
    int el=g5_choose_element(p,e,3,rng);double xmax=(e/12.3984193)*(e/12.3984193);
    const double *x=g5_field(p,el,11),*a=g5_field(p,el,12);
    double cut=g5_interp(xmax,x,a,2000);
    for(;;){double v=g5_interp(rng_uniform(rng)*cut,a,x,2000),mu=1-2*v/xmax;
        if(rng_uniform(rng)<=(1+mu*mu)/2)return mu;}
}
P5_FN int g5_shell(const double *p,int el,double e,RngState *rng){
    const double *d=g5_element(p,el),*edges=g5_field(p,el,5);double prob[9],sum=0.;
    for(int sh=0;sh<9;++sh){prob[sh]=e>edges[sh]?g5_interp(e,g5_field(p,el,0),g5_field(p,el,4),int(d[2]),9,sh):0.;sum+=prob[sh];}
    double r=rng_uniform(rng)*sum;int last=-1;
    for(int sh=0;sh<9;++sh)if(prob[sh]>0){last=sh;r-=prob[sh];if(r<=0)return sh;}return last;
}
// CK is an unconditional competing probability, not multiplied by 1-omega.
// Nonradiative termination/Auger energy remains local; outer child vacancies
// beyond M5 and all photons below 5 keV are deposited locally.
P5_FN int g5_relax(const double *p,int el,int shell,RngState *rng,bool enabled,double *out,double *local,long long *trace,int *events){
    const double *d=g5_element(p,el),*edges=g5_field(p,el,5),*yields=g5_field(p,el,6),*ck=g5_field(p,el,7),*lines=g5_field(p,el,8);
    int used=0;double emitted=0.;*events=0;for(int j=0;j<27;++j)trace[j]=-1;
    int sh=shell;
    for(int depth=0;enabled && sh<9 && depth<9;++depth){
        int current=sh;double u=rng_uniform(rng),omega=yields[current];sh=9;
        trace[3*(*events)]=current;++*events;
        if(u<omega){double r=rng_uniform(rng);int selected=-1;
            for(int j=0;j<int(d[3]);++j)if(int(lines[4*j])==current){selected=j;r-=lines[4*j+2];if(r<=0)break;}
            if(selected<0)break;
            double e=lines[4*selected+1];sh=int(lines[4*selected+3]);trace[3*(*events-1)+1]=selected;
            if(e>=5){out[4*used]=e;double mu=2*rng_uniform(rng)-1,phi=2*3.141592653589793*rng_uniform(rng),si=sqrt(1-mu*mu);
                out[4*used+1]=si*cos(phi);out[4*used+2]=si*sin(phi);out[4*used+3]=mu;++used;emitted+=e;}
        }else {
            u-=omega;
            for(int dest=current+1;dest<9;++dest){double f=ck[9*current+dest];
                if(G5_PHOTON_FAULT==1)f=0.;if(G5_PHOTON_FAULT==2 && current==1 && dest==3)f=0.;
                if(u<f){sh=dest;break;}u-=f;}
        }
        trace[3*(*events-1)+2]=sh;
    }
    *local=edges[shell]-emitted;return *local>=-1e-9?used:-1;
}
#endif
// A4u S2: one photon model; frozen legacy branches are verification-only.
// Each section is included only after its table types and helpers are available.
#if defined(G5_PHOTON_LOCAL) && !defined(G5_PHOTON_LOCAL_H)
#define G5_PHOTON_LOCAL_H
#include "g1_common.h"
#ifndef G5_TEST_LEGACY_LOCAL
#define G5_TEST_LEGACY_LOCAL 0
#endif
// Scalar ports of chatcarlo/kernel.py:285-536. Preserve evaluation order.
P5_FN double g1_lerp(const double *x_grid,const double *y_grid,int n,double x) {
    if(x<=x_grid[0])return y_grid[0];
    if(x>=x_grid[n-1])return y_grid[n-1];
    int lo=0,hi=n-1;
    while(hi-lo>1) {int mid=(lo+hi)/2;if(x_grid[mid]<=x)lo=mid;else hi=mid;}
    double x0=x_grid[lo],x1=x_grid[hi],y0=y_grid[lo],y1=y_grid[hi];
    return y0+(x-x0)/(x1-x0)*(y1-y0);
}
P5_FN void g1_index(const double *log_e_row,double step,int n,double log_eq,int *index,double *fraction) {
    int hi=n-1,idx;
    if(step>0.) {
        idx=int((log_eq-log_e_row[0])/step)+1;
        if(idx<1)idx=1;if(idx>hi)idx=hi;
        if(log_eq<=log_e_row[idx-1]){--idx;if(idx<1)idx=1;}
        if(log_e_row[idx]<log_eq){++idx;if(idx>hi)idx=hi;}
    } else {
        int lo_i=0,hi_i=n;
        while(lo_i<hi_i){int mid=(lo_i+hi_i)/2;if(log_e_row[mid]<log_eq)lo_i=mid+1;else hi_i=mid;}
        idx=lo_i;if(idx<1)idx=1;if(idx>hi)idx=hi;
    }
    double x0=log_e_row[idx-1],x1=log_e_row[idx];
    *index=idx;*fraction=(log_eq-x0)/(x1-x0);
}
P5_FN int g1_select(int n,const double *fracs,const double *cs,double total,RngState *rng) {
    double target=rng_uniform(rng)*total,cum=0.;
    for(int i=0;i<n;++i){cum+=fracs[i]*cs[i];if(target<=cum)return i;}
    return n-1;
}
P5_FN void g1_scatter(double dx,double dy,double dz,double cos_theta,RngState *rng,double *out) {
    double s=1.-cos_theta*cos_theta;
    double sin_theta=sqrt(s>0.?s:0.);
    double phi=rng_uniform(rng)*2.0*3.141592653589793;
    double upx,upy,upz;
    if(fabs(dz)<.999){upx=0.;upy=0.;upz=1.;}else{upx=1.;upy=0.;upz=0.;}
    double ux=upy*dz-upz*dy,uy=upz*dx-upx*dz,uz=upx*dy-upy*dx;
    double un=sqrt(ux*ux+uy*uy+uz*uz);
    ux/=un;uy/=un;uz/=un;
    double vx=dy*uz-dz*uy,vy=dz*ux-dx*uz,vz=dx*uy-dy*ux;
    double cphi=cos(phi),sphi=sin(phi);
    double nx=sin_theta*cphi*ux+sin_theta*sphi*vx+cos_theta*dx;
    double ny=sin_theta*cphi*uy+sin_theta*sphi*vy+cos_theta*dy;
    double nz=sin_theta*cphi*uz+sin_theta*sphi*vz+cos_theta*dz;
    double nn=sqrt(nx*nx+ny*ny+nz*nz);
    out[0]=nx/nn;out[1]=ny/nn;out[2]=nz/nn;
}
P5_FN void g1_isotropic(RngState *rng,double *out) {
    double cos_theta=rng_uniform(rng)*2.0-1.0;
    double s=1.0-cos_theta*cos_theta;
    double sin_theta=sqrt(s>0.?s:0.);
    double phi=rng_uniform(rng)*2.0*3.141592653589793;
    out[0]=sin_theta*cos(phi);out[1]=sin_theta*sin(phi);out[2]=cos_theta;
}

// Field order follows SceneMaterialTables (bake_scene_materials).
enum {NE,ZS,FRACS,LOGE,STEP,NG,PHOTO,COMPT,RAYL,IQ,IS,RX,RA,KE,KO,KLE,KLP,NL,KF,DENS};
struct G1Tables {const double *data;const long long *offsets;int me,ng,nm;};
P5_FN const double *g1_row(const G1Tables *t,int field,int m=0,int e=0,int width=1) {
    return t->data+t->offsets[field]+(m*t->me+e)*width;
}
P5_FN void g1_mu(double energy,int m,const G1Tables *t,double *p,double *c,double *r,int *idx,double *frac,double *out) {
    int n=int(g1_row(t,NE)[m]);double log_eq=log(energy);
    double tp=0.,tc=0.,tr=0.;
    for(int i=0;i<n;++i){
        g1_index(g1_row(t,LOGE,m,i,t->ng),g1_row(t,STEP,m,i)[0],int(g1_row(t,NG,m,i)[0]),log_eq,idx+i,frac+i);
        int j=idx[i];double f=frac[i];
        const double *pp=g1_row(t,PHOTO,m,i,t->ng),*cc=g1_row(t,COMPT,m,i,t->ng),*rr=g1_row(t,RAYL,m,i,t->ng);
        p[i]=exp(pp[j-1]+f*(pp[j]-pp[j-1]));
        c[i]=exp(cc[j-1]+f*(cc[j]-cc[j-1]));
        r[i]=exp(rr[j-1]+f*(rr[j]-rr[j-1]));
        double fr=g1_row(t,FRACS,m,i)[0];tp+=fr*p[i];tc+=fr*c[i];tr+=fr*r[i];
    }
    out[0]=(tp+tc+tr)*g1_row(t,DENS)[m];out[1]=tp;out[2]=tc;out[3]=tr;
}
P5_FN double g1_fluorescence(int m,int ei,double energy,int idx,double frac,const G1Tables *t,RngState *rng) {
    double edge=g1_row(t,KE,m,ei)[0];if(energy<=edge)return 0.;
    double omega=g1_row(t,KO,m,ei)[0];if(omega<=0.)return 0.;
    const double *kf=g1_row(t,KF,m,ei,t->ng);
    double k_frac=kf[idx-1]+frac*(kf[idx]-kf[idx-1]);
    if(rng_uniform(rng)>=k_frac)return 0.;
    if(rng_uniform(rng)>=omega)return 0.;
    int n=int(g1_row(t,NL,m,ei)[0]);double r=rng_uniform(rng),cum=0.;int chosen=n-1;
    for(int j=0;j<n;++j){cum+=g1_row(t,KLP,m,ei,8)[j];if(r<=cum){chosen=j;break;}}
    double line=g1_row(t,KLE,m,ei,8)[chosen];
#if P5_FAULT != 3
    if(line<5.)return 0.;
#endif
    return line;
}
P5_FN void g1_compton(int m,double energy,const G1Tables *t,const double *c,double total,RngState *rng,double *out) {
    int ei=g1_select(int(g1_row(t,NE)[m]),g1_row(t,FRACS,m),c,total,rng);
    double z=g1_row(t,ZS,m,ei)[0],alpha=energy/511.;
    double eps_min=1./(1.+2.*alpha),envelope=1./eps_min+eps_min;
    while(true){
        double xi1=rng_uniform(rng),xi2=rng_uniform(rng);
        double ep=eps_min+xi1*(1.-eps_min);
        double co=1.-(1./ep-1.)/alpha;
        double sin2=1.-co*co,g=1./ep+ep-sin2;
        if(xi2*envelope>g)continue;
        double cc=co;if(cc>1.)cc=1.;if(cc< -1.)cc=-1.;
        double theta=acos(cc),q=energy*sin(theta/2.)/12.3984193;
        double s_over_z=g1_lerp(g1_row(t,IQ,m,ei,2000),g1_row(t,IS,m,ei,2000),2000,q)/z;
        double xi3=rng_uniform(rng);
        if(P5_FAULT==2 || xi3<=s_over_z){out[0]=ep;out[1]=co;out[2]=ei;return;}
    }
}
P5_FN void g1_rayleigh(int m,double energy,const G1Tables *t,const double *r,double total,RngState *rng,double *out) {
    int ei=g1_select(int(g1_row(t,NE)[m]),g1_row(t,FRACS,m),r,total,rng);
    double ratio=energy/12.3984193,x_max=ratio*ratio;
    while(true){
        double cut=g1_lerp(g1_row(t,RX,m,ei,2000),g1_row(t,RA,m,ei,2000),2000,x_max);
        double xi1=rng_uniform(rng);
        double x=g1_lerp(g1_row(t,RA,m,ei,2000),g1_row(t,RX,m,ei,2000),2000,xi1*cut);
        if(x>x_max)x=x_max;
        double co=1.-2.*x/x_max;if(co>1.)co=1.;if(co< -1.)co=-1.;
        double xi2=rng_uniform(rng);
        if(P5_FAULT==1 || xi2<=(1.+co*co)/2.){out[0]=co;out[1]=ei;return;}
    }
}
P5_FN const double *g1_photons(const G1Tables *t,int m){const double *p=t->data+t->offsets[20];return p+int(p[m]);}
#endif

#if defined(G5_PHOTON_P5) && !defined(G5_PHOTON_P5_H)
#define G5_PHOTON_P5_H
#ifndef G5_TEST_LEGACY_ELECTRON
#define G5_TEST_LEGACY_ELECTRON 0
#endif
P5_FN void p5_rotate(double,double,double,double,double,double *);
#if G5_TEST_LEGACY_ELECTRON
P5_FN double p5_kn_total(double E,const P5Tables *t) {
    double x=E/t->constants[0],L=log1p(2*x),one=1+2*x;
    double RE=t->constants[3],NE=t->constants[4];
    return 2*3.141592653589793*(RE*RE)*NE*((1+x)/(x*x)
        *(2*(1+x)/(1+2*x)-L/x)+L/(2*x)-(1+3*x)/(one*one));
}
#endif

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
#if G5_TEST_LEGACY_ELECTRON
    c[0]=p5_kn_total(E,t);
#else
    c[0]=g5_total(t->photon,E*1000,1);
#endif
    c[1]=p5_log_interp(E,t->pes,t->water+2,t->np,t->watercols);
    c[2]=p5_pair_interp(E,t->pes,t->water+3,t->np,t->watercols,t->thresholds[0]);
    c[3]=p5_pair_interp(E,t->pes,t->water+4,t->np,t->watercols,t->thresholds[1]);
    c[4]=0.;
#if !G5_TEST_LEGACY_ELECTRON
    c[1]=g5_total(t->photon,E*1000,0);c[4]=g5_total(t->photon,E*1000,2);
#endif
}

P5_FN void p5_compton(double E,double ux,double uy,double uz,const P5Tables *t,RngState *s,double *out) {
    double M=t->constants[0],qmin=1/(1+2*E/M),a=log(1/qmin),b=(1-qmin*qmin)/2,q,mu;
    int element=0;
#if !G5_TEST_LEGACY_ELECTRON
    element=g5_choose_element(t->photon,E*1000,2,s);
#endif
    for(;;) {
        double choose=rng_uniform(s);
        if(choose*(a+b)<a) {double u=rng_uniform(s);q=exp(-u*a);}
        else {double u=rng_uniform(s);q=sqrt(qmin*qmin+u*(1-qmin*qmin));}
        mu=1-M/E*(1/q-1);double sin2=p5_max(0.,1-mu*mu);
        double accept=rng_uniform(s);if(accept*(q+1/q)>q+1/q-sin2)continue;
#if !G5_TEST_LEGACY_ELECTRON
        if(rng_uniform(s)>g5_bound(t->photon,element,E*1000,mu))continue;
#endif
        break;
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
#endif

#if defined(G5_PHOTON_P8F) && !defined(G5_PHOTON_P8F_H)
#define G5_PHOTON_P8F_H
P5_FN void p8f_coefficients(double E,const P8Material &m,double *c) {
#if G5_TEST_LEGACY_ELECTRON
    double x=E/m.p.constants[0],L=log1p(2*x),RE=m.p.constants[3],one=1+2*x;
    c[0]=2*3.141592653589793*(RE*RE)*(*m.ne)*((1+x)/(x*x)*(2*(1+x)/(1+2*x)-L/x)+L/(2*x)-(1+3*x)/(one*one));
#endif
    double pe=p8f_log_interp(E,m.photo_E,m.photo,m.nphoto);int i=p8f_upper(E,m.photo_E,m.nphoto)-1;
    if(i>=0 && i<m.nphoto-1) {
        bool near=(i>0 && m.photo_E[i]==m.photo_E[i-1]) || (i+2<m.nphoto && m.photo_E[i+1]==m.photo_E[i+2]);
        if(near) {
            int left=i,right=i+1;
#if P8F_FAULT == 4
            if(i>0 && m.photo_E[i]==m.photo_E[i-1])--left;
            if(i+2<m.nphoto && m.photo_E[i+1]==m.photo_E[i+2])++right;
#endif
            double u=log(E/m.photo_E[i])/log(m.photo_E[i+1]/m.photo_E[i]);
            pe=exp(log(m.photo[left])+u*log(m.photo[right]/m.photo[left]));
        }
    }
    c[1]=pe;c[2]=p5_pair_interp(E,m.pair_E,m.pair_n,m.npair,1,m.thresholds[0]);
    c[3]=p5_pair_interp(E,m.pair_E,m.pair_e,m.npair,1,m.thresholds[1]);
    c[4]=0.;
#if !G5_TEST_LEGACY_ELECTRON
    c[0]=g5_total(m.p.photon,E*1000,1);c[1]=g5_total(m.p.photon,E*1000,0);c[4]=g5_total(m.p.photon,E*1000,2);
#endif
}

P5_FN void p8f_shell_distribution(double E,const P8Material &m,double *xs) {
    double l=log(E*1000),total=0.;
    for(int i=0;i<m.nz;++i) {
        for(int s=0;s<5;++s) {const double *q=m.pm+(i*5+s)*4;double v=0.;
            if(E>m.bindings[i*5+s] && (s==0 || q[0]!=0))v=exp(q[0]+q[1]*l+q[2]*l*l+q[3]*l*l*l);
            if(E<=m.bindings[i*5+4])v=0.;xs[i*5+s]=v*m.shell_nj[i];}
        if(E<=m.bindings[i*5+3]){double row=0.;for(int s=0;s<5;++s)row+=xs[i*5+s];for(int s=0;s<5;++s)xs[i*5+s]=0.;xs[i*5+4]=row;}
    }
    for(int j=0;j<m.nz*5;++j)total+=xs[j];if(total>0)for(int j=0;j<m.nz*5;++j)xs[j]/=total;
}

P5_FN void p8f_choose_shell(const double *xs,int nz,double r,int *element,int *shell){
    *element=-1;*shell=-1;for(int i=0;i<nz;++i)for(int j=0;j<5;++j)if(xs[i*5+j]>0){*element=i;*shell=j;if(r<xs[i*5+j])return;r-=xs[i*5+j];}
}

P5_FN void p8f_select_shell(double E,const P8Material &m,RngState *s,int *element,int *shell) {
    double xs[150];p8f_shell_distribution(E,m,xs);double total=0.;for(int j=0;j<m.nz*5;++j)total+=xs[j];
    *element=-1;*shell=-1;if(total==0)return;double r=rng_uniform(s);
    p8f_choose_shell(xs,m.nz,r,element,shell);
}

P5_FN int p8f_relax(int element,int shell,const P8Material &m,RngState *s,bool enabled,double *out,double *local,long long *trace,int *events) {
    int queue[8],top=0,used=0;queue[0]=shell;double emitted=0.;*events=0;
    for(int j=0;j<24;++j)trace[j]=-1;for(int j=0;j<32;++j)out[j]=0.;
    while(top>=0){int sh=queue[top--];if(sh>=4 || !enabled)continue;
        double u=rng_uniform(s);if(u>=m.yields[element*4+sh]){trace[3*(*events)]=sh;++*events;continue;}
        double r=rng_uniform(s);int j=0;const double *lines=m.lines+((element*4+sh)*m.nline)*4;
        while(j<int(m.line_counts[element*4+sh])-1 && r>=lines[j*4+1]){r-=lines[j*4+1];++j;}
        double E=lines[j*4];int child=int(lines[j*4+2]);trace[3*(*events)]=sh;trace[3*(*events)+1]=j;trace[3*(*events)+2]=child;++*events;
        if(E>=.01){double v[3];p5_annihilation(s,v);out[4*used]=E;for(int k=0;k<3;++k)out[4*used+1+k]=v[k];++used;emitted+=E;}
        if(child<4)queue[++top]=child;
    }
    *local=m.bindings[element*5+shell]-emitted;return used;
}
#endif

#if defined(G5_PHOTON_P8G) && !defined(G5_PHOTON_P8G_H)
#define G5_PHOTON_P8G_H
P5_FN void p8g_coefficients(double E,const P8gMaterial &m,double *c) {
#if G5_TEST_LEGACY_ELECTRON
    double x=E/m.p.constants[0],L=log1p(2*x),RE=m.p.constants[3],one=1+2*x;
    c[0]=2*3.141592653589793*(RE*RE)*(*m.ne)*((1+x)/(x*x)*(2*(1+x)/(1+2*x)-L/x)+L/(2*x)-(1+3*x)/(one*one));
#endif
    double pe=p8g_log_interp(E,m.photo_E,m.photo,m.nphoto,m,10,11);int i=p8g_upper(E,m.photo_E,m.nphoto)-1;
    if(i>=0 && i<m.nphoto-1) {
        bool near=(i>0 && m.photo_E[i]==m.photo_E[i-1]) || (i+2<m.nphoto && m.photo_E[i+1]==m.photo_E[i+2]);
        if(near) {
            int left=i,right=i+1;
#if P8G_FAULT == 4
            if(i>0 && m.photo_E[i]==m.photo_E[i-1])--left;
            if(i+2<m.nphoto && m.photo_E[i+1]==m.photo_E[i+2])++right;
#endif
            double u=log(E/m.photo_E[i])/p8g_read(m,12,i);
            pe=exp(p8g_read(m,14,left)+u*p8g_read(m,13,left));
        }
    }
    c[1]=pe;c[2]=p8g_pair_interp(E,m.pair_E,m.pair_n,m.npair,1,m.thresholds[0],m,6);
    c[3]=p8g_pair_interp(E,m.pair_E,m.pair_e,m.npair,1,m.thresholds[1],m,7);
    c[4]=0.;
#if !G5_TEST_LEGACY_ELECTRON
    c[0]=g5_total(m.p.photon,E*1000,1);
    c[1]=g5_total(m.p.photon,E*1000,0);
    c[4]=g5_total(m.p.photon,E*1000,2);
#endif
}

P5_FN void p8g_shell_distribution(double E,const P8gMaterial &m,double *xs) {
    double l=log(E*1000),total=0.;
    for(int i=0;i<m.nz;++i) {
        for(int s=0;s<5;++s) {const double *q=m.pm+(i*5+s)*4;double v=0.;
            if(E>m.bindings[i*5+s] && (s==0 || q[0]!=0))v=exp(q[0]+q[1]*l+q[2]*l*l+q[3]*l*l*l);
            if(E<=m.bindings[i*5+4])v=0.;xs[i*5+s]=v*m.shell_nj[i];}
        if(E<=m.bindings[i*5+3]){double row=0.;for(int s=0;s<5;++s)row+=xs[i*5+s];for(int s=0;s<5;++s)xs[i*5+s]=0.;xs[i*5+4]=row;}
    }
    for(int j=0;j<m.nz*5;++j)total+=xs[j];if(total>0)for(int j=0;j<m.nz*5;++j)xs[j]/=total;
}

P5_FN void p8g_choose_shell(const double *xs,int nz,double r,int *element,int *shell){
    *element=-1;*shell=-1;for(int i=0;i<nz;++i)for(int j=0;j<5;++j)if(xs[i*5+j]>0){*element=i;*shell=j;if(r<xs[i*5+j])return;r-=xs[i*5+j];}
}

P5_FN void p8g_select_shell(double E,const P8gMaterial &m,RngState *s,int *element,int *shell) {
#if !G5_TEST_LEGACY_ELECTRON
    *element=g5_choose_element(m.p.photon,E*1000,1,s);*shell=g5_shell(m.p.photon,*element,E*1000,s);return;
#endif
    double xs[150];p8g_shell_distribution(E,m,xs);double total=0.;for(int j=0;j<m.nz*5;++j)total+=xs[j];
    *element=-1;*shell=-1;if(total==0)return;double r=rng_uniform(s);
    p8g_choose_shell(xs,m.nz,r,element,shell);
}

P5_FN int p8g_relax(int element,int shell,const P8gMaterial &m,RngState *s,bool enabled,double *out,double *local,long long *trace,int *events) {
#if !G5_TEST_LEGACY_ELECTRON
    double kev[36],budget;long long tr[27];int n=g5_relax(m.p.photon,element,shell,s,enabled,kev,&budget,tr,events);
    if(n<0)return -1;
    for(int j=0;j<4*n;++j)out[j]=(j%4==0)?kev[j]*.001:kev[j];
    for(int j=0;j<24;++j)trace[j]=tr[j];*local=budget*.001;return n;
#endif
    int queue[8],top=0,used=0;queue[0]=shell;double emitted=0.;*events=0;
    for(int j=0;j<24;++j)trace[j]=-1;for(int j=0;j<32;++j)out[j]=0.;
    while(top>=0){int sh=queue[top--];if(sh>=4 || !enabled)continue;
        double u=rng_uniform(s);if(u>=m.yields[element*4+sh]){trace[3*(*events)]=sh;++*events;continue;}
        double r=rng_uniform(s);int j=0;const double *lines=m.lines+((element*4+sh)*m.nline)*4;
        while(j<int(m.line_counts[element*4+sh])-1 && r>=lines[j*4+1]){r-=lines[j*4+1];++j;}
        double E=lines[j*4];int child=int(lines[j*4+2]);trace[3*(*events)]=sh;trace[3*(*events)+1]=j;trace[3*(*events)+2]=child;++*events;
        if(E>=.01){double v[3];p5_annihilation(s,v);out[4*used]=E;for(int k=0;k<3;++k)out[4*used+1+k]=v[k];++used;emitted+=E;}
        if(child<4)queue[++top]=child;
    }
    *local=m.bindings[element*5+shell]-emitted;return used;
}
#endif

#if defined(G5_PHOTON_P8G) && !defined(G5_BINDING_H)
#define G5_BINDING_H
P5_FN double g5_binding(const P8gMaterial &m,int element,int shell){
#if G5_TEST_LEGACY_ELECTRON
return m.bindings[element*5+shell];
#else
return g5_field(m.p.photon,element,5)[shell]*.001;
#endif
}
#endif

#if defined(G5_PHOTON_P8G) && !defined(G5_LINE_ORIGIN_H)
#define G5_LINE_ORIGIN_H
P5_FN bool g5_k_line(const P8gMaterial &m,int el,double mev){
    const double *d=g5_element(m.p.photon,el),*lines=g5_field(m.p.photon,el,8);
    for(int j=0;j<int(d[3]);++j)if(lines[4*j]==0 && lines[4*j+1]*.001==mev)return true;return false;
}
#endif
