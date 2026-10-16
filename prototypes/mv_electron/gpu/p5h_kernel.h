#ifndef P5H_KERNEL_H
#define P5H_KERNEL_H
// Frozen helpers are reused read-only; history() is never called by the new driver.
#include "p5h_lookup.h"

P5_FN double p5h_interp(double x,const double *a,const double *v,int n,const P5hTables *t,int site) {
    // Numba np_interp_impl_inner: slope first, multiply, then add.
    if(x<a[0])return v[0];if(x>a[n-1])return v[n-1];
    int j=p5h_find(x,a,n,1,0,0,t,3,site);
    if(j<n && x==a[j])return v[j];
    --j;
    double slope=(v[j+1]-v[j])/(a[j+1]-a[j]);
    return slope*(x-a[j])+v[j];
}

P5_FN void p5h_quantities(double t,const double *e,const double *v,int n,
    const double *le,const double *er,int m,double cut,int disabled,
    double *rate,double *discrete,double *soft,double *env,int *idx,double *raw,const P5hTables *ctx) {
    *idx=-1;for(int j=0;j<12;++j)raw[j]=0.;
    double rad=p5h_interp(log(t>cut?t:cut),le,er,m,ctx,1);
    *rate=0.;*discrete=0.;*soft=rad;*env=0.;
    if(disabled || t<=cut)return;
    int i=p5h_find(t,e,n,1,0,0,ctx,1,2)-1;if(i<0)i=0;if(i>n-2)i=n-2;*idx=i;
    raw[0]=e[i];raw[1]=e[i+1];
    for(int j=0;j<5;++j){raw[2+j]=v[5*i+j];raw[7+j]=v[5*(i+1)+j];}
    double u=(log(t)-log(e[i]))/(log(e[i+1])-log(e[i]));
    if(i==0) {
        *rate=u*v[5];*discrete=u*v[6];*soft=rad-*discrete;
        *env=u*v[8]+(1-u)*v[3];return;
    }
    *rate=exp((1-u)*log(v[5*i])+u*log(v[5*(i+1)]));
    *soft=exp((1-u)*log(v[5*i+2])+u*log(v[5*(i+1)+2]));
    *discrete=exp((1-u)*log(v[5*i+1])+u*log(v[5*(i+1)+1]));
    *env=(1-u)*v[5*i+3]+u*v[5*(i+1)+3];
}

P5_FN double p5h_collision(double T,const P5hTables *t) {
    double M=t->constants[0],C=t->constants[1],I=t->constants[2];
    double tau=T/M,g=1+tau,b2=1-1/(g*g),d=tau/2;
    double F=-1-b2+log((tau-d)*d)+tau/(tau-d)
        +(d*d/2+(2*tau+1)*log1p(-d/tau))/(g*g);
    double density=p5h_interp(log(T),t->loget,t->ed,t->ne,t,0);
    double im=I/M;
    return C/b2*(log(2*(tau+2)/(im*im))+F-density);
}

P5_FN double p5h_log_interp(double E,const double *es,const double *vs,int n,int stride,const P5hTables *t) {
    // np.interp searches the LOG grid, including ties after rounding log().
    double x=log(E);
    if(x<log(es[0]))return exp(log(vs[0]));
    if(x>log(es[n-1]))return exp(log(vs[(n-1)*stride]));
    int lo=p5h_find(x,es,n,1,1,0,t,4,3);
    int j=lo;
    if(j<n && x==log(es[j]))return exp(log(vs[j*stride]));
    --j;
    double y0=log(vs[j*stride]),y1=log(vs[(j+1)*stride]);
    double slope=(y1-y0)/(log(es[j+1])-log(es[j]));
    return exp(slope*(x-log(es[j]))+y0);
}

P5_FN double p5h_pair_interp(double E,const double *es,const double *vs,int n,int stride,double th,const P5hTables *t,int pair) {
    if(E<=th)return 0.;
    int start=t->pair_start[pair];
    double x=log(E),y;
    if(x<log(es[start]))y=p5_pair_value(start,es,vs,stride,th);
    else if(x>log(es[n-1]))y=p5_pair_value(n-1,es,vs,stride,th);
    else {
        int lo=p5h_find(x,es,n,1,1,start,t,4,4+pair);
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

P5_FN void p5h_coefficients(double E,const P5hTables *t,double *c) {
    c[0]=p5_kn_total(E,t);
    c[1]=p5h_log_interp(E,t->pes,t->water+2,t->np,t->watercols,t);
    c[2]=p5h_pair_interp(E,t->pes,t->water+3,t->np,t->watercols,t->thresholds[0],t,0);
    c[3]=p5h_pair_interp(E,t->pes,t->water+4,t->np,t->watercols,t->thresholds[1],t,1);
}

P5_FN void p5h_table_rates(double T,const P5hTables *t,double *total,double *G1) {
    int shift=t->ratecols==3?1:0,n=shift?t->nrates:t->nt;
    int lo=p5h_find(T,shift?t->rates:t->ts,n,shift?t->ratecols:1,0,0,t,shift?2:5,6);
    int i=lo-1;if(i<0)i=0;if(i>n-2)i=n-2;
    double e0=shift?t->rates[i*t->ratecols]:t->ts[i];
    double e1=shift?t->rates[(i+1)*t->ratecols]:t->ts[i+1];
    double a=log(T/e0)/log(e1/e0);
    *total=exp((1-a)*log(t->rates[i*t->ratecols+shift])+a*log(t->rates[(i+1)*t->ratecols+shift]));
    *G1=exp((1-a)*log(t->rates[i*t->ratecols+shift+1])+a*log(t->rates[(i+1)*t->ratecols+shift+1]));
}

P5_FN void p5h_bracket(const double *grid,int n,double x,int *i,double *a,const P5hTables *t) {
    *i=p5h_find(x,grid,n,1,0,0,t,0,8)-1;if(*i<0)*i=0;if(*i>n-2)*i=n-2;
    *a=(x-grid[*i])/(grid[*i+1]-grid[*i]);
}

P5_FN double p5h_gs_mu_cached(double T,double K,const P5hTables *t,double total,double G1,RngState *s) {
    double p0=exp(-K/G1*total),zero=rng_uniform(s);
    if(zero<p0)return 1.;
    double u=rng_uniform(s);int i=p5h_find(T,t->ts,t->nt,1,0,0,t,5,7)-1;
    if(i<0)i=0;if(i>t->nt-2)i=t->nt-2;
    double a=log(T/t->ts[i])/log(t->ts[i+1]/t->ts[i]);int j,h;double b,c;
    p5_bracket(t->ks,t->nk,K,&j,&b);p5h_bracket(t->us,t->nu,u,&h,&c,t);
    int q00=(i*t->nk+j)*t->nu+h,q01=q00+t->nu;
    int q10=q00+t->nk*t->nu,q11=q10+t->nu;
    double y00=(1-c)*t->ys[q00]+c*t->ys[q00+1];
    double y01=(1-c)*t->ys[q01]+c*t->ys[q01+1];
    double y10=(1-c)*t->ys[q10]+c*t->ys[q10+1];
    double y11=(1-c)*t->ys[q11]+c*t->ys[q11+1];
    return 1-((1-a)*((1-b)*y00+b*y01)+a*((1-b)*y10+b*y11));
}

P5_FN void p5h_quantities(double T,const P5hTables *t,double *q) {
    int idx;double raw[12];
    p5h_quantities(T,t->brems_e,t->brems_v,t->nbrems,t->loget,
               t->er,t->ne,.01,0,q,q+1,q+2,q+3,&idx,raw,t);
}

P5_FN double p5h_sample_k(double T,const P5hTables *t,RngState *s,int *excess) {
    double q[4];p5h_quantities(T,t,q);double envelope=q[3],lo=log(.01),width=log(T/.01);
    *excess=0;
    for(;;) {
        double u=rng_uniform(s),k=p5_min(T,p5_max(.01,exp(lo+width*u)));
        double f=p5_shape(T,k,t);if(f>envelope)++*excess;
        double accept=rng_uniform(s);if(accept*envelope<=f)return k;
    }
}
enum P5hPhase {P5H_P=0,P5H_C=1,P5H_E=2,P5H_H=3,P5H_D=4};
struct P5hState {
    P5State stack[P5_STACK_CAP],state;
    double hs[150],fine[600];int touch[4],top,steps,phase,batch;
    double hn,expected_n,de,radiation,source_energy,dep,escaped,rest,positron_escape_rest;
    double cached_ev,arate,S,rate,total,G1,brate,bsoft;
    bool failed,annihilated,suspended;
};
// Trace: from,to,steps,calls,top,charge,energy,reason,hard_count,brems_count.
// reason bits: failed=1, suspended=2, annihilated=4, exterior=8. Exact integers, no RNG calls.
struct P5hTrace {double *data;int capacity,used;long long overflow;};
P5_FN void p5h_record(P5hTrace *tr,int from,const P5hState *q,const RngState *s,const P5Output *o) {
    if(!tr)return;
    if(tr->used<tr->capacity) {
        double *v=tr->data+10*tr->used;
        v[0]=from;v[1]=q->phase;v[2]=q->steps;v[3]=double(s->calls);v[4]=q->top;
        v[5]=q->state.a[19];v[6]=q->state.a[6];
        v[7]=int(q->failed)+2*int(q->suspended)+4*int(q->annihilated)
            +8*int(q->phase==P5H_E && (q->state.a[16]<0 || q->state.a[16]>=1));
        v[8]=double(o->counts[2]);v[9]=o->brems[0];
    } else ++tr->overflow;
    ++tr->used;
}
P5_FN void p5h_init(P5hState *q,const P5hTables *t,RngState *s,int batch0,P5Output *o,P5hTrace *tr) {
    *q=P5hState{};q->touch[0]=150;q->touch[1]=-1;q->touch[2]=600;q->touch[3]=-1;q->batch=batch0;
    P5State (&stack)[P5_STACK_CAP]=q->stack;P5State &state=q->state;double *a=state.a;
    double *hs=q->hs,*fine=q->fine;int *touch=q->touch;
    int &top=q->top,&steps=q->steps,&phase=q->phase;
    double M=t->constants[0];
    double &hn=q->hn;
    double &expected_n=q->expected_n;
    double &de=q->de;
    double &radiation=q->radiation;
    double &source_energy=q->source_energy;
    double &dep=q->dep;
    double &escaped=q->escaped;
    double &rest=q->rest;
    double &positron_escape_rest=q->positron_escape_rest;
    double &cached_ev=q->cached_ev;
    double &arate=q->arate;
    double &S=q->S;
    double &rate=q->rate;
    double &total=q->total;
    double &G1=q->G1;
    double &brate=q->brate;
    double &bsoft=q->bsoft;
    bool &failed=q->failed;
    bool &annihilated=q->annihilated;
    bool &suspended=q->suspended;
    long long *counts=o->counts;int batch=q->batch;
    double u1=rng_uniform(s);double u2=rng_uniform(s);int j=0;
    while(t->cum[j]<u1)++j;
    double width=t->edges[j+1]-t->edges[j];source_energy=t->edges[j]+u2*width;
    int cell=int((source_energy-t->edges[j])/width*10);if(cell>9)cell=9;++o->nspec[j*10+cell];
    p5_photon(stack,source_energy,0.,0.,0.,0.,0.,1.,0,0);
    top=0; phase=P5H_P;
    p5h_record(tr,-1,q,s,o);
}
P5_FN void p5h_particle(P5hState *q,const P5hTables *t,RngState *s,int maxsteps,P5Output *o,P5Diagnostic *diagnostic) {
    P5State (&stack)[P5_STACK_CAP]=q->stack;P5State &state=q->state;double *a=state.a;
    double *hs=q->hs,*fine=q->fine;int *touch=q->touch;
    int &top=q->top,&steps=q->steps,&phase=q->phase;
    double M=t->constants[0];
    double &hn=q->hn;
    double &expected_n=q->expected_n;
    double &de=q->de;
    double &radiation=q->radiation;
    double &source_energy=q->source_energy;
    double &dep=q->dep;
    double &escaped=q->escaped;
    double &rest=q->rest;
    double &positron_escape_rest=q->positron_escape_rest;
    double &cached_ev=q->cached_ev;
    double &arate=q->arate;
    double &S=q->S;
    double &rate=q->rate;
    double &total=q->total;
    double &G1=q->G1;
    double &brate=q->brate;
    double &bsoft=q->bsoft;
    bool &failed=q->failed;
    bool &annihilated=q->annihilated;
    bool &suspended=q->suspended;
    long long *counts=o->counts;int batch=q->batch;
        if(top<0) {phase=P5H_H;return;}
        state=stack[top--];
        if(a[19]==0) {
            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];phase=P5H_H;return;}
            double E=a[6];
            if(E<.01) {
                // The fault macro for charged deposits must not change photons.
                p5_local(hs,150,a[2],E,.2);p5_local(fine,600,a[2],E,.05);
                int b0=p5_clip_bin(int(a[2]/.2),150),b1=p5_clip_bin(int(a[2]/.05),600);
                if(b0<touch[0])touch[0]=b0;if(b0>touch[1])touch[1]=b0;
                if(b1<touch[2])touch[2]=b1;if(b1>touch[3])touch[3]=b1;
                dep+=E;phase=P5H_P;return;
            }
            double c[4];p5h_coefficients(E,t,c);double mu=0.;for(int z=0;z<4;++z)mu+=c[z];
            double bb;int nr;p5_boundary(a[2],a[5],int(a[16]),&bb,&nr);
            double flight_u=rng_uniform(s),tau=-log(flight_u),flight=tau/mu,length=p5_min(flight,bb);
            a[0]+=length*a[3];a[1]+=length*a[4];a[2]+=length*a[5];
            if(bb<=flight) {escaped+=E;++counts[7];phase=P5H_P;return;}
#if P5_TRANSPORT_FAULT == 3
            double unused=rng_uniform(s);(void)unused;
#endif
            double select=rng_uniform(s),r=select*mu;int process=0;
            while(process<3 && r>=c[process]) {r-=c[process];++process;}
            ++counts[8+process];
            if(top+3>=P5_STACK_CAP) {failed=true;++counts[6];phase=P5H_H;return;}
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
            if(top+1>o->maxdepth)o->maxdepth=top+1;phase=P5H_P;return;
        }
        cached_ev=-1.;arate=0.;S=0.;rate=0.;total=0.;G1=0.;brate=0.;bsoft=0.;
        annihilated=false;suspended=false;phase=P5H_C;
}
P5_FN void p5h_charged(P5hState *q,const P5hTables *t,RngState *s,int maxsteps,P5Output *o,P5Diagnostic *diagnostic) {
    P5State (&stack)[P5_STACK_CAP]=q->stack;P5State &state=q->state;double *a=state.a;
    double *hs=q->hs,*fine=q->fine;int *touch=q->touch;
    int &top=q->top,&steps=q->steps,&phase=q->phase;
    double M=t->constants[0];
    double &hn=q->hn;
    double &expected_n=q->expected_n;
    double &de=q->de;
    double &radiation=q->radiation;
    double &source_energy=q->source_energy;
    double &dep=q->dep;
    double &escaped=q->escaped;
    double &rest=q->rest;
    double &positron_escape_rest=q->positron_escape_rest;
    double &cached_ev=q->cached_ev;
    double &arate=q->arate;
    double &S=q->S;
    double &rate=q->rate;
    double &total=q->total;
    double &G1=q->G1;
    double &brate=q->brate;
    double &bsoft=q->bsoft;
    bool &failed=q->failed;
    bool &annihilated=q->annihilated;
    bool &suspended=q->suspended;
    long long *counts=o->counts;int batch=q->batch;
            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];phase=failed?P5H_H:P5H_E;return;}
            int region=int(a[16]);
            if(region<0 || region>=1) {escaped+=a[6];if(a[19]==1) {++counts[13];positron_escape_rest+=2*M;}phase=failed?P5H_H:P5H_E;return;}
            if(a[6]<=.01+1e-14 && a[18]!=1) {
                p5_track_local(hs,150,a[2],a[6],.2,touch);p5_track_local(fine,600,a[2],a[6],.05,touch+2);
                dep+=a[6];a[6]=0.;phase=failed?P5H_H:P5H_E;return;
            }
            double ev=p5_max(.01,a[7]);
            if(ev!=cached_ev) {
                double q[4];p5h_quantities(ev,t,q);brate=q[0];bsoft=q[2];double hardrate,hardstop;
                p5_hard(ev,t,&hardrate,&hardstop);S=p5h_collision(ev,t)-hardstop+bsoft;rate=hardrate;
                p5h_table_rates(ev,t,&total,&G1);
                if(a[19]==1)arate=p5_annihilation_rate(ev,t);cached_ev=ev;
            }
            double Sq=S*1.,rateq=rate*1.,G1q=G1*1.,brateq=brate*1.,bsoftq=bsoft*1.,arateq=arate*1.;
            double eb=a[8]/Sq,mb=G1q>0?a[12]/G1q:INFINITY,hb=rateq>0?a[15]/rateq:INFINITY;
            if(a[18]==1) {mb=INFINITY;hb=INFINITY;}
            double bb;int nr;p5_boundary(a[2],a[5],region,&bb,&nr);++counts[4];
            double rb=brateq>0 && a[18]!=1?a[22]/brateq:INFINITY;
            double step=p5_min(p5_min(p5_min(p5_min(p5_min(eb,mb),hb),bb),rb),a[6]/Sq),ab=INFINITY;
            if(a[19]==1 && a[18]!=1) {ab=arateq>0?a[21]/arateq:INFINITY;step=p5_min(step,ab);}
            if(!isfinite(step) || step<0) {failed=true;++counts[6];phase=failed?P5H_H:P5H_E;return;}
            double z0=a[2];a[0]+=step*a[3];a[1]+=step*a[4];a[2]+=step*a[5];
            double loss=step*Sq;a[6]-=loss;dep+=loss;radiation+=step*bsoftq;
            if(a[18]!=1) {expected_n+=step*brateq;a[22]=p5_max(0.,a[22]-step*brateq);}
            p5_track_segment(hs,150,z0,a[2],loss,.2,touch);p5_track_segment(fine,600,z0,a[2],loss,.05,touch+2);
            a[8]=p5_max(0.,a[8]-loss);a[12]=p5_max(0.,a[12]-step*G1q);a[15]=p5_max(0.,a[15]-step*rateq);
            if(a[19]==1 && a[18]!=1)a[21]=p5_max(0.,a[21]-step*arateq);++counts[0];
            if(bb<=step+1e-13) {a[2]=a[5]>0?30.:0.;a[16]=nr;++counts[3];phase=P5H_C;return;}
            if(eb<=step+1e-13) {
                if(a[18]==1) {
                    p5_track_local(hs,150,a[2],a[6],.2,touch);p5_track_local(fine,600,a[2],a[6],.05,touch+2);
                    dep+=a[6];a[6]=0.;phase=failed?P5H_H:P5H_E;return;
                }
                a[7]=p5_max(.01,a[7]-a[11]);
                if(a[7]<=.01+1e-14) {a[8]=a[9];a[9]=0.;a[10]=0.;a[18]=1.;phase=P5H_C;return;}
                double prev=a[9];a[11]=p5_min(.05*a[7],a[7]-.01);double u=rng_uniform(s);a[10]=u*a[11];
                a[9]=a[11]-a[10];a[8]=prev+a[10];phase=P5H_C;return;
            }
            if(mb<=step+1e-13) {
                double mu=p5h_gs_mu_cached(ev,a[14],t,total,G1,s);
                double angle=rng_uniform(s),v[3];p5_rotate(a[3],a[4],a[5],mu,2*3.141592653589793*angle,v);
                for(int z=0;z<3;++z)a[3+z]=v[z];
                double u=rng_uniform(s),knew=p5_rule(ev);a[12]=a[13]+u*knew;a[13]=(1-u)*knew;a[14]=knew;
                ++counts[1];phase=P5H_C;return;
            }
            if(hb<=step+1e-13) {
                p5_energy_init(&state,s);double u=rng_uniform(s);a[15]=-log(u);if(a[6]<=2*.01) {phase=P5H_C;return;}
                double W=p5_sample_W(a[6],t,s),dp[3],ds[3],k=(a[6]+2*M)/a[6];
                double mp=sqrt((a[6]-W)/(a[6]-W+2*M)*k),ms=sqrt(W/(W+2*M)*k);
                double angle=rng_uniform(s),phi=2*3.141592653589793*angle;
                p5_rotate(a[3],a[4],a[5],mp,phi,dp);p5_rotate(a[3],a[4],a[5],ms,phi+3.141592653589793,ds);
                a[6]-=W;for(int z=0;z<3;++z)a[3+z]=dp[z];p5_energy_init(&state,s);++counts[2];
                if(top+2>=P5_STACK_CAP) {a[6]+=W;failed=true;++counts[6];phase=failed?P5H_H:P5H_E;return;}
                stack[++top]=state;++top;p5_initial(stack+top,W,a[0],a[1],a[2],ds[0],ds[1],ds[2],int(a[16]),int(a[20]),t,s);
                if(top+1>o->maxdepth)o->maxdepth=top+1;phase=failed?P5H_H:P5H_E;return;
            }
            if(rb<=step+1e-13) {
                p5_energy_init(&state,s);double T=a[6],u=rng_uniform(s);a[22]=-log(u);if(T<=.01) {phase=P5H_C;return;}
                int excess;double photon_k=p5h_sample_k(T,t,s,&excess);
#if P5_TRANSPORT_FAULT == 1
                photon_k*=1+1e-12;
#endif
                o->brems[2]+=excess;if(photon_k>T)o->brems[3]+=1.;
                if(excess || photon_k>T) {failed=true;++counts[6];phase=failed?P5H_H:P5H_E;return;}
                double q[4];p5h_quantities(T,t,q);hn+=1.;de+=photon_k-q[1]/q[0];radiation+=photon_k;
                o->brems[0]+=1.;o->brems[1]+=photon_k;
                double angle=rng_uniform(s),theta=M/(T+M),v[3];
                p5_rotate(a[3],a[4],a[5],cos(theta),2*3.141592653589793*angle,v);
                if(top+2>=P5_STACK_CAP) {failed=true;++counts[6];phase=failed?P5H_H:P5H_E;return;}
                a[6]-=photon_k;p5_energy_init(&state,s);stack[++top]=state;++top;
                p5_photon(stack+top,photon_k,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));
                if(top+1>o->maxdepth)o->maxdepth=top+1;suspended=true;phase=failed?P5H_H:P5H_E;return;
            }
            if(a[19]==1 && ab<=step+1e-13) {
                double T=a[6];if(T<=.01) {double u=rng_uniform(s);a[21]=-log(u);phase=P5H_C;return;}
                if(top+3>=P5_STACK_CAP) {failed=true;++counts[6];phase=failed?P5H_H:P5H_E;return;}
                double g[8];p5_flight_photons(T,a[3],a[4],a[5],t,s,g);++top;
                p5_photon(stack+top,g[0],a[0],a[1],a[2],g[1],g[2],g[3],int(a[16]),int(a[20]));++top;
                p5_photon(stack+top,g[4],a[0],a[1],a[2],g[5],g[6],g[7],int(a[16]),int(a[20]));
                rest-=2*M;a[6]=0.;annihilated=true;++counts[14];if(a[17]==1)++counts[15];
                if(top+1>o->maxdepth)o->maxdepth=top+1;phase=failed?P5H_H:P5H_E;return;
            }
            p5_track_local(hs,150,a[2],a[6],.2,touch);p5_track_local(fine,600,a[2],a[6],.05,touch+2);
            dep+=a[6];a[6]=0.;phase=failed?P5H_H:P5H_E;return;
}
P5_FN void p5h_end(P5hState *q,const P5hTables *t,RngState *s,int maxsteps,P5Output *o,P5Diagnostic *diagnostic) {
    P5State (&stack)[P5_STACK_CAP]=q->stack;P5State &state=q->state;double *a=state.a;
    double *hs=q->hs,*fine=q->fine;int *touch=q->touch;
    int &top=q->top,&steps=q->steps,&phase=q->phase;
    double M=t->constants[0];
    double &hn=q->hn;
    double &expected_n=q->expected_n;
    double &de=q->de;
    double &radiation=q->radiation;
    double &source_energy=q->source_energy;
    double &dep=q->dep;
    double &escaped=q->escaped;
    double &rest=q->rest;
    double &positron_escape_rest=q->positron_escape_rest;
    double &cached_ev=q->cached_ev;
    double &arate=q->arate;
    double &S=q->S;
    double &rate=q->rate;
    double &total=q->total;
    double &G1=q->G1;
    double &brate=q->brate;
    double &bsoft=q->bsoft;
    bool &failed=q->failed;
    bool &annihilated=q->annihilated;
    bool &suspended=q->suspended;
    long long *counts=o->counts;int batch=q->batch;
        if(!suspended && !annihilated && a[19]==1 && a[6]==0 && a[16]>=0 && a[16]<1) {
            if(top+3>=P5_STACK_CAP) {a[6]=0.;failed=true;++counts[6];phase=P5H_H;return;}
            double v[3];p5_annihilation(s,v);rest-=2*M;++counts[12];++top;
            p5_photon(stack+top,M,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));++top;
            p5_photon(stack+top,M,a[0],a[1],a[2],-v[0],-v[1],-v[2],int(a[16]),int(a[20]));
            if(top+1>o->maxdepth)o->maxdepth=top+1;
        }
    phase=P5H_P;
}
P5_FN void p5h_finish(P5hState *q,const P5hTables *t,RngState *s,int maxsteps,P5Output *o,P5Diagnostic *diagnostic) {
    P5State (&stack)[P5_STACK_CAP]=q->stack;P5State &state=q->state;double *a=state.a;
    double *hs=q->hs,*fine=q->fine;int *touch=q->touch;
    int &top=q->top,&steps=q->steps,&phase=q->phase;
    double M=t->constants[0];
    double &hn=q->hn;
    double &expected_n=q->expected_n;
    double &de=q->de;
    double &radiation=q->radiation;
    double &source_energy=q->source_energy;
    double &dep=q->dep;
    double &escaped=q->escaped;
    double &rest=q->rest;
    double &positron_escape_rest=q->positron_escape_rest;
    double &cached_ev=q->cached_ev;
    double &arate=q->arate;
    double &S=q->S;
    double &rate=q->rate;
    double &total=q->total;
    double &G1=q->G1;
    double &brate=q->brate;
    double &bsoft=q->bsoft;
    bool &failed=q->failed;
    bool &annihilated=q->annihilated;
    bool &suspended=q->suspended;
    long long *counts=o->counts;int batch=q->batch;
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
    phase=P5H_D;
}
P5_FN void p5h_transition(P5hState *q,const P5hTables *t,RngState *s,int maxsteps,P5Output *o,P5Diagnostic *d,P5hTrace *tr) {
    int from=q->phase;
    if(from==P5H_P)p5h_particle(q,t,s,maxsteps,o,d);
    else if(from==P5H_C)p5h_charged(q,t,s,maxsteps,o,d);
    else if(from==P5H_E)p5h_end(q,t,s,maxsteps,o,d);
    else if(from==P5H_H)p5h_finish(q,t,s,maxsteps,o,d);
    p5h_record(tr,from,q,s,o);
}
P5_FN void p5h_history(const P5hTables *t,RngState *s,int batch,int maxsteps,P5Output *o,P5Diagnostic *d,P5hTrace *tr=0) {
    P5hState q;p5h_init(&q,t,s,batch,o,tr);
    while(q.phase!=P5H_D)p5h_transition(&q,t,s,maxsteps,o,d,tr);
}
#endif
