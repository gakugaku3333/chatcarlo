#ifndef P6_KERNEL_H
#define P6_KERNEL_H
#include "p5j_kernel.h"
#include "p5b_wrap.h"
#include "p6_field.h"
// Transport state is composed, not a modified copy of the frozen P5j type.
struct P6State {P5jState base;double region_sums[15];};
P5_FN void p6_init(P6State *q,const P5jTables *t,RngState *s,int batch,P5Output *o,
                  bool field,P6Records *r,P5jTrace *tr=0) {
    for(int j=0;j<15;++j)q->region_sums[j]=0.;
    p5j_init(&q->base,t,s,batch,o,0);
    double *a=q->base.stack[0].a;
    if(field) {
        double x=(2*rng_uniform(s)-1)*5.,y=(2*rng_uniform(s)-1)*5.;
#if P6_FAULT == 3
        double tmp=x;x=y;y=tmp;
#endif
        double norm=sqrt(x*x+y*y+100.*100.);
        a[0]=x;a[1]=y;a[3]=x/norm;a[4]=y/norm;a[5]=100./norm;
    }
    if(r && r->source_capacity) {
        if(r->source_used<r->source_capacity)for(int j=0;j<7;++j)r->sources[7*r->source_used+j]=a[j];
        else ++r->source_overflow;
        ++r->source_used;
    }
    p5j_record(tr,-1,&q->base,s,o);
}
P5_FN void p6_particle(P6State *p,const P5jTables *t,RngState *s,int maxsteps,P5Output *o,P5Diagnostic *diagnostic,P6Sink *sink,P6Records *records) {
    P5jState *q=&p->base;
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
        if(top<0) {phase=P5J_H;return;}
        state=stack[top--];
        if(a[19]==0) {
            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];phase=P5J_H;return;}
            double E=a[6];
            if(E<.01) {
                p6_deposit_point(sink,records,1,a,E);
                p5_local(hs,150,a[2],E,.2);p5_local(fine,600,a[2],E,.05);
                int b0=p5_clip_bin(int(a[2]/.2),150),b1=p5_clip_bin(int(a[2]/.05),600);
                if(b0<touch[0])touch[0]=b0;if(b0>touch[1])touch[1]=b0;
                if(b1<touch[2])touch[2]=b1;if(b1>touch[3])touch[3]=b1;
                dep+=E;phase=P5J_P;return;
            }
            double c[4];p5j_coefficients(E,t,c);double mu=0.;for(int z=0;z<4;++z)mu+=c[z];
            double bb;int nr;p5_boundary(a[2],a[5],int(a[16]),&bb,&nr);
            double flight_u=rng_uniform(s),tau=-log(flight_u),flight=tau/mu,length=p5_min(flight,bb);
            a[0]+=length*a[3];a[1]+=length*a[4];a[2]+=length*a[5];
            if(bb<=flight) {escaped+=E;++counts[7];phase=P5J_P;return;}
#if P5_TRANSPORT_FAULT == 3
            double unused=rng_uniform(s);(void)unused;
#endif
            double select=rng_uniform(s),r=select*mu;int process=0;
            while(process<3 && r>=c[process]) {r-=c[process];++process;}
            ++counts[8+process];
            if(top+3>=P5_STACK_CAP) {failed=true;++counts[6];phase=P5J_H;return;}
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
            if(top+1>o->maxdepth)o->maxdepth=top+1;phase=P5J_P;return;
        }
        cached_ev=-1.;arate=0.;S=0.;rate=0.;total=0.;G1=0.;brate=0.;bsoft=0.;
        annihilated=false;suspended=false;phase=P5J_C;
}
P5_FN void p6_charged(P6State *p,const P5jTables *t,RngState *s,int maxsteps,P5Output *o,P5Diagnostic *diagnostic,P6Sink *sink,P6Records *records) {
    P5jState *q=&p->base;
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
            if(steps>maxsteps) {failed=true;++counts[5];phase=failed?P5J_H:P5J_E;return;}
            int region=int(a[16]);
            if(region<0 || region>=1) {escaped+=a[6];if(a[19]==1) {++counts[13];positron_escape_rest+=2*M;}phase=failed?P5J_H:P5J_E;return;}
            if(a[6]<=.01+1e-14 && a[18]!=1) {
                p6_deposit_point(sink,records,2,a,a[6]);
                p5_track_local(hs,150,a[2],a[6],.2,touch);p5_track_local(fine,600,a[2],a[6],.05,touch+2);
                dep+=a[6];a[6]=0.;phase=failed?P5J_H:P5J_E;return;
            }
            double ev=p5_max(.01,a[7]);
            if(ev!=cached_ev) {
                double Lsrc=ev,Lval=log(ev);
                double q[4];p5j_quantities(ev,t,q,Lsrc,Lval);brate=q[0];bsoft=q[2];double hardrate,hardstop;
                p5j_hard(ev,t,&hardrate,&hardstop);S=p5j_collision(ev,t,Lsrc,Lval)-hardstop+bsoft;rate=hardrate;
                p5j_table_rates(ev,t,&total,&G1);
                if(a[19]==1)arate=p5_annihilation_rate(ev,t);cached_ev=ev;
            }
            double Sq=S*1.,rateq=rate*1.,G1q=G1*1.,brateq=brate*1.,bsoftq=bsoft*1.,arateq=arate*1.;
            double eb=a[8]/Sq,mb=G1q>0?a[12]/G1q:INFINITY,hb=rateq>0?a[15]/rateq:INFINITY;
            if(a[18]==1) {mb=INFINITY;hb=INFINITY;}
            double bb;int nr;p5_boundary(a[2],a[5],region,&bb,&nr);++counts[4];
            double rb=brateq>0 && a[18]!=1?a[22]/brateq:INFINITY;
            double step=p5_min(p5_min(p5_min(p5_min(p5_min(eb,mb),hb),bb),rb),a[6]/Sq),ab=INFINITY;
            if(a[19]==1 && a[18]!=1) {ab=arateq>0?a[21]/arateq:INFINITY;step=p5_min(step,ab);}
            if(!isfinite(step) || step<0) {failed=true;++counts[6];phase=failed?P5J_H:P5J_E;return;}
            double x0=a[0],y0=a[1],z0=a[2];a[0]+=step*a[3];a[1]+=step*a[4];a[2]+=step*a[5];
            double loss=step*Sq;a[6]-=loss;dep+=loss;radiation+=step*bsoftq;
            if(a[18]!=1) {expected_n+=step*brateq;a[22]=p5_max(0.,a[22]-step*brateq);}
            double start_point[3]={x0,y0,z0};p6_deposit_segment(sink,records,start_point,a,loss);
            p5_track_segment(hs,150,z0,a[2],loss,.2,touch);p5_track_segment(fine,600,z0,a[2],loss,.05,touch+2);
            a[8]=p5_max(0.,a[8]-loss);a[12]=p5_max(0.,a[12]-step*G1q);a[15]=p5_max(0.,a[15]-step*rateq);
            if(a[19]==1 && a[18]!=1)a[21]=p5_max(0.,a[21]-step*arateq);++counts[0];
            if(bb<=step+1e-13) {a[2]=a[5]>0?30.:0.;a[16]=nr;++counts[3];phase=P5J_C;return;}
            if(eb<=step+1e-13) {
                if(a[18]==1) {
                    p6_deposit_point(sink,records,4,a,a[6]);
                    p5_track_local(hs,150,a[2],a[6],.2,touch);p5_track_local(fine,600,a[2],a[6],.05,touch+2);
                    dep+=a[6];a[6]=0.;phase=failed?P5J_H:P5J_E;return;
                }
                a[7]=p5_max(.01,a[7]-a[11]);
                if(a[7]<=.01+1e-14) {a[8]=a[9];a[9]=0.;a[10]=0.;a[18]=1.;phase=P5J_C;return;}
                double prev=a[9];a[11]=p5_min(.05*a[7],a[7]-.01);double u=rng_uniform(s);a[10]=u*a[11];
                a[9]=a[11]-a[10];a[8]=prev+a[10];phase=P5J_C;return;
            }
            if(mb<=step+1e-13) {
                double mu=p5j_gs_mu_cached(ev,a[14],t,total,G1,s);
                double angle=rng_uniform(s),v[3];p5_rotate(a[3],a[4],a[5],mu,2*3.141592653589793*angle,v);
                for(int z=0;z<3;++z)a[3+z]=v[z];
                double u=rng_uniform(s),knew=p5_rule(ev);a[12]=a[13]+u*knew;a[13]=(1-u)*knew;a[14]=knew;
                ++counts[1];phase=P5J_C;return;
            }
            if(hb<=step+1e-13) {
                p5_energy_init(&state,s);double u=rng_uniform(s);a[15]=-log(u);if(a[6]<=2*.01) {phase=P5J_C;return;}
                double W=p5_sample_W(a[6],t,s),dp[3],ds[3],k=(a[6]+2*M)/a[6];
                double mp=sqrt((a[6]-W)/(a[6]-W+2*M)*k),ms=sqrt(W/(W+2*M)*k);
                double angle=rng_uniform(s),phi=2*3.141592653589793*angle;
                p5_rotate(a[3],a[4],a[5],mp,phi,dp);p5_rotate(a[3],a[4],a[5],ms,phi+3.141592653589793,ds);
                a[6]-=W;for(int z=0;z<3;++z)a[3+z]=dp[z];p5_energy_init(&state,s);++counts[2];
                if(top+2>=P5_STACK_CAP) {a[6]+=W;failed=true;++counts[6];phase=failed?P5J_H:P5J_E;return;}
                stack[++top]=state;++top;p5_initial(stack+top,W,a[0],a[1],a[2],ds[0],ds[1],ds[2],int(a[16]),int(a[20]),t,s);
                if(top+1>o->maxdepth)o->maxdepth=top+1;phase=failed?P5J_H:P5J_E;return;
            }
            if(rb<=step+1e-13) {
                p5_energy_init(&state,s);double T=a[6],u=rng_uniform(s);a[22]=-log(u);if(T<=.01) {phase=P5J_C;return;}
                int excess;double photon_k=p5j_sample_k(T,t,s,&excess);
#if P5_TRANSPORT_FAULT == 1
                photon_k*=1+1e-12;
#endif
                o->brems[2]+=excess;if(photon_k>T)o->brems[3]+=1.;
                if(excess || photon_k>T) {failed=true;++counts[6];phase=failed?P5J_H:P5J_E;return;}
                double q[4];p5j_quantities(T,t,q);hn+=1.;de+=photon_k-q[1]/q[0];radiation+=photon_k;
                o->brems[0]+=1.;o->brems[1]+=photon_k;
                double angle=rng_uniform(s),theta=M/(T+M),v[3];
                p5_rotate(a[3],a[4],a[5],cos(theta),2*3.141592653589793*angle,v);
                if(top+2>=P5_STACK_CAP) {failed=true;++counts[6];phase=failed?P5J_H:P5J_E;return;}
                a[6]-=photon_k;p5_energy_init(&state,s);stack[++top]=state;++top;
                p5_photon(stack+top,photon_k,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));
                if(top+1>o->maxdepth)o->maxdepth=top+1;suspended=true;phase=failed?P5J_H:P5J_E;return;
            }
            if(a[19]==1 && ab<=step+1e-13) {
                double T=a[6];if(T<=.01) {double u=rng_uniform(s);a[21]=-log(u);phase=P5J_C;return;}
                if(top+3>=P5_STACK_CAP) {failed=true;++counts[6];phase=failed?P5J_H:P5J_E;return;}
                double g[8];p5_flight_photons(T,a[3],a[4],a[5],t,s,g);++top;
                p5_photon(stack+top,g[0],a[0],a[1],a[2],g[1],g[2],g[3],int(a[16]),int(a[20]));++top;
                p5_photon(stack+top,g[4],a[0],a[1],a[2],g[5],g[6],g[7],int(a[16]),int(a[20]));
                rest-=2*M;a[6]=0.;annihilated=true;++counts[14];if(a[17]==1)++counts[15];
                if(top+1>o->maxdepth)o->maxdepth=top+1;phase=failed?P5J_H:P5J_E;return;
            }
            p6_deposit_point(sink,records,5,a,a[6]);
            p5_track_local(hs,150,a[2],a[6],.2,touch);p5_track_local(fine,600,a[2],a[6],.05,touch+2);
            dep+=a[6];a[6]=0.;phase=failed?P5J_H:P5J_E;return;
}
P5_FN void p6_transition(P6State *q,const P5jTables *t,RngState *s,int maxsteps,P5Output *o,P5Diagnostic *d,P6Sink *sink,long long *ledger,P6Records *r,P5jTrace *tr=0) {
    int from=q->base.phase;
    if(from==P5J_P)p6_particle(q,t,s,maxsteps,o,d,sink,r);
    else if(from==P5J_C)p6_charged(q,t,s,maxsteps,o,d,sink,r);
    else if(from==P5J_E)p5j_end(&q->base,t,s,maxsteps,o,d);
    else if(from==P5J_H){p6_ledger(sink,q->base.dep,q->base.source_energy,ledger);p5j_finish(&q->base,t,s,maxsteps,o,d);}
    p5j_record(tr,from,&q->base,s,o);
}
struct P6Slot {
    P6State q;RngState rng;P5bLocal L;long long nspec_work[640];
    double regions[P6_BINS],sources[7];long long ledger[15];
    int i,active,diag_used,trace_used;long long diag_overflow,trace_overflow;
};
P5_FN P5Output p6_output(P6Slot *s) {
    return P5Output{s->L.batches,s->L.fine_batches,s->L.totals,s->L.maxima,s->L.brems,
        s->L.violations,s->L.counts,s->nspec_work,&s->L.steps,s->L.maxdepth};
}
P5_FN P5jTrace p6_trace(P6Slot *s,double *trace,int capacity) {
    return P5jTrace{trace?trace+uint64_t(s->i)*capacity*10:0,capacity,s->trace_used,s->trace_overflow};
}
P5_FN void p6_initialize(P6Slot *s,int i,int n,const P5jTables *t,uint64_t seed,uint64_t start,bool field,double *trace=0,int capacity=0) {
    s->active=0;s->q.base.phase=P5J_D;s->i=i;s->L=P5bLocal{};
    s->diag_used=0;s->diag_overflow=0;s->trace_used=0;s->trace_overflow=0;
    for(int j=0;j<640;++j)s->nspec_work[j]=0;
    for(int j=0;j<15;++j)s->ledger[j]=0;
#if P6_FAULT != 4
    for(int j=0;j<P6_BINS;++j)s->regions[j]=0.;
#endif
    if(i>=n)return;
    s->active=1;rng_init(&s->rng,seed,start+uint64_t(i));
    P5Output o=p6_output(s);P5jTrace tr=p6_trace(s,trace,capacity);
    P6Records records={0,s->sources,0,0,1,0,0,0,0,uint64_t(i)};
    p6_init(&s->q,t,&s->rng,0,&o,field,&records,trace?&tr:0);
    s->L.maxdepth=o.maxdepth;s->trace_used=tr.used;s->trace_overflow=tr.overflow;
}
P5_FN void p6_work(P6Slot *s,const P5jTables *t,const P6Geometry *g,bool regions,int maxsteps,double *trace=0,int capacity=0,unsigned long long *additions=0) {
    P5Output o=p6_output(s);P5Diagnostic d={s->L.diagnostic,4,s->diag_used,s->diag_overflow};
    P5jTrace tr=p6_trace(s,trace,capacity);P6Sink sink={s->regions,s->q.region_sums,g,int(regions),additions};
    p6_transition(&s->q,t,&s->rng,maxsteps,&o,&d,&sink,s->ledger,0,trace?&tr:0);
    s->L.maxdepth=o.maxdepth;s->diag_used=d.used;s->diag_overflow=d.overflow;
    s->trace_used=tr.used;s->trace_overflow=tr.overflow;
}
P5_FN void p6_finalize(P6Slot *s,const P5jTables *t,double *trace=0,int capacity=0,long long *meta=0) {
    s->L.calls=s->rng.calls;s->L.diag_used=s->diag_used<4?s->diag_used:4;s->L.nspec_index=-1;
    for(int j=0;j<t->nspec*10;++j)if(s->nspec_work[j])s->L.nspec_index=j;
    P5Output o=p6_output(s);P5jTrace tr=p6_trace(s,trace,capacity);p5j_record(trace?&tr:0,5,&s->q.base,&s->rng,&o);
    if(meta){meta[2*s->i]=tr.used;meta[2*s->i+1]=tr.overflow;}
}
#endif
