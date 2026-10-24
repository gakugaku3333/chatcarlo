#ifndef P8G2_KERNEL_H
#define P8G2_KERNEL_H
#include "p8g_kernel.h"
#ifndef P8G2_FAULT
#define P8G2_FAULT 0
#endif
enum P8g2Phase {P8G2_P,P8G2_C,P8G2_E,P8G2_H,P8G2_D};
#ifndef P8G2_CHECK
#define P8G2_CHECK 0
#endif
// hits is always caller-owned global device memory in CUDA builds.
P5_FN void p8g2_hit(unsigned long long *hits,int j){
#if P8G2_CHECK
if(!hits)return;
#ifdef P5_GPU
atomicAdd(hits+j,1ULL);
#else
++hits[j];
#endif
#endif
}
struct P8g2State {
P5State stack[P8G_STACK_CAP],state;
RngState rng;P8gLocal local;P8gRecord record;
int resume_top,resume_kind;bool boundary_pending;
int phase,top,steps,touch[4],diag_used,maxdepth;long long diag_overflow;
bool failed,annihilated,suspended;
double M,hn,expected_n,de,radiation,source_energy,dep,escaped,rest,positron_escape_rest;
double cached_ev,arate,S,rate,total,G1,brate,bsoft;
};
// Initial slot setup touches only output values needed before the first begin.
// Stack entries are initialized exclusively by pushes, never by a blanket clear.
P5_FN void p8g2_slot_init(P8g2State *q,const P8gGeometry *g){
q->phase=P8G2_P;for(int j=0;j<g->nb;++j)q->local.bins[j]=0.;
}
P5_FN P8gOutput p8g2_output(const P8gGeometry *g,P8gRecord *r){
int b=g->nb,c=b+g->nc,f=c+g->nf;
P8gOutput o={r->d+b,r->d+c,r->d+f,r->d+f+6,r->d+f+9,r->d,r->d+f+19,
r->i,r->i+3,r->i+19,r->i+20,r->i+25,r->i+43,r->i+61,r->i+69,0};return o;
}
P5_FN void p8g2_begin(P8g2State *q,const double *data,const long long *meta,const double *constants,const P8gGeometry *g,uint64_t seed,uint64_t hid,unsigned long long *hits,const P5State *injected=0){
P8gOutput out=p8g2_output(g,&q->record);P8gOutput *o=&out;
auto &phase=q->phase;
auto &top=q->top;
auto &steps=q->steps;
auto &touch=q->touch;
auto &failed=q->failed;
auto &annihilated=q->annihilated;
auto &suspended=q->suspended;
auto &M=q->M;
auto &hn=q->hn;
auto &expected_n=q->expected_n;
auto &de=q->de;
auto &radiation=q->radiation;
auto &source_energy=q->source_energy;
auto &dep=q->dep;
auto &escaped=q->escaped;
auto &rest=q->rest;
auto &positron_escape_rest=q->positron_escape_rest;
auto &cached_ev=q->cached_ev;
auto &arate=q->arate;
auto &S=q->S;
auto &rate=q->rate;
auto &total=q->total;
auto &G1=q->G1;
auto &brate=q->brate;
auto &bsoft=q->bsoft;
auto &stack=q->stack;
auto &state=q->state;
RngState *s=&q->rng;P8gLocal *local=&q->local;
double *hs=local->coarse,*fine=local->fine,*bins=local->bins;
double *a=state.a;long long *counts=o->counts;
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p;

#if P8G2_FAULT == 2
if(q->phase==P8G2_D)p8g2_hit(hits,31);
#else
for(int j=0;j<g->nb;++j)local->bins[j]=0.;
#endif
for(int j=0;j<g->nb;++j)local->additions[j]=0;
for(int j=0;j<g->nc;++j)local->coarse[j]=0.;for(int j=0;j<g->nf;++j)local->fine[j]=0.;
for(int j=0;j<p8g_dwidth(g);++j)q->record.d[j]=0.;for(int j=0;j<p8g_iwidth(g);++j)q->record.i[j]=0;
q->diag_used=q->diag_overflow=q->maxdepth=0;
touch[0]=g->nc;touch[1]=-1;touch[2]=g->nf;touch[3]=-1;
M=t->constants[0];hn=expected_n=de=radiation=0.;
rng_init(s,seed,hid);
source_energy=g->energy;local->nspec_index=-1;
    if(g->nspec){double u1=rng_uniform(s),u2=rng_uniform(s);int j=0;while(g->cum[j]<u1)++j;
        double width=g->edges[j+1]-g->edges[j];source_energy=g->edges[j]+u2*width;int cell=int((source_energy-g->edges[j])/width*10);if(cell>9)cell=9;local->nspec_index=j*10+cell;}
    local->source=source_energy;local->event_used=0;local->skipped=0;
    p5_photon(stack,source_energy,0.,0.,0.,0.,0.,1.,0,0);
    if(injected){stack[0]=*injected;if(stack[0].a[19]!=0){double u=rng_uniform(s);stack[0].a[22]=-log(u);}if(stack[0].a[19]==1){double u=rng_uniform(s);stack[0].a[21]=-log(u);}}
    q->resume_top=-2;q->boundary_pending=false;top=0;steps=0;dep=escaped=rest=positron_escape_rest=0.;failed=false;
    
phase=P8G2_P;
}
P5_FN void p8g2_p(P8g2State *q,const double *data,const long long *meta,const double *constants,const P8gGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){
auto &phase=q->phase;
auto &top=q->top;
auto &steps=q->steps;
auto &touch=q->touch;
auto &failed=q->failed;
auto &annihilated=q->annihilated;
auto &suspended=q->suspended;
auto &M=q->M;
auto &hn=q->hn;
auto &expected_n=q->expected_n;
auto &de=q->de;
auto &radiation=q->radiation;
auto &source_energy=q->source_energy;
auto &dep=q->dep;
auto &escaped=q->escaped;
auto &rest=q->rest;
auto &positron_escape_rest=q->positron_escape_rest;
auto &cached_ev=q->cached_ev;
auto &arate=q->arate;
auto &S=q->S;
auto &rate=q->rate;
auto &total=q->total;
auto &G1=q->G1;
auto &brate=q->brate;
auto &bsoft=q->bsoft;
auto &stack=q->stack;
auto &state=q->state;
RngState *s=&q->rng;P8gLocal *local=&q->local;
double *hs=local->coarse,*fine=local->fine,*bins=local->bins;
double *a=state.a;long long *counts=o->counts;
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p;

        if(top<0){phase=P8G2_H;return;}
        if(top==q->resume_top){p8g2_hit(hits,q->resume_kind==3?12:16);q->resume_top=-2;}state=stack[top--];
        if(a[19]==0) {
            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];phase=P8G2_H;return;}
            double E=a[6];
            m=p8g_material(data,meta+p8g_mid(g->ids[int(a[16])])*136,constants);t=&m.p;
            if(E<.01) {
                if(a[17]>=2)++o->fluorescence_counts[3];if(a[17]==3)++o->fluorescence_counts[7];
                p8g_sink(g,local,a[2],a[2],int(a[16]),E);
                // The fault macro for charged deposits must not change photons.
                p5_local(hs,g->nc,a[2],E,.2);p5_local(fine,g->nf,a[2],E,.05);
                int b0=p5_clip_bin(int(a[2]/.2),g->nc),b1=p5_clip_bin(int(a[2]/.05),g->nf);
                if(b0<touch[0])touch[0]=b0;if(b0>touch[1])touch[1]=b0;
                if(b1<touch[2])touch[2]=b1;if(b1>touch[3])touch[3]=b1;
                dep+=E;p8g2_hit(hits,10);return;
            }
            double c[4];p8g_coefficients(E,m,c);for(int z=0;z<4;++z)c[z]*=g->rho[int(a[16])];double mu=0.;for(int z=0;z<4;++z)mu+=c[z];
            double bb;int nr;p8g_boundary(a[2],a[5],int(a[16]),g->faces,&bb,&nr);
            double flight_u=rng_uniform(s),tau=-log(flight_u),flight=tau/mu,length=p5_min(flight,bb);
            bool escaped_photon=false;
            for(;;){a[0]+=length*a[3];a[1]+=length*a[4];a[2]+=length*a[5];
                if(bb>flight)break;
                if(nr<0 || nr>=g->nl){escaped_photon=true;break;}
                p8g2_hit(hits,0);++o->branches[0];tau=p5_max(0.,tau-length*mu);int region=int(a[16]);a[2]=a[5]>0?g->faces[region+1]:g->faces[region];a[16]=nr;
                m=p8g_material(data,meta+p8g_mid(g->ids[nr])*136,constants);t=&m.p;p8g_coefficients(E,m,c);mu=0.;
                for(int z=0;z<4;++z){c[z]*=g->rho[nr];mu+=c[z];}
                p8g_boundary(a[2],a[5],int(a[16]),g->faces,&bb,&nr);flight=tau/mu;length=p5_min(flight,bb);
            }
            if(escaped_photon){if(a[17]>=2)++o->fluorescence_counts[1];if(a[17]==3)++o->fluorescence_counts[5];escaped+=E;++counts[7];return;}

            double select=rng_uniform(s),r=select*mu;int process=0;
            while(process<3 && r>=c[process]) {r-=c[process];++process;}
            ++counts[8+process];
            if(top+3>=P8G_STACK_CAP) {failed=true;++counts[6];phase=P8G2_H;return;}
            if(process==0) {
                double p[8];p5_compton(E,a[3],a[4],a[5],t,s,p);
                ++top;p5_photon(stack+top,p[0],a[0],a[1],a[2],p[1],p[2],p[3],int(a[16]),int(a[20]));stack[top].a[17]=a[17];
                ++top;p5_initial(stack+top,p[4],a[0],a[1],a[2],p[5],p[6],p[7],int(a[16]),int(a[20]),t,s);
            } else if(process==1) {
                ++o->branches[1];if(a[17]>=2)++o->fluorescence_counts[2];if(a[17]==3)++o->fluorescence_counts[6];double T=E;
                if(g->binding){int element,shell;p8g_select_shell(E,m,s,&element,&shell);double binding=E,local_energy=E,emitted[32];long long trace[24];int events=0,used=0;
                    if(element>=0){binding=m.bindings[element*5+shell];used=p8g_relax(element,shell,m,s,g->fluorescence,emitted,&local_energy,trace,&events);}
                    T=E-binding;if(used>1)p8g2_hit(hits,1);
                    if(top+used+2>=P8G_STACK_CAP){p8g2_hit(hits,2);failed=true;++counts[6];phase=P8G2_H;return;}
#if P8F_FAULT == 3
                    for(int j=0;j<used;++j)local_energy+=emitted[4*j];used=0;
#endif
                    dep+=local_energy;p8g_sink(g,local,a[2],a[2],int(a[16]),local_energy);
                    p5_track_local(hs,g->nc,a[2],local_energy,.2,touch);p5_track_local(fine,g->nf,a[2],local_energy,.05,touch+2);
                    for(int j=0;j<used;++j){++top;p5_photon(stack+top,emitted[4*j],a[0],a[1],a[2],emitted[4*j+1],emitted[4*j+2],emitted[4*j+3],int(a[16]),int(a[20]));stack[top].a[17]=2;
                        for(int k=0;k<int(m.line_counts[element*4]);++k)if(emitted[4*j]==m.lines[(element*4*m.nline+k)*4])stack[top].a[17]=3;
                        ++o->fluorescence_counts[0];if(stack[top].a[17]==3)++o->fluorescence_counts[4];}
                }
                ++top;p5_initial(stack+top,T,a[0],a[1],a[2],a[3],a[4],a[5],int(a[16]),int(a[20]),t,s);
            } else {
                double p[10];p5_pair(E,a[3],a[4],a[5],t,s,p);rest+=2*M;
                P5State first,second;
                p5_initial(&first,p[0],a[0],a[1],a[2],p[1],p[2],p[3],int(a[16]),1,t,s);first.a[19]=p[4];
                p5_initial(&second,p[5],a[0],a[1],a[2],p[6],p[7],p[8],int(a[16]),1,t,s);second.a[19]=p[9];
                if(first.a[19]==1) {double u=rng_uniform(s);first.a[21]=-log(u);}
                if(second.a[19]==1) {double u=rng_uniform(s);second.a[21]=-log(u);}
                stack[++top]=first;stack[++top]=second;
            }
            if(top+1>o->maxdepth)o->maxdepth=top+1;return;
        }
cached_ev=-1.;arate=S=rate=total=G1=brate=bsoft=0.;annihilated=suspended=false;phase=P8G2_C;
}
P5_FN void p8g2_c(P8g2State *q,const double *data,const long long *meta,const double *constants,const P8gGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){
auto &phase=q->phase;
auto &top=q->top;
auto &steps=q->steps;
auto &touch=q->touch;
auto &failed=q->failed;
auto &annihilated=q->annihilated;
auto &suspended=q->suspended;
auto &M=q->M;
auto &hn=q->hn;
auto &expected_n=q->expected_n;
auto &de=q->de;
auto &radiation=q->radiation;
auto &source_energy=q->source_energy;
auto &dep=q->dep;
auto &escaped=q->escaped;
auto &rest=q->rest;
auto &positron_escape_rest=q->positron_escape_rest;
auto &cached_ev=q->cached_ev;
auto &arate=q->arate;
auto &S=q->S;
auto &rate=q->rate;
auto &total=q->total;
auto &G1=q->G1;
auto &brate=q->brate;
auto &bsoft=q->bsoft;
auto &stack=q->stack;
auto &state=q->state;
RngState *s=&q->rng;P8gLocal *local=&q->local;
double *hs=local->coarse,*fine=local->fine,*bins=local->bins;
double *a=state.a;long long *counts=o->counts;
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p;

            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];phase=failed?P8G2_H:P8G2_E;return;}
            int region=int(a[16]);
            if(region<0 || region>=g->nl) {escaped+=a[6];if(a[19]==1) {p8g2_hit(hits,7);++counts[13];positron_escape_rest+=2*M;}phase=failed?P8G2_H:P8G2_E;return;}
            if(a[6]<=.01+1e-14 && a[18]!=1) {
                p8g_sink(g,local,a[2],a[2],int(a[16]),a[6]);p5_track_local(hs,g->nc,a[2],a[6],.2,touch);p5_track_local(fine,g->nf,a[2],a[6],.05,touch+2);
                dep+=a[6];a[6]=0.;phase=failed?P8G2_H:P8G2_E;return;
            }
            m=p8g_material(data,meta+p8g_mid(g->ids[region])*136,constants);t=&m.p;
            double ev=p5_max(.01,a[7]);
            if(ev!=cached_ev) {p8g2_hit(hits,13);if(q->boundary_pending){p8g2_hit(hits,17);q->boundary_pending=false;}
                double quantities[4];p8g_quantities(ev,m,quantities);brate=quantities[0];bsoft=quantities[2];double hardrate,hardstop;
                p8g_hard(ev,m,&hardrate,&hardstop);S=p8g_collision(ev,m)-hardstop+bsoft;rate=hardrate;
                p8g_table_rates(ev,m,&total,&G1);
                if(a[19]==1)arate=p8g_annihilation_rate(ev,m);cached_ev=ev;
            }
            double Sq=S*g->rho[region],rateq=rate*g->rho[region],G1q=G1*g->rho[region],brateq=brate*g->rho[region],bsoftq=bsoft*g->rho[region],arateq=arate*g->rho[region];
            double eb=a[8]/Sq,mb=G1q>0?a[12]/G1q:INFINITY,hb=rateq>0?a[15]/rateq:INFINITY;
            if(a[18]==1) {mb=INFINITY;hb=INFINITY;}
            double bb;int nr;p8g_boundary(a[2],a[5],region,g->faces,&bb,&nr);++counts[4];
            double rb=brateq>0 && a[18]!=1?a[22]/brateq:INFINITY;
            double step=p5_min(p5_min(p5_min(p5_min(p5_min(eb,mb),hb),bb),rb),a[6]/Sq),ab=INFINITY;
            if(a[19]==1 && a[18]!=1) {ab=arateq>0?a[21]/arateq:INFINITY;step=p5_min(step,ab);}
            if(!isfinite(step) || step<0) {failed=true;++counts[6];phase=failed?P8G2_H:P8G2_E;return;}
            double z0=a[2];a[0]+=step*a[3];a[1]+=step*a[4];a[2]+=step*a[5];
            double loss=step*Sq;a[6]-=loss;dep+=loss;radiation+=step*bsoftq;
            if(a[18]!=1) {expected_n+=step*brateq;a[22]=p5_max(0.,a[22]-step*brateq);}
            p8g_sink(g,local,z0,a[2],region,loss);p5_track_segment(hs,g->nc,z0,a[2],loss,.2,touch);p5_track_segment(fine,g->nf,z0,a[2],loss,.05,touch+2);
            a[8]=p5_max(0.,a[8]-loss);a[12]=p5_max(0.,a[12]-step*G1q);a[15]=p5_max(0.,a[15]-step*rateq);
            if(a[19]==1 && a[18]!=1)a[21]=p5_max(0.,a[21]-step*arateq);++counts[0];
            int event=p8g_event(step,bb,eb,mb,hb,rb,ab);
            if(local->event_used<16){
                long long *tr=local->transitions+5*local->event_used;tr[0]=(long long)a[19];tr[1]=g->ids[region];tr[2]=region;tr[3]=top+1;tr[4]=event;
                double *v=local->events+10*local->event_used;v[0]=event;v[1]=region;v[2]=a[2];v[3]=step;v[4]=isfinite(bb)?bb:-1.;v[5]=isfinite(eb)?eb:-1.;v[6]=isfinite(mb)?mb:-1.;v[7]=isfinite(hb)?hb:-1.;v[8]=isfinite(rb)?rb:-1.;v[9]=isfinite(ab)?ab:-1.;}++local->event_used;
            if(event==0) {a[2]=a[5]>0?g->faces[region+1]:g->faces[region];++counts[3];
                if(nr>=0 && nr<g->nl && g->ids[region]!=g->ids[nr]){
                    double info[5];p8g_material_boundary(a,region,nr,m,total,G1,p5_rule(ev),s,info);
                    if(int(a[16])==region)p8g2_hit(hits,11);++o->boundary_counts[0];o->boundary_counts[1]+=int(info[2]);o->boundary_counts[2]+=int(info[0]==0.);o->boundary_counts[3]+=int(info[3]);o->boundary_counts[4]+=int(int(a[16])==region);
                    int cell=p8g_upper(info[0],g->strength,17);++o->boundary_hist[cell];if(info[2])++o->boundary_omitted_hist[cell];
#if P8F_FAULT != 2
                    p8g2_hit(hits,9);q->boundary_pending=true;cached_ev=-1.;
#endif
                }else a[16]=nr;return;}
            if(event==1) {
                if(a[18]==1) {
                    p8g_sink(g,local,a[2],a[2],int(a[16]),a[6]);p5_track_local(hs,g->nc,a[2],a[6],.2,touch);p5_track_local(fine,g->nf,a[2],a[6],.05,touch+2);
                    dep+=a[6];a[6]=0.;phase=failed?P8G2_H:P8G2_E;return;
                }
                a[7]=p5_max(.01,a[7]-a[11]);
                if(a[7]<=.01+1e-14) {a[8]=a[9];a[9]=0.;a[10]=0.;a[18]=1.;return;}
                double prev=a[9];a[11]=p5_min(.05*a[7],a[7]-.01);double u=rng_uniform(s);a[10]=u*a[11];
                a[9]=a[11]-a[10];a[8]=prev+a[10];return;
            }
            if(event==2) {
                double mu=p8g_gs_mu_cached(ev,a[14],m,total,G1,s);
                double angle=rng_uniform(s),v[3];p5_rotate(a[3],a[4],a[5],mu,2*3.141592653589793*angle,v);
                for(int z=0;z<3;++z)a[3+z]=v[z];
                double u=rng_uniform(s),knew=p5_rule(ev);a[12]=a[13]+u*knew;a[13]=(1-u)*knew;a[14]=knew;
                ++counts[1];return;
            }
            if(event==3) {
                p5_energy_init(&state,s);double u=rng_uniform(s);a[15]=-log(u);if(a[6]<=2*.01)return;
                double W=p5_sample_W(a[6],t,s),dp[3],ds[3],k=(a[6]+2*M)/a[6];
                double mp=sqrt((a[6]-W)/(a[6]-W+2*M)*k),ms=sqrt(W/(W+2*M)*k);
                double angle=rng_uniform(s),phi=2*3.141592653589793*angle;
                p5_rotate(a[3],a[4],a[5],mp,phi,dp);p5_rotate(a[3],a[4],a[5],ms,phi+3.141592653589793,ds);
                a[6]-=W;for(int z=0;z<3;++z)a[3+z]=dp[z];p5_energy_init(&state,s);++counts[2];
                if(top+2>=P8G_STACK_CAP) {p8g2_hit(hits,14);a[6]+=W;failed=true;++counts[6];phase=failed?P8G2_H:P8G2_E;return;}
                p8g2_hit(hits,3);q->resume_top=top+1;q->resume_kind=3;stack[++top]=state;++top;p5_initial(stack+top,W,a[0],a[1],a[2],ds[0],ds[1],ds[2],int(a[16]),int(a[20]),t,s);
                if(top+1>o->maxdepth)o->maxdepth=top+1;phase=failed?P8G2_H:P8G2_E;return;
            }
            if(event==4) {
                p5_energy_init(&state,s);double T=a[6],u=rng_uniform(s);a[22]=-log(u);if(T<=.01)return;
                int excess;double photon_k=p8g_sample_k(T,m,s,&excess);

                o->brems[2]+=excess;if(photon_k>T)o->brems[3]+=1.;
                if(excess || photon_k>T) {failed=true;++counts[6];phase=failed?P8G2_H:P8G2_E;return;}
                double quantities[4];p8g_quantities(T,m,quantities);hn+=1.;de+=photon_k-quantities[1]/quantities[0];radiation+=photon_k;
                o->brems[0]+=1.;o->brems[1]+=photon_k;
                double angle=rng_uniform(s),theta=M/(T+M),v[3];
                p5_rotate(a[3],a[4],a[5],cos(theta),2*3.141592653589793*angle,v);
                if(top+2>=P8G_STACK_CAP) {p8g2_hit(hits,15);failed=true;++counts[6];phase=failed?P8G2_H:P8G2_E;return;}
                p8g2_hit(hits,4);q->resume_top=top+1;q->resume_kind=4;a[6]-=photon_k;p5_energy_init(&state,s);stack[++top]=state;++top;
                p5_photon(stack+top,photon_k,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));
                if(top+1>o->maxdepth)o->maxdepth=top+1;suspended=true;phase=failed?P8G2_H:P8G2_E;return;
            }
            if(a[19]==1 && event==5) {
                double T=a[6];if(T<=.01) {double u=rng_uniform(s);a[21]=-log(u);return;}
                if(top+3>=P8G_STACK_CAP) {failed=true;++counts[6];phase=failed?P8G2_H:P8G2_E;return;}
                double g[8];p5_flight_photons(T,a[3],a[4],a[5],t,s,g);++top;
                p5_photon(stack+top,g[0],a[0],a[1],a[2],g[1],g[2],g[3],int(a[16]),int(a[20]));++top;
                p5_photon(stack+top,g[4],a[0],a[1],a[2],g[5],g[6],g[7],int(a[16]),int(a[20]));
                p8g2_hit(hits,6);rest-=2*M;a[6]=0.;annihilated=true;++counts[14];if(a[17]==1)++counts[15];
                if(top+1>o->maxdepth)o->maxdepth=top+1;phase=failed?P8G2_H:P8G2_E;return;
            }
            p8g_sink(g,local,a[2],a[2],int(a[16]),a[6]);p5_track_local(hs,g->nc,a[2],a[6],.2,touch);p5_track_local(fine,g->nf,a[2],a[6],.05,touch+2);
            dep+=a[6];a[6]=0.;phase=failed?P8G2_H:P8G2_E;return;
}
P5_FN void p8g2_e(P8g2State *q,const double *data,const long long *meta,const double *constants,const P8gGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){
auto &phase=q->phase;
auto &top=q->top;
auto &steps=q->steps;
auto &touch=q->touch;
auto &failed=q->failed;
auto &annihilated=q->annihilated;
auto &suspended=q->suspended;
auto &M=q->M;
auto &hn=q->hn;
auto &expected_n=q->expected_n;
auto &de=q->de;
auto &radiation=q->radiation;
auto &source_energy=q->source_energy;
auto &dep=q->dep;
auto &escaped=q->escaped;
auto &rest=q->rest;
auto &positron_escape_rest=q->positron_escape_rest;
auto &cached_ev=q->cached_ev;
auto &arate=q->arate;
auto &S=q->S;
auto &rate=q->rate;
auto &total=q->total;
auto &G1=q->G1;
auto &brate=q->brate;
auto &bsoft=q->bsoft;
auto &stack=q->stack;
auto &state=q->state;
RngState *s=&q->rng;P8gLocal *local=&q->local;
double *hs=local->coarse,*fine=local->fine,*bins=local->bins;
double *a=state.a;long long *counts=o->counts;
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p;

#if P8G2_FAULT == 1
if(annihilated){annihilated=false;p8g2_hit(hits,30);}
#endif

        if(!suspended && !annihilated && a[19]==1 && a[6]==0 && a[16]>=0 && a[16]<g->nl) {
            if(top+3>=P8G_STACK_CAP) {p8g2_hit(hits,8);a[6]=0.;failed=true;++counts[6];phase=P8G2_H;return;}
            p8g2_hit(hits,5);double v[3];p5_annihilation(s,v);rest-=2*M;++counts[12];++top;
            p5_photon(stack+top,M,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));++top;
            p5_photon(stack+top,M,a[0],a[1],a[2],-v[0],-v[1],-v[2],int(a[16]),int(a[20]));
            if(top+1>o->maxdepth)o->maxdepth=top+1;
        }

phase=P8G2_P;
}
P5_FN void p8g2_h(P8g2State *q,const double *data,const long long *meta,const double *constants,const P8gGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){
auto &phase=q->phase;
auto &top=q->top;
auto &steps=q->steps;
auto &touch=q->touch;
auto &failed=q->failed;
auto &annihilated=q->annihilated;
auto &suspended=q->suspended;
auto &M=q->M;
auto &hn=q->hn;
auto &expected_n=q->expected_n;
auto &de=q->de;
auto &radiation=q->radiation;
auto &source_energy=q->source_energy;
auto &dep=q->dep;
auto &escaped=q->escaped;
auto &rest=q->rest;
auto &positron_escape_rest=q->positron_escape_rest;
auto &cached_ev=q->cached_ev;
auto &arate=q->arate;
auto &S=q->S;
auto &rate=q->rate;
auto &total=q->total;
auto &G1=q->G1;
auto &brate=q->brate;
auto &bsoft=q->bsoft;
auto &stack=q->stack;
auto &state=q->state;
RngState *s=&q->rng;P8gLocal *local=&q->local;
double *hs=local->coarse,*fine=local->fine,*bins=local->bins;
double *a=state.a;long long *counts=o->counts;
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p;
    double unresolved=0.;if(failed) {unresolved=state.a[6];for(int z=0;z<=top;++z)unresolved+=stack[z].a[6];}
    double p8_error=fabs(p8g_sum(bins,g->nb)-dep);o->p8_max[0]=p5_max(o->p8_max[0],p8_error);
#if P8F_FAULT != 6
    for(int z=0;z<g->nb;++z)o->p8_batches[batch*g->nb+z]+=bins[z];
#endif
    double residual=source_energy-dep-escaped-unresolved-rest;
    double coarse_error=fabs(p5_touched_sum(hs,touch)-dep),fine_error=fabs(p5_touched_sum(fine,touch+2)-dep);
    double dn=hn-expected_n;
    o->brems[4]+=dn;o->brems[5]+=dn*dn;o->brems[6]+=de;o->brems[7]+=de*de;o->brems[8]+=radiation;o->brems[9]+=radiation*radiation;
    o->maxima[0]=p5_max(o->maxima[0],fabs(residual)/source_energy);
    o->maxima[1]=p5_max(o->maxima[1],coarse_error/source_energy);o->maxima[2]=p5_max(o->maxima[2],fine_error/source_energy);
    if(fabs(residual)>source_energy*1e-12)++o->violations[0];if(coarse_error>source_energy*1e-9)++o->violations[1];if(fine_error>source_energy*1e-9)++o->violations[2];
    o->totals[0]+=source_energy;o->totals[1]+=dep;o->totals[2]+=escaped;o->totals[3]+=unresolved;o->totals[4]+=rest;o->totals[5]+=positron_escape_rest;
    for(int z=touch[0];z<=touch[1];++z)o->batches[batch*g->nc+z]+=hs[z];
    for(int z=touch[2];z<=touch[3];++z)o->fine_batches[batch*g->nf+z]+=fine[z];
    o->step_batches[batch]+=steps;local->dep=dep;local->calls=s->calls;
int f=g->nb+g->nc+g->nf;P8gRecord *r=&q->record;
r->d[f+21]=local->source;r->d[f+22]=local->dep;r->i[75]=local->calls;r->i[76]=diagnostic->used<2?diagnostic->used:2;r->i[77]=o->maxdepth;r->i[78]=local->nspec_index;
for(int j=0;j<g->nb;++j)r->i[79+j]=local->additions[j];phase=P8G2_D;
}
P5_FN void p8g2_advance(P8g2State *q,const double *data,const long long *meta,const double *constants,const P8gGeometry *g,int batch,int maxsteps,unsigned long long *hits,P8gOutput *shared=0){
P8gOutput out=p8g2_output(g,&q->record);out.maxdepth=q->maxdepth;
P8gOutput *o=shared?shared:&out;int f=g->nb+g->nc+g->nf;
P5Diagnostic diagnostic={q->record.d+f+23,2,q->diag_used,q->diag_overflow};
int before=q->phase;
if(before==P8G2_P)p8g2_p(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits);
else if(before==P8G2_C)p8g2_c(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits);
else if(before==P8G2_E)p8g2_e(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits);
else if(before==P8G2_H)p8g2_h(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits);
q->diag_used=diagnostic.used;q->diag_overflow=diagnostic.overflow;q->maxdepth=o->maxdepth;
p8g2_hit(hits,before*5+q->phase+40);
}
#endif
