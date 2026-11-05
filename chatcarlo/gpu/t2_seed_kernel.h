// G5 A2b fixed P9c specialization; shared t2 physics and Philox.
#ifndef T2_SEED_KERNEL_H
#define T2_SEED_KERNEL_H
#include "t2_seed_geometry.h"
#include "physics_set.h"
struct T2SeedLocal {double bins[220],coarse[160],fine[640];long long additions[220];double source,dep;uint64_t calls;int nspec_index,event_used,skipped;double events[160];long long transitions[80];};
struct T2SeedRecord {double d[1200];long long i[1200];};
P5_FN void t2_seed_sink(const T2SeedGeometry *g,T2SeedLocal *l,double z0,double z1,int region,double loss){/* Depth scoring is disabled; the six-site ledger is independent. */}
#ifndef T2S_FAULT
#define T2S_FAULT 0
#endif
enum T2SeedPhase {T2S_P,T2S_C,T2S_E,T2S_H,T2S_D};
#ifndef T2S_CHECK
#define T2S_CHECK 0
#endif
// hits is always caller-owned global device memory in CUDA builds.
P5_FN void t2_seed_hit(unsigned long long *hits,int j){
#if T2S_CHECK
if(!hits)return;
#ifdef P5_GPU
atomicAdd(hits+j,1ULL);
#else
++hits[j];
#endif
#endif
}
#define T2S_REGION_ROW 3435
struct T2SeedState {
P5State stack[P8G_STACK_CAP],state;
RngState rng;T2SeedLocal local;T2SeedRecord record;
int recomputes;int resume_top,resume_kind;bool boundary_pending;
// P7 region scores (3420) and field sums (15) of this history; caller-owned global row, or null.
// p6_add uses atomicAdd under CUDA, which requires global memory (never thread-local state).
double *region_out;double ledger_energy[6];long long sites[6];unsigned long long *field_additions;long long lateral;int rejected,missed;double coarse_deposit;
int zero_boundaries;
int phase,top,steps,touch[4],diag_used,maxdepth;long long diag_overflow;
bool failed,annihilated,suspended;
double M,hn,expected_n,de,radiation,source_energy,dep,escaped,rest,positron_escape_rest;
double cached_ev,arate,S,rate,total,G1,brate,bsoft;
};
// T2Seed: use the existing step-limit health counter and unresolved-energy path.
P5_FN bool t2_seed_zero_boundary(T2SeedState *q,double bb,bool boundary,long long *counts){
    q->zero_boundaries=boundary && bb==0.?q->zero_boundaries+1:0;
    if(q->zero_boundaries<1000)return false;
    q->failed=true;++counts[5];q->phase=T2S_H;return true;
}
// T2Seed: shared property refresh, also observed by the direct boundary probe.
P5_FN void t2_seed_refresh(T2SeedState *q,const P8gMaterial &m,double ev){
    double quantities[4];p8g_quantities(ev,m,quantities);q->brate=quantities[0];q->bsoft=quantities[2];double hardrate,hardstop;
    p8g_hard(ev,m,&hardrate,&hardstop);q->S=p8g_collision(ev,m)-hardstop+q->bsoft;q->rate=hardrate;
    p8g_table_rates(ev,m,&q->total,&q->G1);
    if(q->state.a[19]==1)q->arate=p8g_annihilation_rate(ev,m);q->cached_ev=ev;++q->recomputes;
}
// Initial slot setup touches only output values needed before the first begin.
// Stack entries are initialized exclusively by pushes, never by a blanket clear.
P5_FN void t2_seed_slot_init(T2SeedState *q,const T2SeedGeometry *g){
q->phase=T2S_P;for(int j=0;j<g->nb;++j)q->local.bins[j]=0.;
}
P5_FN P8gOutput t2_seed_output(const T2SeedGeometry *g,T2SeedRecord *r){
int b=g->nb,c=b+g->nc,f=c+g->nf;
P8gOutput o={r->d+b,r->d+c,r->d+f,r->d+f+6,r->d+f+9,r->d,r->d+f+19,
r->i,r->i+3,r->i+19,r->i+20,r->i+25,r->i+43,r->i+61,r->i+69,0};return o;
}
P5_FN void t2_seed_field(T2SeedState *q,const T2SeedGeometry *g,int site,const double *p0,const double *p1,double loss){
#if T2_SEED_FAULT == 1
return; // lost scoring and ledger, transport deposition remains
#endif
++q->sites[site-1];q->ledger_energy[site-1]+=loss;
double shifted[3]={p1[0]+.1,p1[1],p1[2]};
#if T2_SEED_FAULT == 2
p1=shifted;
#endif
for(int k=0;k<2;++k){double fraction=site==3?t2_seed_segment(g->grids+k,p0,p1,loss):t2_seed_point(g->grids+k,p1,loss);if(k==1)q->coarse_deposit+=loss*fraction;}
if(g->scoring_enabled==2 && q->region_out){P6Sink sink={q->region_out,q->region_out+3420,&g->scoring,1,q->field_additions};
if(site==3)p6_deposit_segment(&sink,0,p0,p1,loss);else p6_deposit_point(&sink,0,site,p1,loss);}
}
P5_FN bool t2_seed_ellipsoid(const double *s,const double *d){
    double axes[3]={2.25,1.75,1.5},A=0.,B=0.,C=-1.;for(int k=0;k<3;++k){double x=s[k]/axes[k],u=d[k]/axes[k];A+=u*u;B+=x*u;C+=x*x;}
    return B<0. && B*B-A*C>=0.;
}
P5_FN double t2_seed_entry(const T2SeedGeometry *g,const double *s,const double *d){
    double lo=0.,hi=INFINITY;
    if(g->outer){double A=0.,B=0.,C=-1.;for(int k=0;k<3;k+=2){double x=s[k]/g->radii[k],u=d[k]/g->radii[k];A+=u*u;B+=x*u;C+=x*x;}
        double D=B*B-A*C;if(A==0.){if(C>=0.)return INFINITY;}else{if(D<=0.)return INFINITY;double r=sqrt(D);lo=p5_max(lo,(-B-r)/A);hi=p5_min(hi,(-B+r)/A);}
        if(d[1]==0.){if(fabs(s[1])>=g->radii[1])return INFINITY;}else{double a=(-g->radii[1]-s[1])/d[1],b=(g->radii[1]-s[1])/d[1];lo=p5_max(lo,p5_min(a,b));hi=p5_min(hi,p5_max(a,b));}
    }else for(int k=0;k<3;++k){double a=k==g->axis?g->faces[0]:-g->half[k],b=k==g->axis?g->faces[g->nl]:g->half[k];
        if(d[k]==0.){if(s[k]<a||s[k]>=b)return INFINITY;}else{double t=(a-s[k])/d[k],u=(b-s[k])/d[k];lo=p5_max(lo,p5_min(t,u));hi=p5_min(hi,p5_max(t,u));}}
    return hi>lo?lo:INFINITY;
}
P5_FN void t2_seed_begin(T2SeedState *q,const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,uint64_t seed,uint64_t hid,unsigned long long *hits,const P5State *injected=0){
P8gOutput out=t2_seed_output(g,&q->record);P8gOutput *o=&out;
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
RngState *s=&q->rng;T2SeedLocal *local=&q->local;
double *hs=local->coarse,*fine=local->fine,*bins=local->bins;
double *a=state.a;long long *counts=o->counts;
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p;

#if T2S_FAULT == 2
if(q->phase==T2S_D)t2_seed_hit(hits,31);
#else
for(int j=0;j<g->nb;++j)local->bins[j]=0.;
#endif
for(int j=0;j<g->nb;++j)local->additions[j]=0;
for(int j=0;j<g->nc;++j)local->coarse[j]=0.;for(int j=0;j<g->nf;++j)local->fine[j]=0.;
for(int j=0;j<p8g_dwidth(g);++j)q->record.d[j]=0.;for(int j=0;j<p8g_iwidth(g);++j)q->record.i[j]=0;
q->recomputes=0;q->diag_used=q->diag_overflow=q->maxdepth=0;
if(q->region_out)for(int j=0;j<T2S_REGION_ROW;++j)q->region_out[j]=0.;for(int j=0;j<6;++j){q->sites[j]=0;q->ledger_energy[j]=0.;}q->lateral=0;q->rejected=q->missed=0;q->coarse_deposit=0.;
touch[0]=g->nc;touch[1]=-1;touch[2]=g->nf;touch[3]=-1;
M=t->constants[0];hn=expected_n=de=radiation=0.;
rng_init(s,seed,hid);
source_energy=g->energy;local->nspec_index=-1;
    if(g->nspec){double u1=rng_uniform(s),u2=rng_uniform(s);int j=0;while(g->cum[j]<u1)++j;
        double width=g->edges[j+1]-g->edges[j];source_energy=g->edges[j]+u2*width;int cell=int((source_energy-g->edges[j])/width*10);if(cell>9)cell=9;local->nspec_index=j*10+cell;}
    local->source=source_energy;local->event_used=0;local->skipped=0;
    double xyz[3],dir[3];double sn=sin(g->angle),cs=cos(g->angle),S0[3]={100.*sn,0.,-100.*cs};
    bool accepted=false;while(!accepted){double u1=rng_uniform(s),u2=rng_uniform(s),w=g->beam?4.:g->field_half;
        double aa=(2.*u1-1.)*w,bb=(2.*u2-1.)*w,p[3]={aa*cs,bb,aa*sn},norm=0.;
        for(int k=0;k<3;++k){dir[k]=p[k]-S0[k];norm+=dir[k]*dir[k];}norm=sqrt(norm);for(int k=0;k<3;++k)dir[k]/=norm;
        accepted=!g->beam||t2_seed_ellipsoid(S0,dir);if(!accepted)++q->rejected;
    }
    double entry=t2_seed_entry(g,S0,dir);if(!isfinite(entry)){q->missed=1;entry=0.;}
    for(int k=0;k<3;++k)xyz[k]=S0[k]+entry*dir[k];
    p5_photon(stack,source_energy,xyz[0],xyz[1],xyz[2],dir[0],dir[1],dir[2],0,0);
    if(injected){stack[0]=*injected;if(stack[0].a[19]!=0){double u=rng_uniform(s);stack[0].a[22]=-log(u);}if(stack[0].a[19]==1){double u=rng_uniform(s);stack[0].a[21]=-log(u);}}
    q->resume_top=-2;q->boundary_pending=false;top=0;steps=0;dep=escaped=rest=positron_escape_rest=0.;failed=false;if(q->missed && !injected){top=-1;escaped=source_energy;}
    
phase=T2S_P;
}
P5_FN void t2_seed_p(T2SeedState *q,const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){
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
RngState *s=&q->rng;T2SeedLocal *local=&q->local;
double *hs=local->coarse,*fine=local->fine,*bins=local->bins;
double *a=state.a;long long *counts=o->counts;
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p;

        if(top<0){phase=T2S_H;return;}
        if(top==q->resume_top){t2_seed_hit(hits,q->resume_kind==3?12:16);q->resume_top=-2;}state=stack[top--];
        q->zero_boundaries=0;
        if(a[19]==0) {
            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];phase=T2S_H;return;}
            double E=a[6];
            m=p8g_material(data,meta+p8g_mid(g->ids[int(a[16])])*136,constants);t=&m.p;
            if(E<G5_ELECTRON_PHOTON_CUT) {
                if(a[17]>=2)++o->fluorescence_counts[3];if(a[17]==3)++o->fluorescence_counts[7];
                t2_seed_field(q,g,1,a,a,E);t2_seed_sink(g,local,a[g->axis],a[g->axis],int(a[16]),E);
                // The fault macro for charged deposits must not change photons.
                
                dep+=E;t2_seed_hit(hits,10);return;
            }
            double c[5];p8g_coefficients(E,m,c);for(int z=0;z<5;++z)c[z]*=g->rho[int(a[16])];double mu=0.;for(int z=0;z<5;++z)mu+=c[z];
            T2SeedHit hit=t2_seed_boundary(a,g);double bb=hit.distance;int nr=hit.next;
            double flight_u=rng_uniform(s),tau=-log(flight_u),flight=tau/mu,length=p5_min(flight,bb);
            bool escaped_photon=false;
            for(;;){a[0]+=length*a[3];a[1]+=length*a[4];a[2]+=length*a[5];
                if(bb>flight)break;
                if(t2_seed_zero_boundary(q,bb,true,counts))return;
                if(nr<0 || nr>=g->nl+g->ns){t2_seed_snap(a,g,hit);if(hit.planes&~(1<<g->axis))++q->lateral;escaped_photon=true;break;}
                t2_seed_hit(hits,0);++o->branches[0];tau=p5_max(0.,tau-length*mu);
                t2_seed_photon_transition(a,g,hit,data,meta,constants,E,&m,c,&mu,&q->recomputes);t=&m.p;
                hit=t2_seed_boundary(a,g);bb=hit.distance;nr=hit.next;flight=tau/mu;length=p5_min(flight,bb);
            }
            if(escaped_photon){if(a[17]>=2)++o->fluorescence_counts[1];if(a[17]==3)++o->fluorescence_counts[5];escaped+=E;++counts[7];return;}

            double select=rng_uniform(s),r=select*mu;int process=0;
            while(process<4 && r>=c[process]) {r-=c[process];++process;}
            if(process<4)++counts[8+process];
            if(top+3>=P8G_STACK_CAP) {failed=true;++counts[6];phase=T2S_H;return;}
            if(process==4){double v[3];p5_rotate(a[3],a[4],a[5],g5_rayleigh(m.p.photon,E*1000,s),2*3.141592653589793*rng_uniform(s),v);
                ++top;p5_photon(stack+top,E,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));stack[top].a[17]=a[17];return;
            }
            if(process==0) {
                double p[8];p5_compton(E,a[3],a[4],a[5],t,s,p);
                ++top;p5_photon(stack+top,p[0],a[0],a[1],a[2],p[1],p[2],p[3],int(a[16]),int(a[20]));stack[top].a[17]=a[17];
                ++top;p5_initial(stack+top,p[4],a[0],a[1],a[2],p[5],p[6],p[7],int(a[16]),int(a[20]),t,s);
            } else if(process==1) {
                ++o->branches[1];if(a[17]>=2)++o->fluorescence_counts[2];if(a[17]==3)++o->fluorescence_counts[6];double T=E;
                if(g->binding){int element,shell;p8g_select_shell(E,m,s,&element,&shell);double binding=E,local_energy=E,emitted[32];long long trace[24];int events=0,used=0;
                    if(element>=0){binding=g5_binding(m,element,shell);used=p8g_relax(element,shell,m,s,g->fluorescence,emitted,&local_energy,trace,&events);}
                    if(used<0){failed=true;++counts[6];phase=T2S_H;return;}
                    T=E-binding;if(used>1)t2_seed_hit(hits,1);
                    if(top+used+2>=P8G_STACK_CAP){t2_seed_hit(hits,2);failed=true;++counts[6];phase=T2S_H;return;}
#if P8F_FAULT == 3
                    for(int j=0;j<used;++j)local_energy+=emitted[4*j];used=0;
#endif
                    dep+=local_energy;t2_seed_field(q,g,6,a,a,local_energy);t2_seed_sink(g,local,a[g->axis],a[g->axis],int(a[16]),local_energy);
                    
                    for(int j=0;j<used;++j){++top;p5_photon(stack+top,emitted[4*j],a[0],a[1],a[2],emitted[4*j+1],emitted[4*j+2],emitted[4*j+3],int(a[16]),int(a[20]));stack[top].a[17]=2;
                        #if G5_TEST_LEGACY_ELECTRON
                        for(int k=0;k<int(m.line_counts[element*4]);++k)if(emitted[4*j]==m.lines[(element*4*m.nline+k)*4])stack[top].a[17]=3;
#else
                        if(g5_k_line(m,element,emitted[4*j]))stack[top].a[17]=3;
#endif
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
cached_ev=-1.;arate=S=rate=total=G1=brate=bsoft=0.;annihilated=suspended=false;phase=T2S_C;
}
P5_FN void t2_seed_c(T2SeedState *q,const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){
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
RngState *s=&q->rng;T2SeedLocal *local=&q->local;
double *hs=local->coarse,*fine=local->fine,*bins=local->bins;
double *a=state.a;long long *counts=o->counts;
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p;

            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];phase=failed?T2S_H:T2S_E;return;}
            int region=int(a[16]);
            if(region<0 || region>=g->nl+g->ns) {escaped+=a[6];if(a[19]==1) {t2_seed_hit(hits,7);++counts[13];positron_escape_rest+=2*M;}phase=failed?T2S_H:T2S_E;return;}
            if(a[6]<=.01+1e-14 && a[18]!=1) {
                t2_seed_field(q,g,2,a,a,a[6]);t2_seed_sink(g,local,a[g->axis],a[g->axis],int(a[16]),a[6]);
                dep+=a[6];a[6]=0.;phase=failed?T2S_H:T2S_E;return;
            }
            m=p8g_material(data,meta+p8g_mid(g->ids[region])*136,constants);t=&m.p;
            double ev=p5_max(.01,a[7]);
            if(ev!=cached_ev) {t2_seed_hit(hits,13);if(q->boundary_pending){t2_seed_hit(hits,17);q->boundary_pending=false;}
                t2_seed_refresh(q,m,ev);
            }
            double Sq=S*g->rho[region],rateq=rate*g->rho[region],G1q=G1*g->rho[region],brateq=brate*g->rho[region],bsoftq=bsoft*g->rho[region],arateq=arate*g->rho[region];
            double eb=a[8]/Sq,mb=G1q>0?a[12]/G1q:INFINITY,hb=rateq>0?a[15]/rateq:INFINITY;
            if(a[18]==1) {mb=INFINITY;hb=INFINITY;}
            T2SeedHit hit=t2_seed_boundary(a,g);double bb=hit.distance;int nr=hit.next;++counts[4];
            double rb=brateq>0 && a[18]!=1?a[22]/brateq:INFINITY;
            double step=p5_min(p5_min(p5_min(p5_min(p5_min(eb,mb),hb),bb),rb),a[6]/Sq),ab=INFINITY;
            if(a[19]==1 && a[18]!=1) {ab=arateq>0?a[21]/arateq:INFINITY;step=p5_min(step,ab);}
            if(!isfinite(step) || step<0) {failed=true;++counts[6];phase=failed?T2S_H:T2S_E;return;}
            double z0=a[g->axis],point0[3]={a[0],a[1],a[2]};a[0]+=step*a[3];a[1]+=step*a[4];a[2]+=step*a[5];
            double loss=step*Sq;a[6]-=loss;dep+=loss;radiation+=step*bsoftq;
            if(a[18]!=1) {expected_n+=step*brateq;a[22]=p5_max(0.,a[22]-step*brateq);}
            t2_seed_field(q,g,3,point0,a,loss);t2_seed_sink(g,local,z0,a[g->axis],region,loss);
            a[8]=p5_max(0.,a[8]-loss);a[12]=p5_max(0.,a[12]-step*G1q);a[15]=p5_max(0.,a[15]-step*rateq);
            if(a[19]==1 && a[18]!=1)a[21]=p5_max(0.,a[21]-step*arateq);++counts[0];
            int event=p8g_event(step,bb,eb,mb,hb,rb,ab);
            if(t2_seed_zero_boundary(q,bb,event==0,counts))return;
            if(local->event_used<16){
                long long *tr=local->transitions+5*local->event_used;tr[0]=(long long)a[19];tr[1]=g->ids[region];tr[2]=region;tr[3]=top+1;tr[4]=event;
                double *v=local->events+10*local->event_used;v[0]=event;v[1]=region;v[2]=a[2];v[3]=step;v[4]=isfinite(bb)?bb:-1.;v[5]=isfinite(eb)?eb:-1.;v[6]=isfinite(mb)?mb:-1.;v[7]=isfinite(hb)?hb:-1.;v[8]=isfinite(rb)?rb:-1.;v[9]=isfinite(ab)?ab:-1.;}++local->event_used;
            if(event==0) {t2_seed_snap(a,g,hit);if(nr<0 && (hit.planes&~(1<<g->axis)))++q->lateral;++counts[3];
                if(nr>=0 && nr<g->nl+g->ns && g->ids[region]!=g->ids[nr]){
                    double info[5];t2_seed_transition(a,g,hit,m,total,G1,s,info);
                    if(int(a[16])==region)t2_seed_hit(hits,11);++o->boundary_counts[0];o->boundary_counts[1]+=int(info[2]);o->boundary_counts[2]+=int(info[0]==0.);o->boundary_counts[3]+=int(info[3]);o->boundary_counts[4]+=int(int(a[16])==region);
                    int cell=p8g_upper(info[0],g->strength,17);++o->boundary_hist[cell];if(info[2])++o->boundary_omitted_hist[cell];
#if P8F_FAULT != 2
                    t2_seed_hit(hits,9);q->boundary_pending=true;cached_ev=-1.;
#endif
                }else a[16]=nr;return;}
            if(event==1) {
                if(a[18]==1) {
                    t2_seed_field(q,g,4,a,a,a[6]);t2_seed_sink(g,local,a[g->axis],a[g->axis],int(a[16]),a[6]);
                    dep+=a[6];a[6]=0.;phase=failed?T2S_H:T2S_E;return;
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
                if(top+2>=P8G_STACK_CAP) {t2_seed_hit(hits,14);a[6]+=W;failed=true;++counts[6];phase=failed?T2S_H:T2S_E;return;}
                t2_seed_hit(hits,3);q->resume_top=top+1;q->resume_kind=3;stack[++top]=state;++top;p5_initial(stack+top,W,a[0],a[1],a[2],ds[0],ds[1],ds[2],int(a[16]),int(a[20]),t,s);
                if(top+1>o->maxdepth)o->maxdepth=top+1;phase=failed?T2S_H:T2S_E;return;
            }
            if(event==4) {
                p5_energy_init(&state,s);double T=a[6],u=rng_uniform(s);a[22]=-log(u);if(T<=.01)return;
                int excess;double photon_k=p8g_sample_k(T,m,s,&excess);

                o->brems[2]+=excess;if(photon_k>T)o->brems[3]+=1.;
                if(excess || photon_k>T) {failed=true;++counts[6];phase=failed?T2S_H:T2S_E;return;}
                double quantities[4];p8g_quantities(T,m,quantities);hn+=1.;de+=photon_k-quantities[1]/quantities[0];radiation+=photon_k;
                o->brems[0]+=1.;o->brems[1]+=photon_k;
                double angle=rng_uniform(s),theta=M/(T+M),v[3];
                p5_rotate(a[3],a[4],a[5],cos(theta),2*3.141592653589793*angle,v);
                if(top+2>=P8G_STACK_CAP) {t2_seed_hit(hits,15);failed=true;++counts[6];phase=failed?T2S_H:T2S_E;return;}
                t2_seed_hit(hits,4);q->resume_top=top+1;q->resume_kind=4;a[6]-=photon_k;p5_energy_init(&state,s);stack[++top]=state;++top;
                p5_photon(stack+top,photon_k,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));
                if(top+1>o->maxdepth)o->maxdepth=top+1;suspended=true;phase=failed?T2S_H:T2S_E;return;
            }
            if(a[19]==1 && event==5) {
                double T=a[6];if(T<=.01) {double u=rng_uniform(s);a[21]=-log(u);return;}
                if(top+3>=P8G_STACK_CAP) {failed=true;++counts[6];phase=failed?T2S_H:T2S_E;return;}
                double g[8];p5_flight_photons(T,a[3],a[4],a[5],t,s,g);++top;
                p5_photon(stack+top,g[0],a[0],a[1],a[2],g[1],g[2],g[3],int(a[16]),int(a[20]));++top;
                p5_photon(stack+top,g[4],a[0],a[1],a[2],g[5],g[6],g[7],int(a[16]),int(a[20]));
                t2_seed_hit(hits,6);rest-=2*M;a[6]=0.;annihilated=true;++counts[14];if(a[17]==1)++counts[15];
                if(top+1>o->maxdepth)o->maxdepth=top+1;phase=failed?T2S_H:T2S_E;return;
            }
            t2_seed_field(q,g,5,a,a,a[6]);t2_seed_sink(g,local,a[g->axis],a[g->axis],int(a[16]),a[6]);
            dep+=a[6];a[6]=0.;phase=failed?T2S_H:T2S_E;return;
}
P5_FN void t2_seed_e(T2SeedState *q,const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){
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
RngState *s=&q->rng;T2SeedLocal *local=&q->local;
double *hs=local->coarse,*fine=local->fine,*bins=local->bins;
double *a=state.a;long long *counts=o->counts;
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p;

#if T2S_FAULT == 1
if(annihilated){annihilated=false;t2_seed_hit(hits,30);}
#endif

        if(!suspended && !annihilated && a[19]==1 && a[6]==0 && a[16]>=0 && a[16]<g->nl+g->ns) {
            if(top+3>=P8G_STACK_CAP) {t2_seed_hit(hits,8);a[6]=0.;failed=true;++counts[6];phase=T2S_H;return;}
            t2_seed_hit(hits,5);double v[3];p5_annihilation(s,v);rest-=2*M;++counts[12];++top;
            p5_photon(stack+top,M,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));++top;
            p5_photon(stack+top,M,a[0],a[1],a[2],-v[0],-v[1],-v[2],int(a[16]),int(a[20]));
            if(top+1>o->maxdepth)o->maxdepth=top+1;
        }

phase=T2S_P;
}
P5_FN void t2_seed_h(T2SeedState *q,const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){
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
RngState *s=&q->rng;T2SeedLocal *local=&q->local;
double *hs=local->coarse,*fine=local->fine,*bins=local->bins;
double *a=state.a;long long *counts=o->counts;
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p;
    double unresolved=0.;if(failed) {unresolved=state.a[6];for(int z=0;z<=top;++z)unresolved+=stack[z].a[6];}
    double ledger=0.;for(int j=0;j<6;++j)ledger+=q->ledger_energy[j];double p8_error=fabs(ledger-dep);o->p8_max[0]=p5_max(o->p8_max[0],p8_error);
#if P8F_FAULT != 6
    for(int z=0;z<g->nb;++z)o->p8_batches[batch*g->nb+z]+=bins[z];
#endif
    double residual=source_energy-dep-escaped-unresolved-rest;
    double coarse_error=fabs(q->coarse_deposit-dep),fine_error=0.;
    double dn=hn-expected_n;
    o->brems[4]+=dn;o->brems[5]+=dn*dn;o->brems[6]+=de;o->brems[7]+=de*de;o->brems[8]+=radiation;o->brems[9]+=radiation*radiation;
    o->maxima[0]=p5_max(o->maxima[0],fabs(residual)/source_energy);
    o->maxima[1]=p5_max(o->maxima[1],coarse_error/source_energy);o->maxima[2]=p5_max(o->maxima[2],fine_error/source_energy);
    if(fabs(residual)>source_energy*1e-12)++o->violations[0];if(coarse_error>source_energy*1e-9)++o->violations[1];if(fine_error>source_energy*1e-9)++o->violations[2];
    o->totals[0]+=source_energy;o->totals[1]+=dep;o->totals[2]+=escaped;o->totals[3]+=unresolved;o->totals[4]+=rest;o->totals[5]+=positron_escape_rest;
    for(int z=touch[0];z<=touch[1];++z)o->batches[batch*g->nc+z]+=hs[z];
    for(int z=touch[2];z<=touch[3];++z)o->fine_batches[batch*g->nf+z]+=fine[z];
    o->step_batches[batch]+=steps;local->dep=dep;local->calls=s->calls;
int f=g->nb+g->nc+g->nf;T2SeedRecord *r=&q->record;
r->d[f+21]=local->source;r->d[f+22]=local->dep;r->i[75]=local->calls;r->i[76]=diagnostic->used<2?diagnostic->used:2;r->i[77]=o->maxdepth;r->i[78]=local->nspec_index;
for(int j=0;j<g->nb;++j)r->i[79+j]=local->additions[j];phase=T2S_D;
}
P5_FN void t2_seed_advance(T2SeedState *q,const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,int batch,int maxsteps,unsigned long long *hits,P8gOutput *shared=0){
P8gOutput out=t2_seed_output(g,&q->record);out.maxdepth=q->maxdepth;
P8gOutput *o=shared?shared:&out;int f=g->nb+g->nc+g->nf;
P5Diagnostic diagnostic={q->record.d+f+23,2,q->diag_used,q->diag_overflow};
int before=q->phase;
if(before==T2S_P)t2_seed_p(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits);
else if(before==T2S_C)t2_seed_c(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits);
else if(before==T2S_E)t2_seed_e(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits);
else if(before==T2S_H)t2_seed_h(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits);
q->diag_used=diagnostic.used;q->diag_overflow=diagnostic.overflow;q->maxdepth=o->maxdepth;
t2_seed_hit(hits,before*5+q->phase+40);
}
#endif
