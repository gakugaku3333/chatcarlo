#ifndef P9C_KERNEL_H // P9c: 3D geometry / dedicated state and scoring.
#define P9C_KERNEL_H // P9c: 3D geometry / dedicated state and scoring.
#include "p9c_geometry.h" // P9c: 3D geometry / dedicated state and scoring.
struct P9cLocal {double bins[220],coarse[160],fine[640];long long additions[220];double source,dep;uint64_t calls;int nspec_index,event_used,skipped;double events[160];long long transitions[80];}; // P9c: 3D geometry / dedicated state and scoring.
struct P9cRecord {double d[1200];long long i[1200];}; // P9c: 3D geometry / dedicated state and scoring.
P5_FN void p9c_sink(const P9cGeometry *g,P9cLocal *l,double z0,double z1,int region,double loss){/* Depth scoring is disabled; the six-site ledger is independent. */} // P9c: 3D geometry / dedicated state and scoring.
#ifndef P8G2_FAULT // P9c: copied namespace / extension.
#define P8G2_FAULT 0 // P9c: copied namespace / extension.
#endif // P9c: copied namespace / extension.
enum P9cPhase {P8G2_P,P8G2_C,P8G2_E,P8G2_H,P8G2_D}; // P9c: 3D geometry / dedicated state and scoring.
#ifndef P8G2_CHECK // P9c: copied namespace / extension.
#define P8G2_CHECK 0 // P9c: copied namespace / extension.
#endif // P9c: copied namespace / extension.
// hits is always caller-owned global device memory in CUDA builds. // P9c: copied namespace / extension.
P5_FN void p9c_hit(unsigned long long *hits,int j){ // P9c: 3D geometry / dedicated state and scoring.
#if P8G2_CHECK // P9c: copied namespace / extension.
if(!hits)return; // P9c: copied namespace / extension.
#ifdef P5_GPU // P9c: copied namespace / extension.
atomicAdd(hits+j,1ULL); // P9c: copied namespace / extension.
#else // P9c: copied namespace / extension.
++hits[j]; // P9c: copied namespace / extension.
#endif // P9c: copied namespace / extension.
#endif // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
struct P9cState { // P9c: 3D geometry / dedicated state and scoring.
P5State stack[P8G_STACK_CAP],state; // P9c: copied namespace / extension.
RngState rng;P9cLocal local;P9cRecord record; // P9c: 3D geometry / dedicated state and scoring.
int recomputes;int resume_top,resume_kind;bool boundary_pending; // P9c: copied namespace / extension.
double field_scores[3420],field_sums[15],ledger_energy[6];long long sites[6];unsigned long long *field_additions;long long lateral;int rejected,missed;double coarse_deposit; // P9c: 3D geometry / dedicated state and scoring.
int zero_boundaries; // P9c: consecutive zero-distance boundary watchdog.
int phase,top,steps,touch[4],diag_used,maxdepth;long long diag_overflow; // P9c: copied namespace / extension.
bool failed,annihilated,suspended; // P9c: copied namespace / extension.
double M,hn,expected_n,de,radiation,source_energy,dep,escaped,rest,positron_escape_rest; // P9c: copied namespace / extension.
double cached_ev,arate,S,rate,total,G1,brate,bsoft; // P9c: copied namespace / extension.
}; // P9c: copied namespace / extension.
// P9c: use the existing step-limit health counter and unresolved-energy path.
P5_FN bool p9c_zero_boundary(P9cState *q,double bb,bool boundary,long long *counts){
    q->zero_boundaries=boundary && bb==0.?q->zero_boundaries+1:0;
    if(q->zero_boundaries<1000)return false;
    q->failed=true;++counts[5];q->phase=P8G2_H;return true;
}
// P9c: shared property refresh, also observed by the direct boundary probe. // P9c: property observation.
P5_FN void p9c_refresh(P9cState *q,const P8gMaterial &m,double ev){ // P9c: property observation.
    double quantities[4];p8g_quantities(ev,m,quantities);q->brate=quantities[0];q->bsoft=quantities[2];double hardrate,hardstop; // P9c: property observation.
    p8g_hard(ev,m,&hardrate,&hardstop);q->S=p8g_collision(ev,m)-hardstop+q->bsoft;q->rate=hardrate; // P9c: property observation.
    p8g_table_rates(ev,m,&q->total,&q->G1); // P9c: property observation.
    if(q->state.a[19]==1)q->arate=p8g_annihilation_rate(ev,m);q->cached_ev=ev;++q->recomputes; // P9c: property observation.
} // P9c: property observation.
// Initial slot setup touches only output values needed before the first begin. // P9c: copied namespace / extension.
// Stack entries are initialized exclusively by pushes, never by a blanket clear. // P9c: copied namespace / extension.
P5_FN void p9c_slot_init(P9cState *q,const P9cGeometry *g){ // P9c: 3D geometry / dedicated state and scoring.
q->phase=P8G2_P;for(int j=0;j<g->nb;++j)q->local.bins[j]=0.; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN P8gOutput p9c_output(const P9cGeometry *g,P9cRecord *r){ // P9c: 3D geometry / dedicated state and scoring.
int b=g->nb,c=b+g->nc,f=c+g->nf; // P9c: copied namespace / extension.
P8gOutput o={r->d+b,r->d+c,r->d+f,r->d+f+6,r->d+f+9,r->d,r->d+f+19, // P9c: copied namespace / extension.
r->i,r->i+3,r->i+19,r->i+20,r->i+25,r->i+43,r->i+61,r->i+69,0};return o; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN void p9c_field(P9cState *q,const P9cGeometry *g,int site,const double *p0,const double *p1,double loss){ // P9c: 3D geometry / dedicated state and scoring.
++q->sites[site-1];q->ledger_energy[site-1]+=loss; // P9c: copied namespace / extension.
for(int k=0;k<2;++k){double fraction=site==3?p9c_segment(g->grids+k,p0,p1,loss):p9c_point(g->grids+k,p1,loss);if(k==1)q->coarse_deposit+=loss*fraction;} // P9c: copied namespace / extension.
if(g->scoring_enabled==2){P6Sink sink={q->field_scores,q->field_sums,&g->scoring,1,q->field_additions}; // P9c: copied namespace / extension.
if(site==3)p6_deposit_segment(&sink,0,p0,p1,loss);else p6_deposit_point(&sink,0,site,p1,loss);} // P9c: copied namespace / extension.
} // P9c: 3D geometry / dedicated state and scoring.
P5_FN bool p9c_ellipsoid(const double *s,const double *d){ // P9c: copied namespace / extension.
    double axes[3]={2.25,1.75,1.5},A=0.,B=0.,C=-1.;for(int k=0;k<3;++k){double x=s[k]/axes[k],u=d[k]/axes[k];A+=u*u;B+=x*u;C+=x*x;} // P9c: copied namespace / extension.
    return B<0. && B*B-A*C>=0.; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN double p9c_entry(const P9cGeometry *g,const double *s,const double *d){ // P9c: copied namespace / extension.
    double lo=0.,hi=INFINITY; // P9c: copied namespace / extension.
    if(g->outer){double A=0.,B=0.,C=-1.;for(int k=0;k<3;k+=2){double x=s[k]/g->radii[k],u=d[k]/g->radii[k];A+=u*u;B+=x*u;C+=x*x;} // P9c: copied namespace / extension.
        double D=B*B-A*C;if(A==0.){if(C>=0.)return INFINITY;}else{if(D<=0.)return INFINITY;double r=sqrt(D);lo=p5_max(lo,(-B-r)/A);hi=p5_min(hi,(-B+r)/A);} // P9c: copied namespace / extension.
        if(d[1]==0.){if(fabs(s[1])>=g->radii[1])return INFINITY;}else{double a=(-g->radii[1]-s[1])/d[1],b=(g->radii[1]-s[1])/d[1];lo=p5_max(lo,p5_min(a,b));hi=p5_min(hi,p5_max(a,b));} // P9c: copied namespace / extension.
    }else for(int k=0;k<3;++k){double a=k==g->axis?g->faces[0]:-g->half[k],b=k==g->axis?g->faces[g->nl]:g->half[k]; // P9c: copied namespace / extension.
        if(d[k]==0.){if(s[k]<a||s[k]>=b)return INFINITY;}else{double t=(a-s[k])/d[k],u=(b-s[k])/d[k];lo=p5_max(lo,p5_min(t,u));hi=p5_min(hi,p5_max(t,u));}} // P9c: copied namespace / extension.
    return hi>lo?lo:INFINITY; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN void p9c_begin(P9cState *q,const double *data,const long long *meta,const double *constants,const P9cGeometry *g,uint64_t seed,uint64_t hid,unsigned long long *hits,const P5State *injected=0){ // P9c: 3D geometry / dedicated state and scoring.
P8gOutput out=p9c_output(g,&q->record);P8gOutput *o=&out; // P9c: 3D geometry / dedicated state and scoring.
auto &phase=q->phase; // P9c: copied namespace / extension.
auto &top=q->top; // P9c: copied namespace / extension.
auto &steps=q->steps; // P9c: copied namespace / extension.
auto &touch=q->touch; // P9c: copied namespace / extension.
auto &failed=q->failed; // P9c: copied namespace / extension.
auto &annihilated=q->annihilated; // P9c: copied namespace / extension.
auto &suspended=q->suspended; // P9c: copied namespace / extension.
auto &M=q->M; // P9c: copied namespace / extension.
auto &hn=q->hn; // P9c: copied namespace / extension.
auto &expected_n=q->expected_n; // P9c: copied namespace / extension.
auto &de=q->de; // P9c: copied namespace / extension.
auto &radiation=q->radiation; // P9c: copied namespace / extension.
auto &source_energy=q->source_energy; // P9c: copied namespace / extension.
auto &dep=q->dep; // P9c: copied namespace / extension.
auto &escaped=q->escaped; // P9c: copied namespace / extension.
auto &rest=q->rest; // P9c: copied namespace / extension.
auto &positron_escape_rest=q->positron_escape_rest; // P9c: copied namespace / extension.
auto &cached_ev=q->cached_ev; // P9c: copied namespace / extension.
auto &arate=q->arate; // P9c: copied namespace / extension.
auto &S=q->S; // P9c: copied namespace / extension.
auto &rate=q->rate; // P9c: copied namespace / extension.
auto &total=q->total; // P9c: copied namespace / extension.
auto &G1=q->G1; // P9c: copied namespace / extension.
auto &brate=q->brate; // P9c: copied namespace / extension.
auto &bsoft=q->bsoft; // P9c: copied namespace / extension.
auto &stack=q->stack; // P9c: copied namespace / extension.
auto &state=q->state; // P9c: copied namespace / extension.
RngState *s=&q->rng;P9cLocal *local=&q->local; // P9c: 3D geometry / dedicated state and scoring.
double *hs=local->coarse,*fine=local->fine,*bins=local->bins; // P9c: copied namespace / extension.
double *a=state.a;long long *counts=o->counts; // P9c: copied namespace / extension.
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p; // P9c: copied namespace / extension.

#if P8G2_FAULT == 2 // P9c: copied namespace / extension.
if(q->phase==P8G2_D)p9c_hit(hits,31); // P9c: 3D geometry / dedicated state and scoring.
#else // P9c: copied namespace / extension.
for(int j=0;j<g->nb;++j)local->bins[j]=0.; // P9c: copied namespace / extension.
#endif // P9c: copied namespace / extension.
for(int j=0;j<g->nb;++j)local->additions[j]=0; // P9c: copied namespace / extension.
for(int j=0;j<g->nc;++j)local->coarse[j]=0.;for(int j=0;j<g->nf;++j)local->fine[j]=0.; // P9c: copied namespace / extension.
for(int j=0;j<p8g_dwidth(g);++j)q->record.d[j]=0.;for(int j=0;j<p8g_iwidth(g);++j)q->record.i[j]=0; // P9c: copied namespace / extension.
q->recomputes=0;q->diag_used=q->diag_overflow=q->maxdepth=0; // P9c: copied namespace / extension.
for(int j=0;j<3420;++j)q->field_scores[j]=0.;for(int j=0;j<15;++j)q->field_sums[j]=0.;for(int j=0;j<6;++j){q->sites[j]=0;q->ledger_energy[j]=0.;}q->lateral=0;q->rejected=q->missed=0;q->coarse_deposit=0.; // P9c: 3D geometry / dedicated state and scoring.
touch[0]=g->nc;touch[1]=-1;touch[2]=g->nf;touch[3]=-1; // P9c: copied namespace / extension.
M=t->constants[0];hn=expected_n=de=radiation=0.; // P9c: copied namespace / extension.
rng_init(s,seed,hid); // P9c: copied namespace / extension.
source_energy=g->energy;local->nspec_index=-1; // P9c: copied namespace / extension.
    if(g->nspec){double u1=rng_uniform(s),u2=rng_uniform(s);int j=0;while(g->cum[j]<u1)++j; // P9c: copied namespace / extension.
        double width=g->edges[j+1]-g->edges[j];source_energy=g->edges[j]+u2*width;int cell=int((source_energy-g->edges[j])/width*10);if(cell>9)cell=9;local->nspec_index=j*10+cell;} // P9c: copied namespace / extension.
    local->source=source_energy;local->event_used=0;local->skipped=0; // P9c: copied namespace / extension.
    double xyz[3],dir[3];double sn=sin(g->angle),cs=cos(g->angle),S0[3]={100.*sn,0.,-100.*cs}; // P9c: copied namespace / extension.
    bool accepted=false;while(!accepted){double u1=rng_uniform(s),u2=rng_uniform(s),w=g->beam?4.:g->field_half; // P9c: copied namespace / extension.
        double aa=(2.*u1-1.)*w,bb=(2.*u2-1.)*w,p[3]={aa*cs,bb,aa*sn},norm=0.; // P9c: copied namespace / extension.
        for(int k=0;k<3;++k){dir[k]=p[k]-S0[k];norm+=dir[k]*dir[k];}norm=sqrt(norm);for(int k=0;k<3;++k)dir[k]/=norm; // P9c: copied namespace / extension.
        accepted=!g->beam||p9c_ellipsoid(S0,dir);if(!accepted)++q->rejected; // P9c: copied namespace / extension.
    } // P9c: copied namespace / extension.
    double entry=p9c_entry(g,S0,dir);if(!isfinite(entry)){q->missed=1;entry=0.;} // P9c: copied namespace / extension.
    for(int k=0;k<3;++k)xyz[k]=S0[k]+entry*dir[k]; // P9c: copied namespace / extension.
    p5_photon(stack,source_energy,xyz[0],xyz[1],xyz[2],dir[0],dir[1],dir[2],0,0); // P9c: copied namespace / extension.
    if(injected){stack[0]=*injected;if(stack[0].a[19]!=0){double u=rng_uniform(s);stack[0].a[22]=-log(u);}if(stack[0].a[19]==1){double u=rng_uniform(s);stack[0].a[21]=-log(u);}} // P9c: copied namespace / extension.
    q->resume_top=-2;q->boundary_pending=false;top=0;steps=0;dep=escaped=rest=positron_escape_rest=0.;failed=false;if(q->missed && !injected){top=-1;escaped=source_energy;} // P9c: copied namespace / extension.
    
phase=P8G2_P; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN void p9c_p(P9cState *q,const double *data,const long long *meta,const double *constants,const P9cGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){ // P9c: 3D geometry / dedicated state and scoring.
auto &phase=q->phase; // P9c: copied namespace / extension.
auto &top=q->top; // P9c: copied namespace / extension.
auto &steps=q->steps; // P9c: copied namespace / extension.
auto &touch=q->touch; // P9c: copied namespace / extension.
auto &failed=q->failed; // P9c: copied namespace / extension.
auto &annihilated=q->annihilated; // P9c: copied namespace / extension.
auto &suspended=q->suspended; // P9c: copied namespace / extension.
auto &M=q->M; // P9c: copied namespace / extension.
auto &hn=q->hn; // P9c: copied namespace / extension.
auto &expected_n=q->expected_n; // P9c: copied namespace / extension.
auto &de=q->de; // P9c: copied namespace / extension.
auto &radiation=q->radiation; // P9c: copied namespace / extension.
auto &source_energy=q->source_energy; // P9c: copied namespace / extension.
auto &dep=q->dep; // P9c: copied namespace / extension.
auto &escaped=q->escaped; // P9c: copied namespace / extension.
auto &rest=q->rest; // P9c: copied namespace / extension.
auto &positron_escape_rest=q->positron_escape_rest; // P9c: copied namespace / extension.
auto &cached_ev=q->cached_ev; // P9c: copied namespace / extension.
auto &arate=q->arate; // P9c: copied namespace / extension.
auto &S=q->S; // P9c: copied namespace / extension.
auto &rate=q->rate; // P9c: copied namespace / extension.
auto &total=q->total; // P9c: copied namespace / extension.
auto &G1=q->G1; // P9c: copied namespace / extension.
auto &brate=q->brate; // P9c: copied namespace / extension.
auto &bsoft=q->bsoft; // P9c: copied namespace / extension.
auto &stack=q->stack; // P9c: copied namespace / extension.
auto &state=q->state; // P9c: copied namespace / extension.
RngState *s=&q->rng;P9cLocal *local=&q->local; // P9c: 3D geometry / dedicated state and scoring.
double *hs=local->coarse,*fine=local->fine,*bins=local->bins; // P9c: copied namespace / extension.
double *a=state.a;long long *counts=o->counts; // P9c: copied namespace / extension.
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p; // P9c: copied namespace / extension.

        if(top<0){phase=P8G2_H;return;} // P9c: copied namespace / extension.
        if(top==q->resume_top){p9c_hit(hits,q->resume_kind==3?12:16);q->resume_top=-2;}state=stack[top--]; // P9c: 3D geometry / dedicated state and scoring.
        q->zero_boundaries=0; // P9c: each new particle starts a fresh flight/step.
        if(a[19]==0) { // P9c: copied namespace / extension.
            ++steps;p5_record(diagnostic,&state,s->calls); // P9c: copied namespace / extension.
            if(steps>maxsteps) {failed=true;++counts[5];phase=P8G2_H;return;} // P9c: copied namespace / extension.
            double E=a[6]; // P9c: copied namespace / extension.
            m=p8g_material(data,meta+p8g_mid(g->ids[int(a[16])])*136,constants);t=&m.p; // P9c: copied namespace / extension.
            if(E<.01) { // P9c: copied namespace / extension.
                if(a[17]>=2)++o->fluorescence_counts[3];if(a[17]==3)++o->fluorescence_counts[7]; // P9c: copied namespace / extension.
                p9c_field(q,g,1,a,a,E);p9c_sink(g,local,a[g->axis],a[g->axis],int(a[16]),E); // P9c: 3D geometry / dedicated state and scoring.
                // The fault macro for charged deposits must not change photons. // P9c: copied namespace / extension.
                 // P9c: 3D geometry / dedicated state and scoring.
                dep+=E;p9c_hit(hits,10);return; // P9c: 3D geometry / dedicated state and scoring.
            } // P9c: copied namespace / extension.
            double c[4];p8g_coefficients(E,m,c);for(int z=0;z<4;++z)c[z]*=g->rho[int(a[16])];double mu=0.;for(int z=0;z<4;++z)mu+=c[z]; // P9c: copied namespace / extension.
            P9cHit hit=p9c_boundary(a,g);double bb=hit.distance;int nr=hit.next; // P9c: 3D geometry / dedicated state and scoring.
            double flight_u=rng_uniform(s),tau=-log(flight_u),flight=tau/mu,length=p5_min(flight,bb); // P9c: copied namespace / extension.
            bool escaped_photon=false; // P9c: copied namespace / extension.
            for(;;){a[0]+=length*a[3];a[1]+=length*a[4];a[2]+=length*a[5]; // P9c: copied namespace / extension.
                if(bb>flight)break; // P9c: copied namespace / extension.
                if(p9c_zero_boundary(q,bb,true,counts))return; // P9c: bound the optical-depth boundary loop.
                if(nr<0 || nr>=g->nl+g->ns){p9c_snap(a,g,hit);if(hit.planes&~(1<<g->axis))++q->lateral;escaped_photon=true;break;} // P9c: 3D geometry / dedicated state and scoring.
                p9c_hit(hits,0);++o->branches[0];tau=p5_max(0.,tau-length*mu); // P9c: unchanged optical-depth update.
                p9c_photon_transition(a,g,hit,data,meta,constants,E,&m,c,&mu,&q->recomputes);t=&m.p; // P9c: shared photon transition and observation.
                hit=p9c_boundary(a,g);bb=hit.distance;nr=hit.next;flight=tau/mu;length=p5_min(flight,bb); // P9c: 3D geometry / dedicated state and scoring.
            } // P9c: copied namespace / extension.
            if(escaped_photon){if(a[17]>=2)++o->fluorescence_counts[1];if(a[17]==3)++o->fluorescence_counts[5];escaped+=E;++counts[7];return;} // P9c: copied namespace / extension.

            double select=rng_uniform(s),r=select*mu;int process=0; // P9c: copied namespace / extension.
            while(process<3 && r>=c[process]) {r-=c[process];++process;} // P9c: copied namespace / extension.
            ++counts[8+process]; // P9c: copied namespace / extension.
            if(top+3>=P8G_STACK_CAP) {failed=true;++counts[6];phase=P8G2_H;return;} // P9c: copied namespace / extension.
            if(process==0) { // P9c: copied namespace / extension.
                double p[8];p5_compton(E,a[3],a[4],a[5],t,s,p); // P9c: copied namespace / extension.
                ++top;p5_photon(stack+top,p[0],a[0],a[1],a[2],p[1],p[2],p[3],int(a[16]),int(a[20]));stack[top].a[17]=a[17]; // P9c: copied namespace / extension.
                ++top;p5_initial(stack+top,p[4],a[0],a[1],a[2],p[5],p[6],p[7],int(a[16]),int(a[20]),t,s); // P9c: copied namespace / extension.
            } else if(process==1) { // P9c: copied namespace / extension.
                ++o->branches[1];if(a[17]>=2)++o->fluorescence_counts[2];if(a[17]==3)++o->fluorescence_counts[6];double T=E; // P9c: copied namespace / extension.
                if(g->binding){int element,shell;p8g_select_shell(E,m,s,&element,&shell);double binding=E,local_energy=E,emitted[32];long long trace[24];int events=0,used=0; // P9c: copied namespace / extension.
                    if(element>=0){binding=m.bindings[element*5+shell];used=p8g_relax(element,shell,m,s,g->fluorescence,emitted,&local_energy,trace,&events);} // P9c: copied namespace / extension.
                    T=E-binding;if(used>1)p9c_hit(hits,1); // P9c: 3D geometry / dedicated state and scoring.
                    if(top+used+2>=P8G_STACK_CAP){p9c_hit(hits,2);failed=true;++counts[6];phase=P8G2_H;return;} // P9c: 3D geometry / dedicated state and scoring.
#if P8F_FAULT == 3 // P9c: copied namespace / extension.
                    for(int j=0;j<used;++j)local_energy+=emitted[4*j];used=0; // P9c: copied namespace / extension.
#endif // P9c: copied namespace / extension.
                    dep+=local_energy;p9c_field(q,g,6,a,a,local_energy);p9c_sink(g,local,a[g->axis],a[g->axis],int(a[16]),local_energy); // P9c: 3D geometry / dedicated state and scoring.
                     // P9c: 3D geometry / dedicated state and scoring.
                    for(int j=0;j<used;++j){++top;p5_photon(stack+top,emitted[4*j],a[0],a[1],a[2],emitted[4*j+1],emitted[4*j+2],emitted[4*j+3],int(a[16]),int(a[20]));stack[top].a[17]=2; // P9c: copied namespace / extension.
                        for(int k=0;k<int(m.line_counts[element*4]);++k)if(emitted[4*j]==m.lines[(element*4*m.nline+k)*4])stack[top].a[17]=3; // P9c: copied namespace / extension.
                        ++o->fluorescence_counts[0];if(stack[top].a[17]==3)++o->fluorescence_counts[4];} // P9c: copied namespace / extension.
                } // P9c: copied namespace / extension.
                ++top;p5_initial(stack+top,T,a[0],a[1],a[2],a[3],a[4],a[5],int(a[16]),int(a[20]),t,s); // P9c: copied namespace / extension.
            } else { // P9c: copied namespace / extension.
                double p[10];p5_pair(E,a[3],a[4],a[5],t,s,p);rest+=2*M; // P9c: copied namespace / extension.
                P5State first,second; // P9c: copied namespace / extension.
                p5_initial(&first,p[0],a[0],a[1],a[2],p[1],p[2],p[3],int(a[16]),1,t,s);first.a[19]=p[4]; // P9c: copied namespace / extension.
                p5_initial(&second,p[5],a[0],a[1],a[2],p[6],p[7],p[8],int(a[16]),1,t,s);second.a[19]=p[9]; // P9c: copied namespace / extension.
                if(first.a[19]==1) {double u=rng_uniform(s);first.a[21]=-log(u);} // P9c: copied namespace / extension.
                if(second.a[19]==1) {double u=rng_uniform(s);second.a[21]=-log(u);} // P9c: copied namespace / extension.
                stack[++top]=first;stack[++top]=second; // P9c: copied namespace / extension.
            } // P9c: copied namespace / extension.
            if(top+1>o->maxdepth)o->maxdepth=top+1;return; // P9c: copied namespace / extension.
        } // P9c: copied namespace / extension.
cached_ev=-1.;arate=S=rate=total=G1=brate=bsoft=0.;annihilated=suspended=false;phase=P8G2_C; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN void p9c_c(P9cState *q,const double *data,const long long *meta,const double *constants,const P9cGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){ // P9c: 3D geometry / dedicated state and scoring.
auto &phase=q->phase; // P9c: copied namespace / extension.
auto &top=q->top; // P9c: copied namespace / extension.
auto &steps=q->steps; // P9c: copied namespace / extension.
auto &touch=q->touch; // P9c: copied namespace / extension.
auto &failed=q->failed; // P9c: copied namespace / extension.
auto &annihilated=q->annihilated; // P9c: copied namespace / extension.
auto &suspended=q->suspended; // P9c: copied namespace / extension.
auto &M=q->M; // P9c: copied namespace / extension.
auto &hn=q->hn; // P9c: copied namespace / extension.
auto &expected_n=q->expected_n; // P9c: copied namespace / extension.
auto &de=q->de; // P9c: copied namespace / extension.
auto &radiation=q->radiation; // P9c: copied namespace / extension.
auto &source_energy=q->source_energy; // P9c: copied namespace / extension.
auto &dep=q->dep; // P9c: copied namespace / extension.
auto &escaped=q->escaped; // P9c: copied namespace / extension.
auto &rest=q->rest; // P9c: copied namespace / extension.
auto &positron_escape_rest=q->positron_escape_rest; // P9c: copied namespace / extension.
auto &cached_ev=q->cached_ev; // P9c: copied namespace / extension.
auto &arate=q->arate; // P9c: copied namespace / extension.
auto &S=q->S; // P9c: copied namespace / extension.
auto &rate=q->rate; // P9c: copied namespace / extension.
auto &total=q->total; // P9c: copied namespace / extension.
auto &G1=q->G1; // P9c: copied namespace / extension.
auto &brate=q->brate; // P9c: copied namespace / extension.
auto &bsoft=q->bsoft; // P9c: copied namespace / extension.
auto &stack=q->stack; // P9c: copied namespace / extension.
auto &state=q->state; // P9c: copied namespace / extension.
RngState *s=&q->rng;P9cLocal *local=&q->local; // P9c: 3D geometry / dedicated state and scoring.
double *hs=local->coarse,*fine=local->fine,*bins=local->bins; // P9c: copied namespace / extension.
double *a=state.a;long long *counts=o->counts; // P9c: copied namespace / extension.
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p; // P9c: copied namespace / extension.

            ++steps;p5_record(diagnostic,&state,s->calls); // P9c: copied namespace / extension.
            if(steps>maxsteps) {failed=true;++counts[5];phase=failed?P8G2_H:P8G2_E;return;} // P9c: copied namespace / extension.
            int region=int(a[16]); // P9c: copied namespace / extension.
            if(region<0 || region>=g->nl+g->ns) {escaped+=a[6];if(a[19]==1) {p9c_hit(hits,7);++counts[13];positron_escape_rest+=2*M;}phase=failed?P8G2_H:P8G2_E;return;} // P9c: 3D geometry / dedicated state and scoring.
            if(a[6]<=.01+1e-14 && a[18]!=1) { // P9c: copied namespace / extension.
                p9c_field(q,g,2,a,a,a[6]);p9c_sink(g,local,a[g->axis],a[g->axis],int(a[16]),a[6]); // P9c: 3D geometry / dedicated state and scoring.
                dep+=a[6];a[6]=0.;phase=failed?P8G2_H:P8G2_E;return; // P9c: copied namespace / extension.
            } // P9c: copied namespace / extension.
            m=p8g_material(data,meta+p8g_mid(g->ids[region])*136,constants);t=&m.p; // P9c: copied namespace / extension.
            double ev=p5_max(.01,a[7]); // P9c: copied namespace / extension.
            if(ev!=cached_ev) {p9c_hit(hits,13);if(q->boundary_pending){p9c_hit(hits,17);q->boundary_pending=false;} // P9c: 3D geometry / dedicated state and scoring.
                p9c_refresh(q,m,ev); // P9c: shared unchanged property calculations.
            } // P9c: copied namespace / extension.
            double Sq=S*g->rho[region],rateq=rate*g->rho[region],G1q=G1*g->rho[region],brateq=brate*g->rho[region],bsoftq=bsoft*g->rho[region],arateq=arate*g->rho[region]; // P9c: copied namespace / extension.
            double eb=a[8]/Sq,mb=G1q>0?a[12]/G1q:INFINITY,hb=rateq>0?a[15]/rateq:INFINITY; // P9c: copied namespace / extension.
            if(a[18]==1) {mb=INFINITY;hb=INFINITY;} // P9c: copied namespace / extension.
            P9cHit hit=p9c_boundary(a,g);double bb=hit.distance;int nr=hit.next;++counts[4]; // P9c: 3D geometry / dedicated state and scoring.
            double rb=brateq>0 && a[18]!=1?a[22]/brateq:INFINITY; // P9c: copied namespace / extension.
            double step=p5_min(p5_min(p5_min(p5_min(p5_min(eb,mb),hb),bb),rb),a[6]/Sq),ab=INFINITY; // P9c: copied namespace / extension.
            if(a[19]==1 && a[18]!=1) {ab=arateq>0?a[21]/arateq:INFINITY;step=p5_min(step,ab);} // P9c: copied namespace / extension.
            if(!isfinite(step) || step<0) {failed=true;++counts[6];phase=failed?P8G2_H:P8G2_E;return;} // P9c: copied namespace / extension.
            double z0=a[g->axis],point0[3]={a[0],a[1],a[2]};a[0]+=step*a[3];a[1]+=step*a[4];a[2]+=step*a[5]; // P9c: 3D geometry / dedicated state and scoring.
            double loss=step*Sq;a[6]-=loss;dep+=loss;radiation+=step*bsoftq; // P9c: copied namespace / extension.
            if(a[18]!=1) {expected_n+=step*brateq;a[22]=p5_max(0.,a[22]-step*brateq);} // P9c: copied namespace / extension.
            p9c_field(q,g,3,point0,a,loss);p9c_sink(g,local,z0,a[g->axis],region,loss); // P9c: 3D geometry / dedicated state and scoring.
            a[8]=p5_max(0.,a[8]-loss);a[12]=p5_max(0.,a[12]-step*G1q);a[15]=p5_max(0.,a[15]-step*rateq); // P9c: copied namespace / extension.
            if(a[19]==1 && a[18]!=1)a[21]=p5_max(0.,a[21]-step*arateq);++counts[0]; // P9c: copied namespace / extension.
            int event=p8g_event(step,bb,eb,mb,hb,rb,ab); // P9c: copied namespace / extension.
            if(p9c_zero_boundary(q,bb,event==0,counts))return; // P9c: bound consecutive charged zero boundaries across advances.
            if(local->event_used<16){ // P9c: copied namespace / extension.
                long long *tr=local->transitions+5*local->event_used;tr[0]=(long long)a[19];tr[1]=g->ids[region];tr[2]=region;tr[3]=top+1;tr[4]=event; // P9c: copied namespace / extension.
                double *v=local->events+10*local->event_used;v[0]=event;v[1]=region;v[2]=a[2];v[3]=step;v[4]=isfinite(bb)?bb:-1.;v[5]=isfinite(eb)?eb:-1.;v[6]=isfinite(mb)?mb:-1.;v[7]=isfinite(hb)?hb:-1.;v[8]=isfinite(rb)?rb:-1.;v[9]=isfinite(ab)?ab:-1.;}++local->event_used; // P9c: copied namespace / extension.
            if(event==0) {p9c_snap(a,g,hit);if(nr<0 && (hit.planes&~(1<<g->axis)))++q->lateral;++counts[3]; // P9c: 3D geometry / dedicated state and scoring.
                if(nr>=0 && nr<g->nl+g->ns && g->ids[region]!=g->ids[nr]){ // P9c: 3D geometry / dedicated state and scoring.
                    double info[5];p9c_transition(a,g,hit,m,total,G1,s,info); // P9c: 3D geometry / dedicated state and scoring.
                    if(int(a[16])==region)p9c_hit(hits,11);++o->boundary_counts[0];o->boundary_counts[1]+=int(info[2]);o->boundary_counts[2]+=int(info[0]==0.);o->boundary_counts[3]+=int(info[3]);o->boundary_counts[4]+=int(int(a[16])==region); // P9c: 3D geometry / dedicated state and scoring.
                    int cell=p8g_upper(info[0],g->strength,17);++o->boundary_hist[cell];if(info[2])++o->boundary_omitted_hist[cell]; // P9c: copied namespace / extension.
#if P8F_FAULT != 2 // P9c: copied namespace / extension.
                    p9c_hit(hits,9);q->boundary_pending=true;cached_ev=-1.; // P9c: 3D geometry / dedicated state and scoring.
#endif // P9c: copied namespace / extension.
                }else a[16]=nr;return;} // P9c: copied namespace / extension.
            if(event==1) { // P9c: copied namespace / extension.
                if(a[18]==1) { // P9c: copied namespace / extension.
                    p9c_field(q,g,4,a,a,a[6]);p9c_sink(g,local,a[g->axis],a[g->axis],int(a[16]),a[6]); // P9c: 3D geometry / dedicated state and scoring.
                    dep+=a[6];a[6]=0.;phase=failed?P8G2_H:P8G2_E;return; // P9c: copied namespace / extension.
                } // P9c: copied namespace / extension.
                a[7]=p5_max(.01,a[7]-a[11]); // P9c: copied namespace / extension.
                if(a[7]<=.01+1e-14) {a[8]=a[9];a[9]=0.;a[10]=0.;a[18]=1.;return;} // P9c: copied namespace / extension.
                double prev=a[9];a[11]=p5_min(.05*a[7],a[7]-.01);double u=rng_uniform(s);a[10]=u*a[11]; // P9c: copied namespace / extension.
                a[9]=a[11]-a[10];a[8]=prev+a[10];return; // P9c: copied namespace / extension.
            } // P9c: copied namespace / extension.
            if(event==2) { // P9c: copied namespace / extension.
                double mu=p8g_gs_mu_cached(ev,a[14],m,total,G1,s); // P9c: copied namespace / extension.
                double angle=rng_uniform(s),v[3];p5_rotate(a[3],a[4],a[5],mu,2*3.141592653589793*angle,v); // P9c: copied namespace / extension.
                for(int z=0;z<3;++z)a[3+z]=v[z]; // P9c: copied namespace / extension.
                double u=rng_uniform(s),knew=p5_rule(ev);a[12]=a[13]+u*knew;a[13]=(1-u)*knew;a[14]=knew; // P9c: copied namespace / extension.
                ++counts[1];return; // P9c: copied namespace / extension.
            } // P9c: copied namespace / extension.
            if(event==3) { // P9c: copied namespace / extension.
                p5_energy_init(&state,s);double u=rng_uniform(s);a[15]=-log(u);if(a[6]<=2*.01)return; // P9c: copied namespace / extension.
                double W=p5_sample_W(a[6],t,s),dp[3],ds[3],k=(a[6]+2*M)/a[6]; // P9c: copied namespace / extension.
                double mp=sqrt((a[6]-W)/(a[6]-W+2*M)*k),ms=sqrt(W/(W+2*M)*k); // P9c: copied namespace / extension.
                double angle=rng_uniform(s),phi=2*3.141592653589793*angle; // P9c: copied namespace / extension.
                p5_rotate(a[3],a[4],a[5],mp,phi,dp);p5_rotate(a[3],a[4],a[5],ms,phi+3.141592653589793,ds); // P9c: copied namespace / extension.
                a[6]-=W;for(int z=0;z<3;++z)a[3+z]=dp[z];p5_energy_init(&state,s);++counts[2]; // P9c: copied namespace / extension.
                if(top+2>=P8G_STACK_CAP) {p9c_hit(hits,14);a[6]+=W;failed=true;++counts[6];phase=failed?P8G2_H:P8G2_E;return;} // P9c: 3D geometry / dedicated state and scoring.
                p9c_hit(hits,3);q->resume_top=top+1;q->resume_kind=3;stack[++top]=state;++top;p5_initial(stack+top,W,a[0],a[1],a[2],ds[0],ds[1],ds[2],int(a[16]),int(a[20]),t,s); // P9c: 3D geometry / dedicated state and scoring.
                if(top+1>o->maxdepth)o->maxdepth=top+1;phase=failed?P8G2_H:P8G2_E;return; // P9c: copied namespace / extension.
            } // P9c: copied namespace / extension.
            if(event==4) { // P9c: copied namespace / extension.
                p5_energy_init(&state,s);double T=a[6],u=rng_uniform(s);a[22]=-log(u);if(T<=.01)return; // P9c: copied namespace / extension.
                int excess;double photon_k=p8g_sample_k(T,m,s,&excess); // P9c: copied namespace / extension.

                o->brems[2]+=excess;if(photon_k>T)o->brems[3]+=1.; // P9c: copied namespace / extension.
                if(excess || photon_k>T) {failed=true;++counts[6];phase=failed?P8G2_H:P8G2_E;return;} // P9c: copied namespace / extension.
                double quantities[4];p8g_quantities(T,m,quantities);hn+=1.;de+=photon_k-quantities[1]/quantities[0];radiation+=photon_k; // P9c: copied namespace / extension.
                o->brems[0]+=1.;o->brems[1]+=photon_k; // P9c: copied namespace / extension.
                double angle=rng_uniform(s),theta=M/(T+M),v[3]; // P9c: copied namespace / extension.
                p5_rotate(a[3],a[4],a[5],cos(theta),2*3.141592653589793*angle,v); // P9c: copied namespace / extension.
                if(top+2>=P8G_STACK_CAP) {p9c_hit(hits,15);failed=true;++counts[6];phase=failed?P8G2_H:P8G2_E;return;} // P9c: 3D geometry / dedicated state and scoring.
                p9c_hit(hits,4);q->resume_top=top+1;q->resume_kind=4;a[6]-=photon_k;p5_energy_init(&state,s);stack[++top]=state;++top; // P9c: 3D geometry / dedicated state and scoring.
                p5_photon(stack+top,photon_k,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20])); // P9c: copied namespace / extension.
                if(top+1>o->maxdepth)o->maxdepth=top+1;suspended=true;phase=failed?P8G2_H:P8G2_E;return; // P9c: copied namespace / extension.
            } // P9c: copied namespace / extension.
            if(a[19]==1 && event==5) { // P9c: copied namespace / extension.
                double T=a[6];if(T<=.01) {double u=rng_uniform(s);a[21]=-log(u);return;} // P9c: copied namespace / extension.
                if(top+3>=P8G_STACK_CAP) {failed=true;++counts[6];phase=failed?P8G2_H:P8G2_E;return;} // P9c: copied namespace / extension.
                double g[8];p5_flight_photons(T,a[3],a[4],a[5],t,s,g);++top; // P9c: copied namespace / extension.
                p5_photon(stack+top,g[0],a[0],a[1],a[2],g[1],g[2],g[3],int(a[16]),int(a[20]));++top; // P9c: copied namespace / extension.
                p5_photon(stack+top,g[4],a[0],a[1],a[2],g[5],g[6],g[7],int(a[16]),int(a[20])); // P9c: copied namespace / extension.
                p9c_hit(hits,6);rest-=2*M;a[6]=0.;annihilated=true;++counts[14];if(a[17]==1)++counts[15]; // P9c: 3D geometry / dedicated state and scoring.
                if(top+1>o->maxdepth)o->maxdepth=top+1;phase=failed?P8G2_H:P8G2_E;return; // P9c: copied namespace / extension.
            } // P9c: copied namespace / extension.
            p9c_field(q,g,5,a,a,a[6]);p9c_sink(g,local,a[g->axis],a[g->axis],int(a[16]),a[6]); // P9c: 3D geometry / dedicated state and scoring.
            dep+=a[6];a[6]=0.;phase=failed?P8G2_H:P8G2_E;return; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN void p9c_e(P9cState *q,const double *data,const long long *meta,const double *constants,const P9cGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){ // P9c: 3D geometry / dedicated state and scoring.
auto &phase=q->phase; // P9c: copied namespace / extension.
auto &top=q->top; // P9c: copied namespace / extension.
auto &steps=q->steps; // P9c: copied namespace / extension.
auto &touch=q->touch; // P9c: copied namespace / extension.
auto &failed=q->failed; // P9c: copied namespace / extension.
auto &annihilated=q->annihilated; // P9c: copied namespace / extension.
auto &suspended=q->suspended; // P9c: copied namespace / extension.
auto &M=q->M; // P9c: copied namespace / extension.
auto &hn=q->hn; // P9c: copied namespace / extension.
auto &expected_n=q->expected_n; // P9c: copied namespace / extension.
auto &de=q->de; // P9c: copied namespace / extension.
auto &radiation=q->radiation; // P9c: copied namespace / extension.
auto &source_energy=q->source_energy; // P9c: copied namespace / extension.
auto &dep=q->dep; // P9c: copied namespace / extension.
auto &escaped=q->escaped; // P9c: copied namespace / extension.
auto &rest=q->rest; // P9c: copied namespace / extension.
auto &positron_escape_rest=q->positron_escape_rest; // P9c: copied namespace / extension.
auto &cached_ev=q->cached_ev; // P9c: copied namespace / extension.
auto &arate=q->arate; // P9c: copied namespace / extension.
auto &S=q->S; // P9c: copied namespace / extension.
auto &rate=q->rate; // P9c: copied namespace / extension.
auto &total=q->total; // P9c: copied namespace / extension.
auto &G1=q->G1; // P9c: copied namespace / extension.
auto &brate=q->brate; // P9c: copied namespace / extension.
auto &bsoft=q->bsoft; // P9c: copied namespace / extension.
auto &stack=q->stack; // P9c: copied namespace / extension.
auto &state=q->state; // P9c: copied namespace / extension.
RngState *s=&q->rng;P9cLocal *local=&q->local; // P9c: 3D geometry / dedicated state and scoring.
double *hs=local->coarse,*fine=local->fine,*bins=local->bins; // P9c: copied namespace / extension.
double *a=state.a;long long *counts=o->counts; // P9c: copied namespace / extension.
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p; // P9c: copied namespace / extension.

#if P8G2_FAULT == 1 // P9c: copied namespace / extension.
if(annihilated){annihilated=false;p9c_hit(hits,30);} // P9c: 3D geometry / dedicated state and scoring.
#endif // P9c: copied namespace / extension.

        if(!suspended && !annihilated && a[19]==1 && a[6]==0 && a[16]>=0 && a[16]<g->nl+g->ns) { // P9c: 3D geometry / dedicated state and scoring.
            if(top+3>=P8G_STACK_CAP) {p9c_hit(hits,8);a[6]=0.;failed=true;++counts[6];phase=P8G2_H;return;} // P9c: 3D geometry / dedicated state and scoring.
            p9c_hit(hits,5);double v[3];p5_annihilation(s,v);rest-=2*M;++counts[12];++top; // P9c: 3D geometry / dedicated state and scoring.
            p5_photon(stack+top,M,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));++top; // P9c: copied namespace / extension.
            p5_photon(stack+top,M,a[0],a[1],a[2],-v[0],-v[1],-v[2],int(a[16]),int(a[20])); // P9c: copied namespace / extension.
            if(top+1>o->maxdepth)o->maxdepth=top+1; // P9c: copied namespace / extension.
        } // P9c: copied namespace / extension.

phase=P8G2_P; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN void p9c_h(P9cState *q,const double *data,const long long *meta,const double *constants,const P9cGeometry *g,int batch,int maxsteps,P8gOutput *o,P5Diagnostic *diagnostic,unsigned long long *hits){ // P9c: 3D geometry / dedicated state and scoring.
auto &phase=q->phase; // P9c: copied namespace / extension.
auto &top=q->top; // P9c: copied namespace / extension.
auto &steps=q->steps; // P9c: copied namespace / extension.
auto &touch=q->touch; // P9c: copied namespace / extension.
auto &failed=q->failed; // P9c: copied namespace / extension.
auto &annihilated=q->annihilated; // P9c: copied namespace / extension.
auto &suspended=q->suspended; // P9c: copied namespace / extension.
auto &M=q->M; // P9c: copied namespace / extension.
auto &hn=q->hn; // P9c: copied namespace / extension.
auto &expected_n=q->expected_n; // P9c: copied namespace / extension.
auto &de=q->de; // P9c: copied namespace / extension.
auto &radiation=q->radiation; // P9c: copied namespace / extension.
auto &source_energy=q->source_energy; // P9c: copied namespace / extension.
auto &dep=q->dep; // P9c: copied namespace / extension.
auto &escaped=q->escaped; // P9c: copied namespace / extension.
auto &rest=q->rest; // P9c: copied namespace / extension.
auto &positron_escape_rest=q->positron_escape_rest; // P9c: copied namespace / extension.
auto &cached_ev=q->cached_ev; // P9c: copied namespace / extension.
auto &arate=q->arate; // P9c: copied namespace / extension.
auto &S=q->S; // P9c: copied namespace / extension.
auto &rate=q->rate; // P9c: copied namespace / extension.
auto &total=q->total; // P9c: copied namespace / extension.
auto &G1=q->G1; // P9c: copied namespace / extension.
auto &brate=q->brate; // P9c: copied namespace / extension.
auto &bsoft=q->bsoft; // P9c: copied namespace / extension.
auto &stack=q->stack; // P9c: copied namespace / extension.
auto &state=q->state; // P9c: copied namespace / extension.
RngState *s=&q->rng;P9cLocal *local=&q->local; // P9c: 3D geometry / dedicated state and scoring.
double *hs=local->coarse,*fine=local->fine,*bins=local->bins; // P9c: copied namespace / extension.
double *a=state.a;long long *counts=o->counts; // P9c: copied namespace / extension.
P8gMaterial m=p8g_material(data,meta,constants);const P5Tables *t=&m.p; // P9c: copied namespace / extension.
    double unresolved=0.;if(failed) {unresolved=state.a[6];for(int z=0;z<=top;++z)unresolved+=stack[z].a[6];} // P9c: copied namespace / extension.
    double ledger=0.;for(int j=0;j<6;++j)ledger+=q->ledger_energy[j];double p8_error=fabs(ledger-dep);o->p8_max[0]=p5_max(o->p8_max[0],p8_error); // P9c: copied namespace / extension.
#if P8F_FAULT != 6 // P9c: copied namespace / extension.
    for(int z=0;z<g->nb;++z)o->p8_batches[batch*g->nb+z]+=bins[z]; // P9c: copied namespace / extension.
#endif // P9c: copied namespace / extension.
    double residual=source_energy-dep-escaped-unresolved-rest; // P9c: copied namespace / extension.
    double coarse_error=fabs(q->coarse_deposit-dep),fine_error=0.; // P9c: copied namespace / extension.
    double dn=hn-expected_n; // P9c: copied namespace / extension.
    o->brems[4]+=dn;o->brems[5]+=dn*dn;o->brems[6]+=de;o->brems[7]+=de*de;o->brems[8]+=radiation;o->brems[9]+=radiation*radiation; // P9c: copied namespace / extension.
    o->maxima[0]=p5_max(o->maxima[0],fabs(residual)/source_energy); // P9c: copied namespace / extension.
    o->maxima[1]=p5_max(o->maxima[1],coarse_error/source_energy);o->maxima[2]=p5_max(o->maxima[2],fine_error/source_energy); // P9c: copied namespace / extension.
    if(fabs(residual)>source_energy*1e-12)++o->violations[0];if(coarse_error>source_energy*1e-9)++o->violations[1];if(fine_error>source_energy*1e-9)++o->violations[2]; // P9c: copied namespace / extension.
    o->totals[0]+=source_energy;o->totals[1]+=dep;o->totals[2]+=escaped;o->totals[3]+=unresolved;o->totals[4]+=rest;o->totals[5]+=positron_escape_rest; // P9c: copied namespace / extension.
    for(int z=touch[0];z<=touch[1];++z)o->batches[batch*g->nc+z]+=hs[z]; // P9c: copied namespace / extension.
    for(int z=touch[2];z<=touch[3];++z)o->fine_batches[batch*g->nf+z]+=fine[z]; // P9c: copied namespace / extension.
    o->step_batches[batch]+=steps;local->dep=dep;local->calls=s->calls; // P9c: copied namespace / extension.
int f=g->nb+g->nc+g->nf;P9cRecord *r=&q->record; // P9c: 3D geometry / dedicated state and scoring.
r->d[f+21]=local->source;r->d[f+22]=local->dep;r->i[75]=local->calls;r->i[76]=diagnostic->used<2?diagnostic->used:2;r->i[77]=o->maxdepth;r->i[78]=local->nspec_index; // P9c: copied namespace / extension.
for(int j=0;j<g->nb;++j)r->i[79+j]=local->additions[j];phase=P8G2_D; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN void p9c_advance(P9cState *q,const double *data,const long long *meta,const double *constants,const P9cGeometry *g,int batch,int maxsteps,unsigned long long *hits,P8gOutput *shared=0){ // P9c: 3D geometry / dedicated state and scoring.
P8gOutput out=p9c_output(g,&q->record);out.maxdepth=q->maxdepth; // P9c: 3D geometry / dedicated state and scoring.
P8gOutput *o=shared?shared:&out;int f=g->nb+g->nc+g->nf; // P9c: copied namespace / extension.
P5Diagnostic diagnostic={q->record.d+f+23,2,q->diag_used,q->diag_overflow}; // P9c: copied namespace / extension.
int before=q->phase; // P9c: copied namespace / extension.
if(before==P8G2_P)p9c_p(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits); // P9c: 3D geometry / dedicated state and scoring.
else if(before==P8G2_C)p9c_c(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits); // P9c: 3D geometry / dedicated state and scoring.
else if(before==P8G2_E)p9c_e(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits); // P9c: 3D geometry / dedicated state and scoring.
else if(before==P8G2_H)p9c_h(q,data,meta,constants,g,batch,maxsteps,o,&diagnostic,hits); // P9c: 3D geometry / dedicated state and scoring.
q->diag_used=diagnostic.used;q->diag_overflow=diagnostic.overflow;q->maxdepth=o->maxdepth; // P9c: copied namespace / extension.
p9c_hit(hits,before*5+q->phase+40); // P9c: 3D geometry / dedicated state and scoring.
} // P9c: copied namespace / extension.
#endif // P9c: copied namespace / extension.
