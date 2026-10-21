#ifndef P8F_KERNEL_H
#define P8F_KERNEL_H
#include "p8f_materials.h"
#ifndef P8F_STACK_CAP
#define P8F_STACK_CAP 4096
#endif
struct P8Geometry {const double *faces,*rho,*bins,*strength,*edges,*cum;const long long *ids,*owner;int nl,nb,nc,nf,nspec,binding,fluorescence;double energy;};
struct P8Output {double *batches,*fine_batches,*totals,*maxima,*brems,*p8_batches,*p8_max;long long *violations,*counts,*step_batches,*boundary_counts,*boundary_hist,*boundary_omitted_hist,*fluorescence_counts,*branches;int maxdepth;};
#ifdef P5_GPU
#define P8F_BIN_CAP 220
#define P8F_COARSE_CAP 108
#define P8F_FINE_CAP 430
#else
#define P8F_BIN_CAP 300
#define P8F_COARSE_CAP 150
#define P8F_FINE_CAP 600
#endif
struct P8Local {double bins[P8F_BIN_CAP],coarse[P8F_COARSE_CAP],fine[P8F_FINE_CAP];long long additions[P8F_BIN_CAP];double source,dep;uint64_t calls;int nspec_index,event_used,skipped;double events[160];long long transitions[80];};
P5_FN double p8f_sum(const double *x,int n){double v=0.;for(int j=0;j<n;++j)v+=x[j];return v;}
P5_FN void p8f_boundary(double z,double w,int region,const double *faces,double *distance,int *nr){*nr=region;if(w>0){*distance=p5_max(0.,(faces[region+1]-z)/w);*nr=region+1;}else if(w<0){*distance=p5_max(0.,(faces[region]-z)/w);*nr=region-1;}else *distance=INFINITY;}
P5_FN void p8f_score(double *score,int n,double z0,double z1,int region,double loss,const double *edges,const long long *owner,long long *adds=0){
    if(loss==0)return;double lo=p5_min(z0,z1),hi=p5_max(z0,z1);
    if(hi-lo<=1e-15){int j=p5_clip_bin(p8f_upper(z0,edges,n+1)-1,n);if(owner[j]!=region && j>0 && z0==edges[j] && owner[j-1]==region)--j;score[j]+=loss;if(adds)++adds[j];return;}
    int first=p5_max(0,p8f_upper(lo,edges,n+1)-1),last=p5_min(n-1,lower_bound(hi,edges,n+1));
    for(int j=first;j<=last;++j)if(owner[j]==region){double overlap=p5_max(0.,p5_min(hi,edges[j+1])-p5_max(lo,edges[j]));score[j]+=loss*overlap/(hi-lo);if(adds)++adds[j];}
}
P5_FN void p8f_sink(const P8Geometry *g,P8Local *l,double z0,double z1,int region,double loss){
#if P8F_FAULT == 5
    if(loss>0 && !l->skipped){l->skipped=1;return;}
#endif
    p8f_score(l->bins,g->nb,z0,z1,region,loss,g->bins,g->owner,l->additions);
}
P5_FN void p8f_material_boundary(double *a,int old,int nr,const P8Material &m,double total,double G1,double knew,RngState *s,double *info,double override_mu=2.){
    double strength=p5_max(0.,(a[14]-a[13])-a[12]),mu=1.;bool terminal=a[18]==1,omitted=false;
    if(!terminal){if(strength>=m.gs_K[0]){mu=override_mu>1?p5_gs_mu_cached(p5_max(.01,a[7]),strength,&m.p,total,G1,s):override_mu;double angle=rng_uniform(s),v[3];p5_rotate(a[3],a[4],a[5],mu,2*3.141592653589793*angle,v);for(int j=0;j<3;++j)a[j+3]=v[j];}else if(strength>0)omitted=true;}
    bool forward=(nr>old && a[5]>0)||(nr<old && a[5]<0);a[16]=forward?nr:old;
    if(!terminal){double u=rng_uniform(s);a[12]=u*knew;a[13]=(1-u)*knew;a[14]=knew;}
    info[0]=strength;info[1]=mu;info[2]=omitted?1.:0.;info[3]=terminal?1.:0.;info[4]=-1.;
}
P5_FN int p8f_event(double step,double bb,double eb,double mb,double hb,double rb,double ab){
    if(bb<=step+1e-13)return 0;if(eb<=step+1e-13)return 1;if(mb<=step+1e-13)return 2;if(hb<=step+1e-13)return 3;if(rb<=step+1e-13)return 4;if(ab<=step+1e-13)return 5;return 6;
}
P5_FN int p8f_mid(int id){
#if P8F_FAULT == 1
    return (id+1)%4;
#else
    return id;
#endif
}
P5_FN void p8f_history(const double *data,const long long *meta,const double *constants,const P8Geometry *g,RngState *s,int batch,int maxsteps,P8Output *o,P5Diagnostic *diagnostic,P8Local *local,const P5State *injected=0) {
    P5State stack[P8F_STACK_CAP],state;
    double *hs=local->coarse,*fine=local->fine,*bins=local->bins;int touch[4]={g->nc,-1,g->nf,-1};
    P8Material m=p8f_material(data,meta,constants);const P5Tables *t=&m.p;
    double M=t->constants[0],hn=0.,expected_n=0.,de=0.,radiation=0.;
    double source_energy=g->energy;local->nspec_index=-1;
    if(g->nspec){double u1=rng_uniform(s),u2=rng_uniform(s);int j=0;while(g->cum[j]<u1)++j;
        double width=g->edges[j+1]-g->edges[j];source_energy=g->edges[j]+u2*width;int cell=int((source_energy-g->edges[j])/width*10);if(cell>9)cell=9;local->nspec_index=j*10+cell;}
    local->source=source_energy;local->event_used=0;local->skipped=0;
    p5_photon(stack,source_energy,0.,0.,0.,0.,0.,1.,0,0);
    if(injected){stack[0]=*injected;if(stack[0].a[19]!=0){double u=rng_uniform(s);stack[0].a[22]=-log(u);}if(stack[0].a[19]==1){double u=rng_uniform(s);stack[0].a[21]=-log(u);}}
    int top=0,steps=0;double dep=0.,escaped=0.,rest=0.,positron_escape_rest=0.;bool failed=false;
    long long *counts=o->counts;
    while(top>=0) {
        state=stack[top--];double *a=state.a;
        if(a[19]==0) {
            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];break;}
            double E=a[6];
            m=p8f_material(data,meta+p8f_mid(g->ids[int(a[16])])*132,constants);t=&m.p;
            if(E<.01) {
                if(a[17]>=2)++o->fluorescence_counts[3];if(a[17]==3)++o->fluorescence_counts[7];
                p8f_sink(g,local,a[2],a[2],int(a[16]),E);
                // The fault macro for charged deposits must not change photons.
                p5_local(hs,g->nc,a[2],E,.2);p5_local(fine,g->nf,a[2],E,.05);
                int b0=p5_clip_bin(int(a[2]/.2),g->nc),b1=p5_clip_bin(int(a[2]/.05),g->nf);
                if(b0<touch[0])touch[0]=b0;if(b0>touch[1])touch[1]=b0;
                if(b1<touch[2])touch[2]=b1;if(b1>touch[3])touch[3]=b1;
                dep+=E;continue;
            }
            double c[4];p8f_coefficients(E,m,c);for(int z=0;z<4;++z)c[z]*=g->rho[int(a[16])];double mu=0.;for(int z=0;z<4;++z)mu+=c[z];
            double bb;int nr;p8f_boundary(a[2],a[5],int(a[16]),g->faces,&bb,&nr);
            double flight_u=rng_uniform(s),tau=-log(flight_u),flight=tau/mu,length=p5_min(flight,bb);
            bool escaped_photon=false;
            for(;;){a[0]+=length*a[3];a[1]+=length*a[4];a[2]+=length*a[5];
                if(bb>flight)break;
                if(nr<0 || nr>=g->nl){escaped_photon=true;break;}
                ++o->branches[0];tau=p5_max(0.,tau-length*mu);int region=int(a[16]);a[2]=a[5]>0?g->faces[region+1]:g->faces[region];a[16]=nr;
                m=p8f_material(data,meta+p8f_mid(g->ids[nr])*132,constants);t=&m.p;p8f_coefficients(E,m,c);mu=0.;
                for(int z=0;z<4;++z){c[z]*=g->rho[nr];mu+=c[z];}
                p8f_boundary(a[2],a[5],int(a[16]),g->faces,&bb,&nr);flight=tau/mu;length=p5_min(flight,bb);
            }
            if(escaped_photon){if(a[17]>=2)++o->fluorescence_counts[1];if(a[17]==3)++o->fluorescence_counts[5];escaped+=E;++counts[7];continue;}

            double select=rng_uniform(s),r=select*mu;int process=0;
            while(process<3 && r>=c[process]) {r-=c[process];++process;}
            ++counts[8+process];
            if(top+3>=P8F_STACK_CAP) {failed=true;++counts[6];break;}
            if(process==0) {
                double p[8];p5_compton(E,a[3],a[4],a[5],t,s,p);
                ++top;p5_photon(stack+top,p[0],a[0],a[1],a[2],p[1],p[2],p[3],int(a[16]),int(a[20]));stack[top].a[17]=a[17];
                ++top;p5_initial(stack+top,p[4],a[0],a[1],a[2],p[5],p[6],p[7],int(a[16]),int(a[20]),t,s);
            } else if(process==1) {
                ++o->branches[1];if(a[17]>=2)++o->fluorescence_counts[2];if(a[17]==3)++o->fluorescence_counts[6];double T=E;
                if(g->binding){int element,shell;p8f_select_shell(E,m,s,&element,&shell);double binding=E,local_energy=E,emitted[32];long long trace[24];int events=0,used=0;
                    if(element>=0){binding=m.bindings[element*5+shell];used=p8f_relax(element,shell,m,s,g->fluorescence,emitted,&local_energy,trace,&events);}
                    T=E-binding;
                    if(top+used+2>=P8F_STACK_CAP){failed=true;++counts[6];break;}
#if P8F_FAULT == 3
                    for(int j=0;j<used;++j)local_energy+=emitted[4*j];used=0;
#endif
                    dep+=local_energy;p8f_sink(g,local,a[2],a[2],int(a[16]),local_energy);
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
            if(top+1>o->maxdepth)o->maxdepth=top+1;continue;
        }
        double cached_ev=-1.,arate=0.,S=0.,rate=0.,total=0.,G1=0.,brate=0.,bsoft=0.;
        bool annihilated=false,suspended=false;
        for(;;) {
            ++steps;p5_record(diagnostic,&state,s->calls);
            if(steps>maxsteps) {failed=true;++counts[5];break;}
            int region=int(a[16]);
            if(region<0 || region>=g->nl) {escaped+=a[6];if(a[19]==1) {++counts[13];positron_escape_rest+=2*M;}break;}
            if(a[6]<=.01+1e-14 && a[18]!=1) {
                p8f_sink(g,local,a[2],a[2],int(a[16]),a[6]);p5_track_local(hs,g->nc,a[2],a[6],.2,touch);p5_track_local(fine,g->nf,a[2],a[6],.05,touch+2);
                dep+=a[6];a[6]=0.;break;
            }
            m=p8f_material(data,meta+p8f_mid(g->ids[region])*132,constants);t=&m.p;
            double ev=p5_max(.01,a[7]);
            if(ev!=cached_ev) {
                double q[4];p8f_quantities(ev,m,q);brate=q[0];bsoft=q[2];double hardrate,hardstop;
                p8f_hard(ev,m,&hardrate,&hardstop);S=p8f_collision(ev,m)-hardstop+bsoft;rate=hardrate;
                p5_table_rates(ev,t,&total,&G1);
                if(a[19]==1)arate=p8f_annihilation_rate(ev,m);cached_ev=ev;
            }
            double Sq=S*g->rho[region],rateq=rate*g->rho[region],G1q=G1*g->rho[region],brateq=brate*g->rho[region],bsoftq=bsoft*g->rho[region],arateq=arate*g->rho[region];
            double eb=a[8]/Sq,mb=G1q>0?a[12]/G1q:INFINITY,hb=rateq>0?a[15]/rateq:INFINITY;
            if(a[18]==1) {mb=INFINITY;hb=INFINITY;}
            double bb;int nr;p8f_boundary(a[2],a[5],region,g->faces,&bb,&nr);++counts[4];
            double rb=brateq>0 && a[18]!=1?a[22]/brateq:INFINITY;
            double step=p5_min(p5_min(p5_min(p5_min(p5_min(eb,mb),hb),bb),rb),a[6]/Sq),ab=INFINITY;
            if(a[19]==1 && a[18]!=1) {ab=arateq>0?a[21]/arateq:INFINITY;step=p5_min(step,ab);}
            if(!isfinite(step) || step<0) {failed=true;++counts[6];break;}
            double z0=a[2];a[0]+=step*a[3];a[1]+=step*a[4];a[2]+=step*a[5];
            double loss=step*Sq;a[6]-=loss;dep+=loss;radiation+=step*bsoftq;
            if(a[18]!=1) {expected_n+=step*brateq;a[22]=p5_max(0.,a[22]-step*brateq);}
            p8f_sink(g,local,z0,a[2],region,loss);p5_track_segment(hs,g->nc,z0,a[2],loss,.2,touch);p5_track_segment(fine,g->nf,z0,a[2],loss,.05,touch+2);
            a[8]=p5_max(0.,a[8]-loss);a[12]=p5_max(0.,a[12]-step*G1q);a[15]=p5_max(0.,a[15]-step*rateq);
            if(a[19]==1 && a[18]!=1)a[21]=p5_max(0.,a[21]-step*arateq);++counts[0];
            int event=p8f_event(step,bb,eb,mb,hb,rb,ab);
            if(local->event_used<16){
                long long *tr=local->transitions+5*local->event_used;tr[0]=(long long)a[19];tr[1]=g->ids[region];tr[2]=region;tr[3]=top+1;tr[4]=event;
                double *v=local->events+10*local->event_used;v[0]=event;v[1]=region;v[2]=a[2];v[3]=step;v[4]=isfinite(bb)?bb:-1.;v[5]=isfinite(eb)?eb:-1.;v[6]=isfinite(mb)?mb:-1.;v[7]=isfinite(hb)?hb:-1.;v[8]=isfinite(rb)?rb:-1.;v[9]=isfinite(ab)?ab:-1.;}++local->event_used;
            if(event==0) {a[2]=a[5]>0?g->faces[region+1]:g->faces[region];++counts[3];
                if(nr>=0 && nr<g->nl && g->ids[region]!=g->ids[nr]){
                    double info[5];p8f_material_boundary(a,region,nr,m,total,G1,p5_rule(ev),s,info);
                    ++o->boundary_counts[0];o->boundary_counts[1]+=int(info[2]);o->boundary_counts[2]+=int(info[0]==0.);o->boundary_counts[3]+=int(info[3]);o->boundary_counts[4]+=int(int(a[16])==region);
                    int cell=p8f_upper(info[0],g->strength,17);++o->boundary_hist[cell];if(info[2])++o->boundary_omitted_hist[cell];
#if P8F_FAULT != 2
                    cached_ev=-1.;
#endif
                }else a[16]=nr;continue;}
            if(event==1) {
                if(a[18]==1) {
                    p8f_sink(g,local,a[2],a[2],int(a[16]),a[6]);p5_track_local(hs,g->nc,a[2],a[6],.2,touch);p5_track_local(fine,g->nf,a[2],a[6],.05,touch+2);
                    dep+=a[6];a[6]=0.;break;
                }
                a[7]=p5_max(.01,a[7]-a[11]);
                if(a[7]<=.01+1e-14) {a[8]=a[9];a[9]=0.;a[10]=0.;a[18]=1.;continue;}
                double prev=a[9];a[11]=p5_min(.05*a[7],a[7]-.01);double u=rng_uniform(s);a[10]=u*a[11];
                a[9]=a[11]-a[10];a[8]=prev+a[10];continue;
            }
            if(event==2) {
                double mu=p5_gs_mu_cached(ev,a[14],t,total,G1,s);
                double angle=rng_uniform(s),v[3];p5_rotate(a[3],a[4],a[5],mu,2*3.141592653589793*angle,v);
                for(int z=0;z<3;++z)a[3+z]=v[z];
                double u=rng_uniform(s),knew=p5_rule(ev);a[12]=a[13]+u*knew;a[13]=(1-u)*knew;a[14]=knew;
                ++counts[1];continue;
            }
            if(event==3) {
                p5_energy_init(&state,s);double u=rng_uniform(s);a[15]=-log(u);if(a[6]<=2*.01)continue;
                double W=p5_sample_W(a[6],t,s),dp[3],ds[3],k=(a[6]+2*M)/a[6];
                double mp=sqrt((a[6]-W)/(a[6]-W+2*M)*k),ms=sqrt(W/(W+2*M)*k);
                double angle=rng_uniform(s),phi=2*3.141592653589793*angle;
                p5_rotate(a[3],a[4],a[5],mp,phi,dp);p5_rotate(a[3],a[4],a[5],ms,phi+3.141592653589793,ds);
                a[6]-=W;for(int z=0;z<3;++z)a[3+z]=dp[z];p5_energy_init(&state,s);++counts[2];
                if(top+2>=P8F_STACK_CAP) {a[6]+=W;failed=true;++counts[6];break;}
                stack[++top]=state;++top;p5_initial(stack+top,W,a[0],a[1],a[2],ds[0],ds[1],ds[2],int(a[16]),int(a[20]),t,s);
                if(top+1>o->maxdepth)o->maxdepth=top+1;break;
            }
            if(event==4) {
                p5_energy_init(&state,s);double T=a[6],u=rng_uniform(s);a[22]=-log(u);if(T<=.01)continue;
                int excess;double photon_k=p8f_sample_k(T,m,s,&excess);

                o->brems[2]+=excess;if(photon_k>T)o->brems[3]+=1.;
                if(excess || photon_k>T) {failed=true;++counts[6];break;}
                double q[4];p8f_quantities(T,m,q);hn+=1.;de+=photon_k-q[1]/q[0];radiation+=photon_k;
                o->brems[0]+=1.;o->brems[1]+=photon_k;
                double angle=rng_uniform(s),theta=M/(T+M),v[3];
                p5_rotate(a[3],a[4],a[5],cos(theta),2*3.141592653589793*angle,v);
                if(top+2>=P8F_STACK_CAP) {failed=true;++counts[6];break;}
                a[6]-=photon_k;p5_energy_init(&state,s);stack[++top]=state;++top;
                p5_photon(stack+top,photon_k,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));
                if(top+1>o->maxdepth)o->maxdepth=top+1;suspended=true;break;
            }
            if(a[19]==1 && event==5) {
                double T=a[6];if(T<=.01) {double u=rng_uniform(s);a[21]=-log(u);continue;}
                if(top+3>=P8F_STACK_CAP) {failed=true;++counts[6];break;}
                double g[8];p5_flight_photons(T,a[3],a[4],a[5],t,s,g);++top;
                p5_photon(stack+top,g[0],a[0],a[1],a[2],g[1],g[2],g[3],int(a[16]),int(a[20]));++top;
                p5_photon(stack+top,g[4],a[0],a[1],a[2],g[5],g[6],g[7],int(a[16]),int(a[20]));
                rest-=2*M;a[6]=0.;annihilated=true;++counts[14];if(a[17]==1)++counts[15];
                if(top+1>o->maxdepth)o->maxdepth=top+1;break;
            }
            p8f_sink(g,local,a[2],a[2],int(a[16]),a[6]);p5_track_local(hs,g->nc,a[2],a[6],.2,touch);p5_track_local(fine,g->nf,a[2],a[6],.05,touch+2);
            dep+=a[6];a[6]=0.;break;
        }
        if(failed)break;
        if(!suspended && !annihilated && a[19]==1 && a[6]==0 && a[16]>=0 && a[16]<g->nl) {
            if(top+3>=P8F_STACK_CAP) {a[6]=0.;failed=true;++counts[6];break;}
            double v[3];p5_annihilation(s,v);rest-=2*M;++counts[12];++top;
            p5_photon(stack+top,M,a[0],a[1],a[2],v[0],v[1],v[2],int(a[16]),int(a[20]));++top;
            p5_photon(stack+top,M,a[0],a[1],a[2],-v[0],-v[1],-v[2],int(a[16]),int(a[20]));
            if(top+1>o->maxdepth)o->maxdepth=top+1;
        }
    }
    double unresolved=0.;if(failed) {unresolved=state.a[6];for(int z=0;z<=top;++z)unresolved+=stack[z].a[6];}
    double p8_error=fabs(p8f_sum(bins,g->nb)-dep);o->p8_max[0]=p5_max(o->p8_max[0],p8_error);
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
}


// Identical record layout on CPU and GPU. No serialization math.
struct P8Record {double d[1200];long long i[1200];};
P5_FN int p8f_dwidth(const P8Geometry *g){return g->nb+g->nc+g->nf+37;}
P5_FN int p8f_iwidth(const P8Geometry *g){return 79+g->nb;}
P5_FN void p8f_record_history(const double *data,const long long *meta,const double *constants,const P8Geometry *g,RngState *rng,int maxsteps,P8Record *r,P8Local *l,const P5State *injected=0){
    for(int j=0;j<p8f_dwidth(g);++j)r->d[j]=0.;for(int j=0;j<p8f_iwidth(g);++j)r->i[j]=0;
    int b=g->nb,c=b+g->nc,f=c+g->nf;
    P8Output o={r->d+b,r->d+c,r->d+f,r->d+f+6,r->d+f+9,r->d,r->d+f+19,
        r->i,r->i+3,r->i+19,r->i+20,r->i+25,r->i+43,r->i+61,r->i+69,0};
    P5Diagnostic diag={r->d+f+23,2,0,0};
    p8f_history(data,meta,constants,g,rng,0,maxsteps,&o,&diag,l,injected);
    r->d[f+21]=l->source;r->d[f+22]=l->dep;r->i[75]=l->calls;r->i[76]=diag.used<2?diag.used:2;r->i[77]=o.maxdepth;r->i[78]=l->nspec_index;
    for(int j=0;j<g->nb;++j)r->i[79+j]=l->additions[j];
}
P5_FN void p8f_clear(const P8Geometry *g,P8Local *l){
#if P8F_FAULT != 7
    for(int j=0;j<g->nb;++j)l->bins[j]=0.;
#endif
    for(int j=0;j<g->nb;++j)l->additions[j]=0;
    for(int j=0;j<g->nc;++j)l->coarse[j]=0.;for(int j=0;j<g->nf;++j)l->fine[j]=0.;
}

#endif
