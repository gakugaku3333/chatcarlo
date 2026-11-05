P5_FN void t2_seed_probe_one(const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,const double *input,const double *fixed,int mode,uint64_t seed,uint64_t hid,double *out){
    double a[23];for(int j=0;j<23;++j){a[j]=input[j];out[j]=a[j];}RngState rng;rng_init(&rng,seed,hid);
    T2SeedHit hit=t2_seed_boundary(a,g);out[46]=hit.distance;out[47]=hit.next;
    if(isfinite(hit.distance)){for(int k=0;k<3;++k)a[k]+=hit.distance*a[3+k];t2_seed_snap(a,g,hit);for(int j=0;j<23;++j)out[51+j]=a[j];
        int old=int(a[16]);P8gMaterial m=p8g_material(data,meta+136*g->ids[old],constants);double info[5]={0.,1.,0.,0.,-1.},total,G1;p8g_table_rates(p5_max(.01,a[7]),m,&total,&G1);
        int photon_rebuilds=0;
        if(a[19]==0){if(hit.next>=0){double c[5],mu;t2_seed_photon_transition(a,g,hit,data,meta,constants,a[6],&m,c,&mu,&photon_rebuilds);}else a[16]=hit.next;}else t2_seed_transition(a,g,hit,m,total,G1,&rng,info,2.,mode==0?fixed:0);
        out[48]=(int(a[16])>=0 && int(a[16])<g->nl+g->ns)?g->ids[int(a[16])]:-1;
        out[49]=0.;
        if(int(a[16])>=0 && ((a[19]==0 && int(a[16])!=old) || (a[19]!=0 && hit.next>=0 && g->ids[old]!=g->ids[hit.next]))){
            P8gMaterial next=p8g_material(data,meta+136*g->ids[int(a[16])],constants);
            if(a[19]==0){out[49]=photon_rebuilds;}
            else{T2SeedState q;for(int j=0;j<23;++j)q.state.a[j]=a[j];q.recomputes=0;q.arate=0.;t2_seed_refresh(&q,next,p5_max(.01,a[7]));out[49]=q.recomputes;}
        }out[50]=rng.calls;
    }else{out[48]=g->ids[int(a[16])];out[49]=out[50]=0.;}
    for(int j=0;j<23;++j)out[23+j]=a[j];
}
