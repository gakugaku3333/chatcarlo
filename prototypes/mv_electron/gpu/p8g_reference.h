#ifndef P8G_REFERENCE_H
#define P8G_REFERENCE_H
// Independent scalar evaluation of the frozen expressions for gate 0b.
// Never used by initialization or transport. The five-column auxiliary record
// separately records the initializer's quotient/q/q^3/argument for diagnosis.
P5_FN double p8g_reference_value(const P8gMaterial &m,int s,int j){
 const P5Tables *t=&m.p;int shift=t->ratecols==3?1:0,i;
 switch(s){
 case 0:return log(t->brems_e[j]);
 case 1:if(j<3)return NAN;return log(t->brems_v[5*(j/3)+j%3]);
 case 2:{i=j;double e0=shift?t->rates[i*t->ratecols]:t->ts[i];double e1=shift?t->rates[(i+1)*t->ratecols]:t->ts[i+1];return log(e1/e0);}
 case 3:return log(t->rates[(j/2)*t->ratecols+shift+j%2]);
 case 4:return log(t->ts[j+1]/t->ts[j]);
 case 5:return log(m.pair_E[j]);
 case 6:case 7:{double th=m.thresholds[s-6];if(m.pair_E[j]<=th)return NAN;const double *vs=s==6?m.pair_n:m.pair_e;double q=1-th/m.pair_E[j];return log(vs[j]/((q*q)*q));}
 case 8:return log(m.stop_T[j]);
 case 9:if(!m.ncollision)return NAN;return log(m.stop_collision[j]);
 case 10:return log(m.photo_E[j]);
 case 11:case 14:return log(m.photo[j]);
 case 12:if(m.photo_E[j+1]==m.photo_E[j])return NAN;return log(m.photo_E[j+1]/m.photo_E[j]);
 case 13:if(m.photo_E[j+1]==m.photo_E[j])return NAN;return log(m.photo[j+1]/m.photo[j]);
 case 15:if(m.z[j]==1)return NAN;return log(1194.*pow(m.z[j],-2./3.));
 case 16:if(m.z[j]==1)return NAN;return log(184.15*pow(m.z[j],-1./3.));
 case 17:return log(m.z[j]);
 }return NAN;
}
#endif
