# P8g-1 log 呼出箇所と置換の根拠

凍結した P8f/P5 のソースを読み取り、関数の静的な呼出グラフを `p8f_history` からたどった一覧。
同じ行に複数の log がある場合は式を省略せずその行に併記する。`log1p` も保持の確認用に列挙する。
到達関数: `draw`, `interp`, `lower_bound`, `p5_annihilation`, `p5_bracket`, `p5_clip_bin`, `p5_compton`, `p5_energy_init`, `p5_flight_photons`, `p5_gs_mu_cached`, `p5_initial`, `p5_local`, `p5_max`, `p5_min`, `p5_moller_params`, `p5_pair`, `p5_pair_interp`, `p5_pair_small_total`, `p5_pair_value`, `p5_photon`, `p5_primitive0`, `p5_primitive1`, `p5_record`, `p5_rotate`, `p5_rule`, `p5_sample_W`, `p5_score_segment`, `p5_shape`, `p5_table_rates`, `p5_touched_sum`, `p5_track_local`, `p5_track_segment`, `p8f_annihilation_rate`, `p8f_boundary`, `p8f_choose_shell`, `p8f_coefficients`, `p8f_collision`, `p8f_event`, `p8f_hard`, `p8f_history`, `p8f_log_interp`, `p8f_material`, `p8f_material_boundary`, `p8f_mid`, `p8f_quantities`, `p8f_relax`, `p8f_sample_k`, `p8f_score`, `p8f_select_shell`, `p8f_shell_distribution`, `p8f_sink`, `p8f_sum`, `p8f_upper`, `philox`, `quantities`, `rng_uniform`, `uniform`

## 対象の各式と配列（材料ごと、元の添え字順）

| ID | 配列 | 元の式／呼出条件 |
|---|---|---|
| 0 | brems_E | quantities の log(e[i]), log(e[i+1]) |
| 1 | brems_V | quantities の log(v[5*i+k]), k=0,1,2。i==0 の3要素は NaN（分岐で読まない） |
| 2 | rates_ratio | p5_table_rates の log(e1/e0)、割ってからlog |
| 3 | rates_values | 同関数の log(rates[i*ratecols+shift+k]), k=0,1 |
| 4 | gs_ratio | p5_gs_mu_cached の log(ts[i+1]/ts[i]) |
| 5 | pair_E | p5_pair_interp の全 log(es[...])。最初の同値節点を選ぶ |
| 6,7 | pair_n,pair_e | p5_pair_value の log(vs[j*stride]/((q*q)*q)), q=1-th/es[j]。しきい値以下は NaN |
| 8,9 | stop_T,stop_collision | p8f_log_interp の log(x[...]), log(y[...])。collision がない場合ID9はNaN |
| 10,11 | photo_E,photo | 同補間の log(x[...]), log(y[...])。最後の同値節点を選ぶ |
| 12 | edge_E_ratio | coefficients の log(photo_E[i+1]/photo_E[i])。重複節点間はNaN |
| 13 | edge_photo_ratio | 吸収端の log(photo[right]/photo[left])。通常版 left=i,right=i+1。重複節点間はNaN |
| 14 | edge_photo_left | 吸収端の log(photo[left])。ID11と別配列 |
| 15,16 | shape_1194,shape_184 | p5_shape の log(1194.*pow(z,-2./3.)), log(184.15*pow(z,-1./3.))。z==1はNaN |
| 17 | shape_z | p5_shape の log(z) |

`p8f_log_interp` の `log(E)`、coefficients の `log(E/photo_E[i])`、quantities の `log(t)` と `log(t>cut?t:cut)`、table_rates の `log(T/e0)`、gs の `log(T/ts[i])`、pair の `log(E)`、shape の `log(d+.952)` は粒子に依存するので保持。
全ての exp、pow、四則演算と項の順はそのまま。比の log を log の差にしない。
`p8f_sample_k` の `log(.01)` は純定数として保持。`p5_sample_k` は到達しない。
`p8f_hard` から primitive0/1 の上端 .5 で呼ぶ log(.5/(1-.5)), log(.5) も純定数として保持。
表から入力される既存 log_stop_T は再計算しない。

## 全出現（凍結ソースの行番号）

| 元ソース | 関数 | 判定 | 行の式 | 理由 |
|---|---|---|---|---|
| `p5a0_common.h:57` | `quantities` | 対象・変数を区別（下記） | `double rad=interp(log(t>cut?t:cut),le,er,m);` | 表の項だけを置換。変数の項は保持。 |
| `p5a0_common.h:63` | `quantities` | 対象・変数を区別（下記） | `double u=(log(t)-log(e[i]))/(log(e[i+1])-log(e[i]));` | 表の項だけを置換。変数の項は保持。 |
| `p5a0_common.h:68` | `quantities` | 対象・変数を区別（下記） | `*rate=exp((1-u)*log(v[5*i])+u*log(v[5*(i+1)]));` | 表の項だけを置換。変数の項は保持。 |
| `p5a0_common.h:69` | `quantities` | 対象・変数を区別（下記） | `*soft=exp((1-u)*log(v[5*i+2])+u*log(v[5*(i+1)+2]));` | 表の項だけを置換。変数の項は保持。 |
| `p5a0_common.h:70` | `quantities` | 対象・変数を区別（下記） | `*discrete=exp((1-u)*log(v[5*i+1])+u*log(v[5*(i+1)+1]));` | 表の項だけを置換。変数の項は保持。 |
| `p5a0_common.h:75` | `maths` | 到達しない | `out[0]=log(u);out[1]=exp(-20*u);out[2]=sqrt(u);` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:20` | `p5_collision` | 到達しない | `double F=-1-b2+log((tau-d)*d)+tau/(tau-d)` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:21` | `p5_collision` | 到達しない | `+(d*d/2+(2*tau+1)*log1p(-d/tau))/(g*g);` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:22` | `p5_collision` | 到達しない | `double density=interp(log(T),t->loget,t->ed,t->ne);` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:24` | `p5_collision` | 到達しない | `return C/b2*(log(2*(tau+2)/(im*im))+F-density);` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:31` | `p5_primitive0` | 保持 | `return a*x-1/x+1/(1-x)-b*log(x/(1-x));` | 上端 .5 は純定数、下端 .01/T は変数。純定数と log1p は計画の対象外。 |
| `p5_kernel.h:34` | `p5_primitive1` | 保持 | `return a*x*x/2+log(x)+1/(1-x)+(1+b)*log1p(-x);` | 上端 .5 は純定数、下端 .01/T は変数。純定数と log1p は計画の対象外。 |
| `p5_kernel.h:58` | `p5_kn_total` | 到達しない | `double x=E/t->constants[0],L=log1p(2*x),one=1+2*x;` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:64` | `p5_log_interp` | 到達しない | `// np.interp searches the LOG grid, including ties after rounding log().` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:65` | `p5_log_interp` | 到達しない | `double x=log(E);` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:66` | `p5_log_interp` | 到達しない | `if(x<log(es[0]))return exp(log(vs[0]));` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:67` | `p5_log_interp` | 到達しない | `if(x>log(es[n-1]))return exp(log(vs[(n-1)*stride]));` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:69` | `p5_log_interp` | 到達しない | `while(lo<hi) {int m=lo+(hi-lo)/2;if(log(es[m])<x)lo=m+1;else hi=m;}` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:71` | `p5_log_interp` | 到達しない | `if(j<n && x==log(es[j]))return exp(log(vs[j*stride]));` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:73` | `p5_log_interp` | 到達しない | `double y0=log(vs[j*stride]),y1=log(vs[(j+1)*stride]);` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:74` | `p5_log_interp` | 到達しない | `double slope=(y1-y0)/(log(es[j+1])-log(es[j]));` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:75` | `p5_log_interp` | 到達しない | `return exp(slope*(x-log(es[j]))+y0);` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:78` | `p5_pair_value` | 対象・変数を区別（下記） | `double q=1-th/es[j];return log(vs[j*stride]/((q*q)*q));` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:83` | `p5_pair_interp` | 対象・変数を区別（下記） | `double x=log(E),y;` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:84` | `p5_pair_interp` | 対象・変数を区別（下記） | `if(x<log(es[start]))y=p5_pair_value(start,es,vs,stride,th);` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:85` | `p5_pair_interp` | 対象・変数を区別（下記） | `else if(x>log(es[n-1]))y=p5_pair_value(n-1,es,vs,stride,th);` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:88` | `p5_pair_interp` | 対象・変数を区別（下記） | `while(lo<hi) {int m=lo+(hi-lo)/2;if(log(es[m])<x)lo=m+1;else hi=m;}` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:90` | `p5_pair_interp` | 対象・変数を区別（下記） | `if(j<n && x==log(es[j]))y=p5_pair_value(j,es,vs,stride,th);` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:94` | `p5_pair_interp` | 対象・変数を区別（下記） | `double slope=(y1-y0)/(log(es[j+1])-log(es[j]));` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:95` | `p5_pair_interp` | 対象・変数を区別（下記） | `y=slope*(x-log(es[j]))+y0;` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:113` | `p5_table_rates` | 対象・変数を区別（下記） | `double a=log(T/e0)/log(e1/e0);` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:114` | `p5_table_rates` | 対象・変数を区別（下記） | `*total=exp((1-a)*log(t->rates[i*t->ratecols+shift])+a*log(t->rates[(i+1)*t->ratecols+shift]));` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:115` | `p5_table_rates` | 対象・変数を区別（下記） | `*G1=exp((1-a)*log(t->rates[i*t->ratecols+shift+1])+a*log(t->rates[(i+1)*t->ratecols+shift+1]));` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:126` | `p5_gs_mu_cached` | 対象・変数を区別（下記） | `double a=log(T/t->ts[i])/log(t->ts[i+1]/t->ts[i]);int j,h;double b,c;` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:160` | `p5_compton` | 保持 | `double M=t->constants[0],qmin=1/(1+2*E/M),a=log(1/qmin),b=(1-qmin*qmin)/2,q,mu;` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p5_kernel.h:188` | `p5_pair_small_total` | 保持 | `else rej=v[3]+v[4]*log(delta+v[5]);` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p5_kernel.h:207` | `p5_annihilation_rate` | 到達しない | `*((g*g+4*g+1)/(g*g-1)*log(g+p)-(g+3)/p);` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:212` | `p5_flight_photons` | 保持 | `double width=-log(lo/(1-lo)),A=g*g+4*g+1,z;` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p5_kernel.h:233` | `p5_shape` | 対象・変数を区別（下記） | `double xi=z==1?6.144/(5.31-fc):log(1194.*pow(z,-2./3.))/(log(184.15*pow(z,-1./3.))-fc);` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:234` | `p5_shape` | 対象・変数を区別（下記） | `double xl=4./3.*log(z),d=136.*pow(z,-1./3.)*M/E*k/rem;` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:238` | `p5_shape` | 対象・変数を区別（下記） | `else {s1=21.12-4.184*log(d+.952)-xl;s2=s1;}` | 表の項だけを置換。変数の項は保持。 |
| `p5_kernel.h:249` | `p5_sample_k` | 到達しない | `double q[4];p5_quantities(T,t,q);double envelope=q[3],lo=log(.01),width=log(T/.01);` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:291` | `p5_initial` | 保持 | `double hard=rng_uniform(s);a[15]=-log(hard);` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p5_kernel.h:292` | `p5_initial` | 保持 | `a[19]=-1.;double br=rng_uniform(s);a[22]=-log(br);a[0]=x;a[1]=y;a[20]=ancestry;` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p5_kernel.h:350` | `history` | 到達しない | `double flight_u=rng_uniform(s),tau=-log(flight_u),flight=tau/mu,length=p5_min(flight,bb);` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:371` | `history` | 到達しない | `if(first.a[19]==1) {double u=rng_uniform(s);first.a[21]=-log(u);}` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:372` | `history` | 到達しない | `if(second.a[19]==1) {double u=rng_uniform(s);second.a[21]=-log(u);}` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:428` | `history` | 到達しない | `p5_energy_init(&state,s);double u=rng_uniform(s);a[15]=-log(u);if(a[6]<=2*.01)continue;` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:439` | `history` | 到達しない | `p5_energy_init(&state,s);double T=a[6],u=rng_uniform(s);a[22]=-log(u);if(T<=.01)continue;` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p5_kernel.h:456` | `history` | 到達しない | `double T=a[6];if(T<=.01) {double u=rng_uniform(s);a[21]=-log(u);continue;}` | P8fの履歴の呼出グラフ外。変更しない。 |
| `p8f_materials.h:38` | `p8f_log_interp` | 対象・変数を区別（下記） | `double l=log(E);if(l<log(x[0]))return exp(log(y[0]));if(l>log(x[n-1]))return exp(log(y[n-1]));` | 表の項だけを置換。変数の項は保持。 |
| `p8f_materials.h:39` | `p8f_log_interp` | 対象・変数を区別（下記） | `int lo=0,hi=n;while(lo<hi){int j=(lo+hi)/2;if(l<log(x[j]))hi=j;else lo=j+1;}` | 表の項だけを置換。変数の項は保持。 |
| `p8f_materials.h:40` | `p8f_log_interp` | 対象・変数を区別（下記） | `int j=lo-1;if(l==log(x[j]) &#124;&#124; j==n-1)return exp(log(y[j]));` | 表の項だけを置換。変数の項は保持。 |
| `p8f_materials.h:41` | `p8f_log_interp` | 対象・変数を区別（下記） | `double slope=(log(y[j+1])-log(y[j]))/(log(x[j+1])-log(x[j]));` | 表の項だけを置換。変数の項は保持。 |
| `p8f_materials.h:42` | `p8f_log_interp` | 対象・変数を区別（下記） | `return exp(slope*(l-log(x[j]))+log(y[j]));` | 表の項だけを置換。変数の項は保持。 |
| `p8f_materials.h:47` | `p8f_collision` | 保持 | `double F=-1-b2+log((tau-d)*d)+tau/(tau-d)+(d*d/2+(2*tau+1)*log1p(-d/tau))/(g*g);` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_materials.h:49` | `p8f_collision` | 保持 | `return *m.c/b2*(log(2*(tau+2)/(im*im))+F-interp(log(T),m.log_stop_T,m.stop_delta,m.nstop));` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_materials.h:60` | `p8f_annihilation_rate` | 保持 | `double cross=3.141592653589793*(RE*RE)/(g+1)*((g*g+4*g+1)/(g*g-1)*log(g+p)-(g+3)/p);return *m.ne*cross;` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_materials.h:63` | `p8f_coefficients` | 対象・変数を区別（下記） | `double x=E/m.p.constants[0],L=log1p(2*x),RE=m.p.constants[3],one=1+2*x;` | 表の項だけを置換。変数の項は保持。 |
| `p8f_materials.h:74` | `p8f_coefficients` | 対象・変数を区別（下記） | `double u=log(E/m.photo_E[i])/log(m.photo_E[i+1]/m.photo_E[i]);` | 表の項だけを置換。変数の項は保持。 |
| `p8f_materials.h:75` | `p8f_coefficients` | 対象・変数を区別（下記） | `pe=exp(log(m.photo[left])+u*log(m.photo[right]/m.photo[left]));` | 表の項だけを置換。変数の項は保持。 |
| `p8f_materials.h:83` | `p8f_quantities` | 保持 | `if(T>.01 && m.ncollision)q[2]=interp(log(T),m.log_stop_T,m.stop_rad,m.nstop)-q[1];` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_materials.h:86` | `p8f_sample_k` | 保持 | `double q[4];p8f_quantities(T,m,q);double lo=log(.01),width=log(T/.01);*excess=0;` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_materials.h:91` | `p8f_shell_distribution` | 保持 | `double l=log(E*1000),total=0.;` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_kernel.h:60` | `p8f_history` | 保持 | `if(injected){stack[0]=*injected;if(stack[0].a[19]!=0){double u=rng_uniform(s);stack[0].a[22]=-log(u);}if(stack[0].a[19]==1){double u=rng_uniform(s);stack[0].a[21]=-log(u);}}` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_kernel.h:82` | `p8f_history` | 保持 | `double flight_u=rng_uniform(s),tau=-log(flight_u),flight=tau/mu,length=p5_min(flight,bb);` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_kernel.h:123` | `p8f_history` | 保持 | `if(first.a[19]==1) {double u=rng_uniform(s);first.a[21]=-log(u);}` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_kernel.h:124` | `p8f_history` | 保持 | `if(second.a[19]==1) {double u=rng_uniform(s);second.a[21]=-log(u);}` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_kernel.h:193` | `p8f_history` | 保持 | `p5_energy_init(&state,s);double u=rng_uniform(s);a[15]=-log(u);if(a[6]<=2*.01)continue;` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_kernel.h:204` | `p8f_history` | 保持 | `p5_energy_init(&state,s);double T=a[6],u=rng_uniform(s);a[22]=-log(u);if(T<=.01)continue;` | 粒子状態・乱数に依存。log1p も対象外。 |
| `p8f_kernel.h:219` | `p8f_history` | 保持 | `double T=a[6];if(T<=.01) {double u=rng_uniform(s);a[21]=-log(u);continue;}` | 粒子状態・乱数に依存。log1p も対象外。 |

## 故障の固定位置と検出

- f1 (`P8G_FAULT=1`): 材料0（水）、ID11 photo、添え字0の初期化値を nextafter(value,+inf) で1 ulpずらす。0e の photo_E[0] より下の入力で exp(log(photo[0])) が読まれる。0b と0eで拒否。
- f2 (`P8G_FAULT=2`): 材料0（水）のID17 shape_z の先頭ポインタだけを隣の材料1（肺）のID17にずらす。添え字1を到達カウンターで数える。0eで拒否。gate1での拒否はClaudeが実行。
- f3 (`P8G_FAULT=3`): 材料0のID17の添え字1の通常読み出しを length にずらす。通常の p8g_read の範囲確認がアクセス前に拒否しNaNを返す。検査ビルドの放射サンプラーはNaNで余剰フラグを立て、既存の失敗経路で停止する（無限の棄却ループを防ぐ）。同じ2000履歴/start=7000の無効読み出しで拒否。

故障の到達専用カウンターは各材料の37番目。通常ビルドはカウンターを更新しない。
初期化・輸送は同じTU/フラグ。0bの比較相手は初期化関数とは独立した `p8g_reference_value` の元のスカラー式。
補助記録の5列は 値・商・q・q³・log引数（合否は1列目の元式とのバイト照合のみ）。

## 2000履歴で読み出しが0の配列

`gate0_reads.json` が種類別・材料別の生の数を記録する。ID12/13/14（吸収端専用）は水・肺・骨で0、鉛では正数。
この履歴範囲の光子がそれら3材料の吸収端近傍の枝に入らないため。0eでは全材料で吸収端・その両隣を直接与えて枝を照合する。
0bは未参照の有効要素もすべて照合する。

## 差分の読み方

`original_copy.diff` は元の関数名・型名を新規側に揃えた比較。P5のコピーは表のlog項をreadに替える差分のみ（追加の材料引数を除く）。
材料ヘッダの新規structと初期化は p8g_logs.h に分離。P8fのP5構造体は元ヘッダからinclude。
P8gの輸送ループはP8fのコピーで、関数名・材料記述子幅（132→136）と追加材料型を変更した。
検査ビルドのNaN停止以外に変数logやexpの変更はない。
